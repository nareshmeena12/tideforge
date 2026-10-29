"""M1 (EXPERIMENTS.md G1.13, owner OK 2026-10-29): known covariates see the whole window, behind a firewall. Same parameters
as the port; only the two attention masks change.

    known_attn: off | mask        known_lookahead: None (the whole segment) | K patches (total over the layers)

off   the port's _core, unchanged (tideforge.model.build returns the port itself; M1Decoder(known_attn="off") calls it too).
mask  time attention, query patch i, key patch j of one row:
          seg[i] == seg[j]  &  (j <= i  |  role[i] == KNOWN & j <= i + K)
      group attention, query lane q, key lane k at the same patch:
          gid[q] == gid[k]  &  ALLOW[role[q], role[k]]
                       target  past  known
          target         x      x     x
          past           .      x     x
          known          .      .     x
known_lookahead K is a TOTAL reach: a band of K patches in every layer would compound (a known token at t reads t+K, which
read t+2K in the layer before, ...: 4 layers x 32 = 128 patches), so the band is split over the layers, b_l = K // L (+1 in
the first K % L layers; a layer with b_l = 0 is causal for known lanes too). A token at t then never sees a known value
beyond patch t + K + 1 (the +1: the look embedding). tests/model/test_m1.py checks the bound exactly.
A target at t reads its group's known token at t, which has read its whole lane (time attention runs before group attention
in every block, so from layer 1). Firewall: a covariate token never reads a target (group rows above; time attention stays
in its own lane), so nothing a target reads carries another target's future; known lanes never read past lanes, so a past
lane's future cannot reach an earlier token through a known lane; past lanes stay causal; the segment and group-id checks keep
packed series apart. The look embedding (known token at p also embeds p+1 of its own lane) is unchanged.

Both masks are built inside _core from seg_pt, gid and role, so the packed training forward and ft3_core.forecast_groups
(which calls model._core) use them with no change to either; the attn_mask / is_causal they pass are replaced.

`breaks` is for the leak tests only (tests/model/test_m1.py): each name opens one hole, so every test can be shown to fail on
a broken mask. Never set it outside the tests.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tideforge.model import ft3_model as FM
from tideforge.model.ft3_model import ROLE_KNOWN, ROLE_PAST, ROLE_TARGET, rms_norm

KNOWN_ATTN = ("off", "mask")
BREAKS = ("target_time_full", "symmetric_group", "known_reads_past", "no_isolation")


def allow_table(breaks=()) -> torch.Tensor:
    """(3, 3) bool, [query role, key role]: which roles a token may read in group attention."""
    a = torch.zeros(3, 3, dtype=torch.bool)
    a[ROLE_TARGET, [ROLE_TARGET, ROLE_PAST, ROLE_KNOWN]] = True
    a[ROLE_PAST, [ROLE_PAST, ROLE_KNOWN]] = True
    a[ROLE_KNOWN, ROLE_KNOWN] = True
    if "symmetric_group" in breaks:
        a[:] = True
    if "known_reads_past" in breaks:
        a[ROLE_KNOWN, ROLE_PAST] = True
    return a


class M1Decoder(FM.MaskedLossDecoder):
    def __init__(self, *args, known_attn: str = "mask", known_lookahead: int | None = None, **kw):
        super().__init__(*args, **kw)
        if known_attn not in KNOWN_ATTN:
            raise ValueError(f"known_attn must be one of {KNOWN_ATTN}, not {known_attn!r}")
        if known_lookahead is not None and (known_attn == "off" or int(known_lookahead) < 1):
            raise ValueError("known_lookahead needs known_attn: mask and K >= 1")
        if known_attn != "off" and not self.group_attn:
            raise ValueError("known_attn: mask needs a group model (group_attn)")
        self.known_attn = known_attn
        self.known_lookahead = None if known_lookahead is None else int(known_lookahead)
        self.breaks: frozenset = frozenset()

    def bands(self) -> list:
        """Per layer: how many patches ahead a known token reads (None = the whole segment); they sum to known_lookahead."""
        L, K = len(self.blocks), self.known_lookahead
        if K is None:
            return [None] * L
        return [K // L + (1 if l < K % L else 0) for l in range(L)]

    def time_mask(self, seg_pt, role, band=None) -> torch.Tensor:
        """(R, 1, P, P) bool: the M1 time-attention mask (module docstring); band = this layer's known lookahead."""
        segp = seg_pt[:, ::self.patch_len]
        P = segp.shape[1]
        i = torch.arange(P, device=segp.device)
        causal = (i[None, :] <= i[:, None]).unsqueeze(0)                     # [., i, j]: j <= i
        if band is None:
            ahead = torch.ones(1, P, P, dtype=torch.bool, device=segp.device)
        else:
            ahead = (i[None, :] <= i[:, None] + band).unsqueeze(0)
        open_ = role == ROLE_KNOWN
        if "target_time_full" in self.breaks:
            open_ = open_ | (role == ROLE_TARGET)
        m = causal | (open_.unsqueeze(2) & ahead)
        if "no_isolation" not in self.breaks:
            m = m & (segp.unsqueeze(2) == segp.unsqueeze(1))
        return m.unsqueeze(1)

    def group_mask(self, gid, role, lanes) -> torch.Tensor:
        """(Bk * P, 1, lanes, lanes) bool: the M1 group-attention mask (module docstring)."""
        R, P = role.shape
        lay = lambda t: t.view(R // lanes, lanes, P).transpose(1, 2).reshape((R // lanes) * P, lanes)   # noqa: E731
        r = lay(role)
        m = allow_table(self.breaks).to(role.device)[r.unsqueeze(2), r.unsqueeze(1)]
        if "no_isolation" not in self.breaks:
            g = lay(gid)
            m = m & (g.unsqueeze(2) == g.unsqueeze(1))
        return m.unsqueeze(1)

    # >>> port of ft3_model.CausalPatchDecoder._core (tideforge/model/ft3_model.py L473-L509, = ft3_core's); CHANGED: the masks
    def _core(self, x, obs, seg_pt, seg_start_pt, n_slots, attn_mask=None, is_causal=False,
              lanes=1, gid=None, role=None):
        if self.known_attn == "off" or role is None:
            return super()._core(x, obs, seg_pt, seg_start_pt, n_slots, attn_mask=attn_mask, is_causal=is_causal,
                                 lanes=lanes, gid=gid, role=role)
        R, W = x.shape
        pl = self.patch_len
        P = W // pl
        mu, sd, cnt = FM._causal_stats(x, obs, seg_pt, seg_start_pt, n_slots, patch_len=pl)
        z = torch.where(obs, self._fwd_tf((x.float() - mu) / sd), torch.zeros((), device=x.device))
        u = torch.cat([z.view(R, P, pl), obs.view(R, P, pl).to(z.dtype)], dim=-1).to(x.dtype)
        h = self.emb_w2(F.relu(self.emb_w1(u))) + self.emb_skip(u)
        segp = seg_pt[:, ::pl]
        same = F.pad(segp[:, 1:] == segp[:, :-1], (0, 1))            # next patch = same series
        un = torch.cat([u[:, 1:], torch.zeros_like(u[:, :1])], 1) * same.unsqueeze(-1).to(u.dtype)
        look = self.look_w2(F.relu(self.look_w1(un))) + self.look_skip(un)
        known = (role == ROLE_KNOWN).unsqueeze(-1).to(h.dtype)
        h = h + self.role_emb(role).to(h.dtype) + look * known
        masks = {b: self.time_mask(seg_pt, role, b) for b in set(self.bands())}                  # CHANGED
        if gid is None:
            lanes = 1
            gmask = torch.ones(R * P, 1, 1, 1, dtype=torch.bool, device=x.device)
        else:
            gmask = self.group_mask(gid, role, lanes)                                               # CHANGED
        for blk, b in zip(self.blocks, self.bands()):
            h = blk(h, attn_mask=masks[b], is_causal=False, lanes=lanes, gmask=gmask)              # CHANGED: masks
        h = rms_norm(h, self.final_norm_w, self.eps)
        q = FM._monotone_quantiles(self.head(h).float().view(R, P, pl, self.n_q))
        e = pl - 1
        return q, mu[:, e::pl], sd[:, e::pl], cnt[:, e::pl]
    # <<< _core


def build_m1(cfg: dict, known_attn: str = "mask", known_lookahead: int | None = None) -> M1Decoder:
    """FM.build_masked_model(cfg)'s model with the M1 masks (same constructor arguments, same parameters)."""
    if cfg.get("decoder", "pool") != "causal":
        raise ValueError("M1 needs the causal decoder")
    return M1Decoder(
        patch_len=cfg["patch_len"], d_model=cfg["d_model"], d_kv=cfg["d_kv"], n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"], dropout=cfg["dropout"], arcsinh=cfg.get("arcsinh", True),
        max_pass_patches=cfg.get("max_pass_patches", 32), group_attn=bool(cfg.get("group_attn", False)),
        group_lanes=int(cfg.get("group_lanes", 8)), horizon_loss_share=cfg.get("horizon_loss_share", None),
        known_attn=known_attn, known_lookahead=known_lookahead)
