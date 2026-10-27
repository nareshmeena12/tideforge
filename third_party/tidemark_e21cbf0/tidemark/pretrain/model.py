"""The twin: Tide's causal patch decoder with a pluggable TIME attention (LEAN_PLAN 4.1/4.2).

Ported from reference/tide_v5_100m_300b.py section 2b (CausalPatchDecoder, _causal_stats,
_monotone_quantiles, the dense next-patch pinball loss with the multi-patch share,
predict_horizon). Changes, all identical in every arm:
  * the time attention is the arm's mixer (softmax | gla | swa8 | swa16) behind the one
    interface (models/attention.py); its pre-norm and residual sit in TwinBlock, as in
    models.backbone.Block (same maths as Tide's TimeSelfAttention);
  * attention-weight dropout in the time attention is 0 (decision 2026-10-08; GLA has no
    equivalent); dropout stays in the FFN and the group attention;
  * packed rows reach the mixers as one models.packing.Pack (masks for softmax / SWA,
    cu_seqlens for GLA) instead of a dense attention mask;
  * the time mixer runs outside torch.compile (fla's kernels cannot be traced); everything
    else may be compiled (owner, 2026-10-15).
Group attention (LaneGroupAttention, 8 lanes), role embeddings, the known-covariate
lookahead, the patch embedding, causal statistics, head and loss are the reference's.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tidemark.models.backbone import FeedForward, rms_norm
from tidemark.models.gdn import GDNMixer
from tidemark.models.gla import GLAMixer
from tidemark.models.group import LaneGroupAttention
from tidemark.models.packing import Pack
from tidemark.models.softmax import SoftmaxMixer
from tidemark.models.swa import SWAMixer
from tidemark.pretrain.arms import parse_kind
from tidemark.pretrain.config import TwinConfig
from tidemark.pretrain.generators import ROLE_KNOWN, ROLE_TARGET


def build_twin_mixer(cfg: TwinConfig, gla_backend: str = "fla", packed_kernel: str = "chunk",
                     layer: int = 0) -> nn.Module:
    """The time mixer of layer `layer` (0-based), from the per-layer pattern (pretrain.arms)."""
    m = cfg.model
    kind, window = parse_kind(cfg.kinds[layer])
    if kind == "softmax":
        return SoftmaxMixer(m.d_model, m.n_heads)
    if kind == "gla":
        return GLAMixer(m.d_model, m.n_heads, m.gla_expand_k, m.gla_expand_v, m.gla_gate_rank,
                        backend=gla_backend, packed_kernel=packed_kernel)
    if kind == "gdn":
        return GDNMixer(m.d_model, m.n_heads, backend="reference" if gla_backend == "reference" else "fla")
    return SWAMixer(m.d_model, m.n_heads, window, masked=True)


@torch.compiler.disable
def _time_mix(mixer: nn.Module, h: torch.Tensor, pack):
    return mixer(h, pack)


@torch.compiler.disable
def _pack(seg_patch: torch.Tensor) -> Pack:
    return Pack(seg_patch)


def _monotone_quantiles(o):
    """(..., n_q) raw -> increasing quantiles: q0 = o0, q_k = q_{k-1} + softplus(o_k)."""
    return torch.cat([o[..., :1], o[..., :1] + torch.cumsum(F.softplus(o[..., 1:]), dim=-1)], dim=-1)


def _causal_stats(x, obs, seg_pt, seg_start_pt, n_slots, std_floor=1e-3, patch_len=32):
    """Running mean / std of OBSERVED points of the same series, from the series' first point
    up to and including t. x, obs: (R, W); seg_pt: (R, W) global series id; seg_start_pt:
    (R, W) column where that series starts in its row (kept for API compatibility).
    Returns mu, sd, count (R, W). Ported as written (float64, segmented scan: each series'
    statistics depend only on its own values, packed or alone)."""
    R, W = x.shape
    dev = x.device
    pl = patch_len
    P = W // pl
    col = torch.arange(W, device=dev).expand(R, W)
    first = torch.full((n_slots,), W - 1, device=dev, dtype=torch.long).scatter_reduce(
        0, seg_pt.reshape(-1), torch.where(obs, col, torch.full_like(col, W - 1)).reshape(-1),
        reduce="amin", include_self=True)
    shift = torch.gather(x, 1, first[seg_pt]).double()
    o = obs.double()
    xs = torch.where(obs, x.double() - shift, torch.zeros((), device=dev, dtype=torch.float64))

    v = torch.stack([o, xs, xs * xs], dim=-1).view(R, P, pl, 3)
    local = v.cumsum(2)                                              # own patch only
    segp = seg_pt[:, ::pl]                                           # (R, P) series of each patch
    pidx = torch.arange(P, device=dev)
    earlier = ((segp.unsqueeze(2) == segp.unsqueeze(1))
               & (pidx.view(1, P, 1) > pidx.view(1, 1, P))).double()  # [r, p, q]: q<p, same series
    before = torch.bmm(earlier, local[:, :, -1, :])                  # (R, P, 3)
    cs = (local + before.unsqueeze(2)).view(R, W, 3)
    n, s1, s2 = cs[..., 0], cs[..., 1], cs[..., 2]

    nc = n.clamp(min=1.0)
    m = s1 / nc
    var = (s2 / nc - m * m).clamp(min=0.0)
    has = n > 0
    mu = torch.where(has, m + shift, torch.zeros_like(m)).float()
    sd = var.sqrt().float().clamp(min=std_floor)
    return mu, sd, n.float()


class TwinBlock(nn.Module):
    """time attention (the arm) -> group attention -> feed-forward."""

    def __init__(self, cfg: TwinConfig, gla_backend: str = "fla", packed_kernel: str = "chunk", layer: int = 0):
        super().__init__()
        m = cfg.model
        self.eps = m.eps
        self.time_norm_w = nn.Parameter(torch.ones(m.d_model))
        self.mixer = build_twin_mixer(cfg, gla_backend, packed_kernel, layer)
        self.group = LaneGroupAttention(m.d_model, m.d_kv, m.n_heads, m.dropout, m.eps)
        self.ffn = FeedForward(m.d_model, m.d_ff, m.dropout, m.eps)

    def forward(self, x, pack=None, lanes=1, gmask=None):
        x = x + _time_mix(self.mixer, rms_norm(x, self.time_norm_w, self.eps), pack)
        x = self.group(x, lanes, gmask)
        return self.ffn(x)


class TwinDecoder(nn.Module):
    is_decoder = True

    def __init__(self, cfg: TwinConfig, gla_backend: str = "fla", packed_kernel: str = "chunk"):
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        pl, d, d_ff = m.patch_len, m.d_model, m.d_ff
        self.patch_len = pl
        self.horizon_loss_share = m.horizon_loss_share
        self.group_attn = True
        self.group_lanes = m.group_lanes
        self.pred_len = pl
        self.quantile_levels = list(m.quantile_levels)
        self.n_q = len(m.quantile_levels)
        self.mask_channel = True
        self.arcsinh = m.arcsinh
        self.eps = m.eps
        self.max_pass_patches = m.max_pass_patches
        # residual patch embedding (Chronos-2 / t0 style): values + validity mask channels
        self.emb_w1 = nn.Linear(2 * pl, d_ff)
        self.emb_w2 = nn.Linear(d_ff, d)
        self.emb_skip = nn.Linear(2 * pl, d)
        self.blocks = nn.ModuleList(TwinBlock(cfg, gla_backend, packed_kernel, i) for i in range(m.n_layers))
        self.role_emb = nn.Embedding(3, d)
        nn.init.normal_(self.role_emb.weight, std=0.02)
        self.look_w1 = nn.Linear(2 * pl, d_ff)
        self.look_w2 = nn.Linear(d_ff, d)
        self.look_skip = nn.Linear(2 * pl, d)
        self.final_norm_w = nn.Parameter(torch.ones(d))
        self.head = nn.Sequential(nn.Linear(d, d_ff), nn.GELU(), nn.Linear(d_ff, pl * self.n_q))

    def mixers(self) -> list[nn.Module]:
        return [b.mixer for b in self.blocks]

    def _fwd_tf(self, z):
        return torch.asinh(z) if self.arcsinh else z

    def _inv_tf(self, o):
        return torch.sinh(o.clamp(-20.0, 20.0)) if self.arcsinh else o

    def _core(self, x, obs, seg_pt, seg_start_pt, n_slots, pack=None, lanes=1, gid=None, role=None):
        """Quantiles for the NEXT patch at every position, in the normalised space of the stats
        frozen at that patch's end: q (R, P, pl, nq); and mu_e, sd_e, cnt_e (R, P). pack=None:
        one series per row (causal); else the packed geometry."""
        R, W = x.shape
        pl = self.patch_len
        P = W // pl
        mu, sd, cnt = _causal_stats(x, obs, seg_pt, seg_start_pt, n_slots, patch_len=pl)
        z = torch.where(obs, self._fwd_tf((x.float() - mu) / sd), torch.zeros((), device=x.device))
        u = torch.cat([z.view(R, P, pl), obs.view(R, P, pl).to(z.dtype)], dim=-1).to(x.dtype)
        h = self.emb_w2(F.relu(self.emb_w1(u))) + self.emb_skip(u)
        if role is None:
            h = h + self.role_emb.weight[ROLE_TARGET].to(h.dtype)
        else:
            segp = seg_pt[:, ::pl]
            same = F.pad(segp[:, 1:] == segp[:, :-1], (0, 1))            # next patch = same series
            un = torch.cat([u[:, 1:], torch.zeros_like(u[:, :1])], 1) * same.unsqueeze(-1).to(u.dtype)
            look = self.look_w2(F.relu(self.look_w1(un))) + self.look_skip(un)
            known = (role == ROLE_KNOWN).unsqueeze(-1).to(h.dtype)
            h = h + self.role_emb(role).to(h.dtype) + look * known
        if gid is None:
            lanes = 1
            gmask = torch.ones(R * P, 1, 1, 1, dtype=torch.bool, device=x.device)
        else:
            g = gid.view(R // lanes, lanes, P).transpose(1, 2).reshape((R // lanes) * P, lanes)
            gmask = (g.unsqueeze(2) == g.unsqueeze(1)).unsqueeze(1)
        for blk in self.blocks:
            h = blk(h, pack, lanes=lanes, gmask=gmask)
        h = rms_norm(h, self.final_norm_w, self.eps)
        q = _monotone_quantiles(self.head(h).float().view(R, P, pl, self.n_q))
        e = pl - 1
        return q, mu[:, e::pl], sd[:, e::pl], cnt[:, e::pl]

    def _packed_geometry(self, seg_patch, n_slots):
        """seg_pt, seg_start_pt (R, W) and the Pack of a packed batch."""
        R, P = seg_patch.shape
        pl = self.patch_len
        pidx = torch.arange(P, device=seg_patch.device).expand(R, P)
        start_patch = torch.full((n_slots,), P, device=seg_patch.device, dtype=torch.long).scatter_reduce(
            0, seg_patch.reshape(-1), pidx.reshape(-1), reduce="amin", include_self=True)
        seg_pt = seg_patch.repeat_interleave(pl, dim=1)
        seg_start_pt = start_patch[seg_pt].clamp(max=P - 1) * pl
        return seg_pt, seg_start_pt, _pack(seg_patch)

    def forward(self, x, target=None, mask=None, seg_patch=None, slot_valid=None, blank=None,
                lanes=1, gid=None, role=None, slot_loss=None):
        """PACKED (training): x, mask (R, W); seg_patch (R, P); slot_valid (S,); blank (R, W)
        = contiguous-patch-masked input points (hidden from the model, still used as targets);
        lanes / gid (R, P) / role (R, P) = group structure; slot_loss (S,) = series that carry
        loss (known covariates do not: their future is given, not forecast).
        UNPACKED: x, mask (B, L) one left-padded series per row -> next-patch quantiles in
        data scale at every position: quantile_preds_all (B, P, pl, nq). With lanes/gid/role the
        B rows are blocks of `lanes` aligned variates (a group run on its own)."""
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        if seg_patch is None:
            B, L = x.shape
            seg_pt = torch.arange(B, device=x.device).unsqueeze(1).expand(B, L)
            q, mu_e, sd_e, _ = self._core(x, mask, seg_pt, torch.zeros_like(seg_pt), B, None,
                                          lanes=lanes, gid=gid, role=role)
            return {"quantile_preds_all": self._inv_tf(q) * sd_e[..., None, None] + mu_e[..., None, None]}

        R, W = x.shape
        pl = self.patch_len
        P = W // pl
        S = slot_valid.shape[0]
        obs_in = mask & ~blank if blank is not None else mask
        seg_pt, seg_start_pt, pack = self._packed_geometry(seg_patch, S)
        q, mu_e, sd_e, cnt_e = self._core(x, obs_in, seg_pt, seg_start_pt, S, pack,
                                          lanes=lanes, gid=gid, role=role)

        # dense next-patch loss: patch p predicts patch p+1 of the SAME series, whenever
        # patch p's input had >= 1 observed point. Targets use the TRUE mask (blanked
        # points are hidden from the input but are still valid targets).
        y = x.float().view(R, P, pl)[:, 1:]
        ym = mask.view(R, P, pl)[:, 1:]
        ok_b = (seg_patch[:, 1:] == seg_patch[:, :-1]) & (cnt_e[:, :-1] > 0)
        tn = self._fwd_tf((y - mu_e[:, :-1, None]) / sd_e[:, :-1, None])
        qq = q[:, :-1]
        lv = torch.tensor(self.quantile_levels, device=x.device, dtype=qq.dtype)
        err = tn.unsqueeze(-1) - qq
        pin = torch.maximum(lv * err, (lv - 1) * err).mean(-1)             # (R, P-1, pl)
        w = (ym & ok_b.unsqueeze(-1)).float()
        seg_b = seg_patch[:, :-1].reshape(-1)
        pw, ww = (pin * w).sum(-1), w.sum(-1)                              # (R, P-1)
        zS = lambda: torch.zeros(S, device=x.device)                       # noqa: E731
        if blank is not None:       # split by what the forecast was made FROM
            bl = blank.view(R, P, pl).all(-1)[:, :-1].float()              # input patch hidden
        else:
            bl = torch.zeros_like(ww)
        num_m = zS().index_add_(0, seg_b, (pw * bl).reshape(-1))           # multi-patch forecasts
        den_m = zS().index_add_(0, seg_b, (ww * bl).reshape(-1))
        num_o = zS().index_add_(0, seg_b, (pw * (1 - bl)).reshape(-1))     # one-patch forecasts
        den_o = zS().index_add_(0, seg_b, (ww * (1 - bl)).reshape(-1))
        per_o, per_m = num_o / den_o.clamp(min=1.0), num_m / den_m.clamp(min=1.0)
        has_o, has_m = den_o > 0, den_m > 0
        den = den_o + den_m
        if self.horizon_loss_share is None:
            per = (num_o + num_m) / den.clamp(min=1.0)                     # v4: plain mean
        else:
            a = float(self.horizon_loss_share)
            per = torch.where(has_o & has_m, (1 - a) * per_o + a * per_m, torch.where(has_m, per_m, per_o))
        has = (den > 0) & slot_valid
        if slot_loss is not None:
            has = has & slot_loss
        hf = has.float()
        nan = torch.full_like(per, float("nan"))
        return {"loss": (per * hf).sum() / hf.sum().clamp(min=1.0),
                "loss_sum": (per * hf).sum(), "loss_n": hf.sum(),
                "per_series_loss": torch.where(has, per, nan),
                "_per_one": torch.where(has & has_o, per_o, nan),
                "_per_multi": torch.where(has & has_m, per_m, nan),
                "_q": q, "_mu_e": mu_e, "_sd_e": sd_e}

    @torch.no_grad()
    def predict_horizon(self, ctx, mask, horizon):
        """ctx, mask (B, L): left-padded, L a multiple of patch_len. Appends ceil(horizon/pl)
        blank patches and reads every horizon patch from ONE forward pass. (B, horizon, nq)."""
        B, L = ctx.shape
        pl = self.patch_len
        hp = -(-horizon // pl)
        xx = torch.cat([ctx, torch.zeros(B, hp * pl, device=ctx.device, dtype=ctx.dtype)], 1)
        mm = torch.cat([mask, torch.zeros(B, hp * pl, device=ctx.device, dtype=torch.bool)], 1)
        allq = self.forward(xx, mask=mm)["quantile_preds_all"]            # (B, P, pl, nq)
        pc = L // pl
        return allq[:, pc - 1:pc - 1 + hp].reshape(B, hp * pl, self.n_q)[:, :horizon]


def build_twin(cfg: TwinConfig, gla_backend: str = "fla", packed_kernel: str = "chunk") -> TwinDecoder:
    return TwinDecoder(cfg, gla_backend, packed_kernel)


def restrict_window(model: TwinDecoder, window: int | None) -> TwinDecoder:
    """Local-window test (LEAN_PLAN 4.6): limit every SOFTMAX time-attention layer to `window`
    patches (itself and window - 1 before it) at inference only; None restores full attention."""
    n = 0
    for m in model.mixers():
        if isinstance(m, SoftmaxMixer):
            m.window = window
            n += 1
    if not n:
        raise ValueError("restrict_window applies to softmax twins only")
    return model


def param_report(model: TwinDecoder) -> dict:
    """Exact counts (G1 rule, plan section 3) for a twin; same fields as utils.params."""
    from tidemark.utils.params import count
    groups = {"main": 0, "gate": 0, "norm": 0, "other": 0}
    for m in model.mixers():
        for k, v in m.param_groups().items():
            groups[k] = groups.get(k, 0) + v
    mixer_total = sum(count(m) for m in model.mixers())
    total = count(model)
    conv = groups.get("conv", 0)
    rep_ = {"params_total": total, "params_time_mixer": mixer_total,
            "params_outside_mixer": total - mixer_total, "params_main_proj": groups["main"],
            "params_gla_gate": groups["gate"], "params_gla_norm": groups["norm"],
            "params_unclassified": mixer_total - groups["main"] - groups["gate"] - groups["norm"] - conv}
    if conv:                                  # only Gated DeltaNet has short convolutions
        rep_["params_gdn_conv"] = conv
    return rep_
