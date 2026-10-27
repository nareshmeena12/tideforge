"""ft3 model with a separate LOSS mask (owner, 2026-10-24): the reference decoder uses ONE mask for what the model sees and for what
is scored. Real series are forward-filled and their filled points are OBSERVED (mask True, as the scorers and _L_corruption do), but
the originally missing positions must not be scored. `loss_mask` (R, W) bool, True where a target counts.

MaskedLossDecoder is CausalPatchDecoder with the packed forward's target mask `mask & loss_mask` instead of `mask`: the two changed lines
are marked CHANGED below, everything else is a verbatim copy of ft3_core.CausalPatchDecoder.forward (tests/test_real_model.py compares the
sources, and checks that the loss equals the reference's exactly when loss_mask is all True). Same parameters, same state dict: ft3_main
loads into it with strict=True.
"""
from __future__ import annotations

import torch

from tidemark.real import ft3_core as C
from tidemark.real.ft3_core import CausalPatchDecoder, F, _monotone_quantiles  # noqa: F401  (F, _monotone_quantiles: names the copied body uses)


class MaskedLossDecoder(CausalPatchDecoder):
    def forward(self, x, target=None, mask=None, seg_patch=None, slot_valid=None, blank=None,
                lanes=1, gid=None, role=None, slot_loss=None, loss_mask=None):                                   # CHANGED: loss_mask
        """PACKED (training): as CausalPatchDecoder.forward, plus loss_mask (R, W) bool: targets that count (None = all observed)."""
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        if seg_patch is None:
            B, L = x.shape
            seg_pt = torch.arange(B, device=x.device).unsqueeze(1).expand(B, L)
            q, mu_e, sd_e, _ = self._core(x, mask, seg_pt, torch.zeros_like(seg_pt), B, is_causal=True,
                                          lanes=lanes, gid=gid, role=role)
            return {"quantile_preds_all": self._inv_tf(q) * sd_e[..., None, None] + mu_e[..., None, None]}

        R, W = x.shape
        pl = self.patch_len
        P = W // pl
        S = slot_valid.shape[0]
        obs_in = mask & ~blank if blank is not None else mask
        seg_pt, seg_start_pt, attn = self._packed_geometry(seg_patch, S)
        q, mu_e, sd_e, cnt_e = self._core(x, obs_in, seg_pt, seg_start_pt, S, attn_mask=attn,
                                          lanes=lanes, gid=gid, role=role)

        # dense next-patch loss: patch p predicts patch p+1 of the SAME series, whenever
        # patch p's input had >= 1 observed point. Targets use the TRUE mask (blanked
        # points are hidden from the input but are still valid targets).
        y = x.float().view(R, P, pl)[:, 1:]
        ym = (mask if loss_mask is None else mask & loss_mask).view(R, P, pl)[:, 1:]                        # CHANGED: loss mask
        ok_b = (seg_patch[:, 1:] == seg_patch[:, :-1]) & (cnt_e[:, :-1] > 0)
        tn = self._fwd_tf((y - mu_e[:, :-1, None]) / sd_e[:, :-1, None])
        qq = q[:, :-1]
        lv = torch.tensor(self.quantile_levels, device=x.device, dtype=qq.dtype)
        err = tn.unsqueeze(-1) - qq
        pin = torch.maximum(lv * err, (lv - 1) * err).mean(-1)             # (R, P-1, pl)
        w = (ym & ok_b.unsqueeze(-1)).float()
        seg_b = seg_patch[:, :-1].reshape(-1)
        pw, ww = (pin * w).sum(-1), w.sum(-1)                              # (R, P-1)
        zS = lambda: torch.zeros(S, device=x.device)
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
                "per_series_loss": torch.where(has, per, nan),
                "_per_one": torch.where(has & has_o, per_o, nan),
                "_per_multi": torch.where(has & has_m, per_m, nan),
                "_q": q, "_mu_e": mu_e, "_sd_e": sd_e}


def build_masked_model(cfg: dict) -> MaskedLossDecoder:
    """The ft3 causal decoder of C.build_model(cfg), with the loss-mask forward."""
    if cfg.get("decoder", "pool") != "causal":
        raise ValueError("only the causal decoder has a loss-masked variant")
    return MaskedLossDecoder(
        patch_len=cfg["patch_len"], d_model=cfg["d_model"], d_kv=cfg["d_kv"], n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"], dropout=cfg["dropout"], arcsinh=cfg.get("arcsinh", True),
        max_pass_patches=cfg.get("max_pass_patches", 32), group_attn=bool(cfg.get("group_attn", False)),
        group_lanes=int(cfg.get("group_lanes", 8)), horizon_loss_share=cfg.get("horizon_loss_share", None))


def packed_forward_masked(model, b):
    """C.packed_forward for batches that may carry b["loss_ok"] (the loss mask)."""
    return model(b["x"], mask=b["mask"], seg_patch=b["seg_patch"], slot_valid=b["slot_valid"], blank=b["blank"],
                 lanes=b.get("lanes", 1), gid=b.get("gid"), role=b.get("role"), slot_loss=b.get("slot_loss"),
                 loss_mask=b.get("loss_ok"))
