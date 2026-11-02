"""Covariate adapter (owner 2026-11-03): cross-lane attention blocks on a frozen backbone, TimeH2O-style.

    adapter_every: 0 (none) | 1 (after every decoder block) | 2 (after every 2nd)

A block is a LaneGroupAttention without the sigmoid gate: RMSNorm in, attention across the lanes of a group at one patch, output
projection zero-initialised, so the adapted model equals the backbone at init. Its own firewall mask:
    gid[q] == gid[k]  &  m1.allow_table()[role[q], role[k]]     (target reads target / past / known; past reads past / known; known
                                                                 reads known)
A token changes only if its group has ANOTHER lane that is a past or known covariate; any other token keeps x exactly (torch.where),
so univariate series and groups of targets alone equal the backbone by construction, trained or not.

Limit: R1.2 and R1.4 have no M1. The firewall covers the adapter's path only; the frozen group attention stays as trained, so a
covariate token can already carry target information when the adapter reads it (owner 2026-11-03).

install() swaps the classes (as tideforge.model.gated; ft3_model.py is generated). _core stores gid, role in a dict the blocks share.
The forward of the backbone is unchanged. forecast_left_padded passes no role, so it skips the adapter.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from tideforge.model import ft3_model as FM
from tideforge.model.ft3_model import ROLE_TARGET
from tideforge.model.m1 import allow_table

ADAPTER_EVERY = (0, 1, 2)


class CovAdapter(FM.LaneGroupAttention):
    def __init__(self, d_model, d_kv, n_heads, eps):
        super().__init__(d_model, d_kv, n_heads, 0.0, eps)
        torch.nn.init.zeros_(self.out_proj.weight)

    def masks(self, role, gid, lanes):
        """(Bk * P, 1, lanes, lanes) attention mask, (Bk * P, lanes) tokens that may change."""
        R, P = role.shape
        lay = lambda t: t.view(R // lanes, lanes, P).transpose(1, 2).reshape((R // lanes) * P, lanes)   # noqa: E731
        r, g = lay(role), lay(gid)
        same = g.unsqueeze(2) == g.unsqueeze(1)
        gmask = allow_table().to(role.device)[r.unsqueeze(2), r.unsqueeze(1)] & same
        other = ~torch.eye(lanes, dtype=torch.bool, device=role.device)
        active = (same & other & (r != ROLE_TARGET).unsqueeze(1)).any(-1)
        return gmask.unsqueeze(1), active

    def forward(self, x, lanes, role, gid):
        R, P, D = x.shape
        Bk = R // lanes
        gmask, active = self.masks(role, gid, lanes)
        h = FM.rms_norm(x, self.norm_w, self.eps)
        h = h.view(Bk, lanes, P, D).transpose(1, 2).reshape(Bk * P, lanes, D)
        q, k, v = self.qkv(h)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=gmask)
        o = o.transpose(1, 2).reshape(Bk, P, lanes, -1).transpose(1, 2).reshape(R, P, -1)
        keep = active.view(Bk, P, lanes).transpose(1, 2).reshape(R, P, 1)
        return torch.where(keep, x + self.out_proj(o), x)


class AdaptedBlock(FM.DecoderBlock):
    def forward(self, x, attn_mask=None, is_causal=False, lanes=1, gmask=None):
        x = super().forward(x, attn_mask, is_causal, lanes, gmask)
        role, gid = self.ad_ctx["role"], self.ad_ctx["gid"]
        if self.adapter is None or role is None or gid is None:
            return x
        return self.adapter(x, lanes, role, gid)


def _adapted(cls):
    class Adapted(cls):
        def _core(self, x, obs, seg_pt, seg_start_pt, n_slots, attn_mask=None, is_causal=False, lanes=1, gid=None, role=None):
            self.ad_ctx.update(role=role, gid=gid)
            return super()._core(x, obs, seg_pt, seg_start_pt, n_slots, attn_mask=attn_mask, is_causal=is_causal,
                                 lanes=lanes, gid=gid, role=role)
    return Adapted


def install(model, every: int):
    """Adds the adapter blocks to a built group model; the existing weights stay."""
    if every not in ADAPTER_EVERY[1:] or not model.group_attn:
        raise ValueError(f"adapter_every={every!r} needs a group model and one of {ADAPTER_EVERY[1:]}")
    model.ad_ctx = dict(role=None, gid=None)
    for i, blk in enumerate(model.blocks):
        a = blk.group
        blk.__class__ = AdaptedBlock
        blk.ad_ctx = model.ad_ctx
        blk.adapter = CovAdapter(a.norm_w.numel(), a.qkv.d_kv, a.qkv.n_heads, a.eps) if (i + 1) % every == 0 else None
    model.__class__ = _adapted(type(model))
    return model


def adapter_parameters(model) -> list:
    return [p for n, p in model.named_parameters() if ".adapter." in n]


def freeze_backbone(model) -> int:
    """Only adapter parameters keep requires_grad; returns their count."""
    for n, p in model.named_parameters():
        p.requires_grad_(".adapter." in n)
    return sum(p.numel() for p in adapter_parameters(model))
