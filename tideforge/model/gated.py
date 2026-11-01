"""Gated attention (attn_gate: sigmoid; Timestral's sdpa_sigmoid, owner 2026-11-02). Every time and group attention gets
    o = SDPA(q, k, v); g = sigmoid(W_g h); out = W_o(o * g)
with h the same normed input that makes q, per head and element. The gate reads only the token's own input, so causality and M1's
firewall do not change. The forwards are ports of ft3_model.TimeSelfAttention (L89-L100) and LaneGroupAttention (L396-L404); CHANGED marks
the difference. M1's masks reach them unchanged, as attn_mask / gmask.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tideforge.model import ft3_model as FM

ATTN_GATE = ("none", "sigmoid")


class GatedTime(FM.TimeSelfAttention):
    def forward(self, x, attn_mask=None, is_causal=False):
        B, T, D = x.shape
        h = FM.rms_norm(x, self.norm_w, self.eps)
        q, k, v = self.qkv(h)
        cos, sin = self.rope(T, x.device, q.dtype)
        q, k = FM.apply_rope(q, k, cos, sin)
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal and attn_mask is None,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        o = o.transpose(1, 2).reshape(B, T, -1)
        return x + self.out_proj(o * torch.sigmoid(self.w_g(h)))                 # CHANGED


class GatedGroup(FM.LaneGroupAttention):
    def forward(self, x, lanes, gmask):
        R, P, D = x.shape
        Bk = R // lanes
        h0 = FM.rms_norm(x, self.norm_w, self.eps)                               # CHANGED: kept for the gate, in x's layout
        h = h0.view(Bk, lanes, P, D).transpose(1, 2).reshape(Bk * P, lanes, D)
        q, k, v = self.qkv(h)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=gmask,
                                           dropout_p=self.dropout_p if self.training else 0.0)
        o = o.transpose(1, 2).reshape(Bk, P, lanes, -1).transpose(1, 2).reshape(R, P, -1)
        return x + self.out_proj(o * torch.sigmoid(self.w_g(h0)))                # CHANGED


def install(model):
    """Turns every attention of a built model into its gated form; the existing weights stay."""
    for blk in model.blocks:
        for a, cls in ((blk.time, GatedTime), (blk.group, GatedGroup)):
            if a is not None:
                a.__class__ = cls
                a.w_g = nn.Linear(a.norm_w.numel(), a.out_proj.in_features, bias=False)
    return model
