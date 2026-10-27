"""Group attention, ported as written from reference/tide_v5_pretrain_1T.py.
Identical in every arm. Here a group is the channels of one multivariate
window: rows come in blocks of `lanes` consecutive rows (one per channel)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tidemark.models.softmax import FusedQKV


def rms_norm(x, weight, eps):
    return F.rms_norm(x, (x.shape[-1],), weight.to(x.dtype), eps)


class LaneGroupAttention(nn.Module):
    """Attention ACROSS the variates of a group at the SAME patch position.

    Rows come in blocks of `lanes` consecutive rows that share one layout, so the variates of
    a group occupy the same patch columns in different rows. x (R, P, D) is viewed as
    (R/lanes blocks, P, lanes, D) and attention runs over the `lanes` axis only; `gmask`
    (blocks*P, 1, lanes, lanes) lets a token attend exactly to the tokens with its group id
    (itself included, so a single series is a group of one). No positional encoding: a group
    is a set. Cost is O(R * P * lanes) -- negligible next to time attention.

    Causality is preserved: a token at patch p only receives tokens at patch p, each of which
    has only seen its own series up to p."""

    def __init__(self, d_model, d_kv, n_heads, dropout, eps):
        super().__init__()
        self.qkv = FusedQKV(d_model, d_kv, n_heads)
        self.out_proj = nn.Linear(n_heads * d_kv, d_model, bias=False)
        self.norm_w = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.dropout_p = dropout

    def forward(self, x, lanes, gmask):
        R, P, D = x.shape
        Bk = R // lanes
        h = rms_norm(x, self.norm_w, self.eps)
        h = h.view(Bk, lanes, P, D).transpose(1, 2).reshape(Bk * P, lanes, D)
        q, k, v = self.qkv(h)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=gmask,
                                           dropout_p=self.dropout_p if self.training else 0.0)
        o = o.transpose(1, 2).reshape(Bk, P, lanes, -1).transpose(1, 2).reshape(R, P, -1)
        return x + self.out_proj(o)
