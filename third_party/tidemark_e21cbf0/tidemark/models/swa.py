"""SWA arm: sliding-window causal softmax attention, a true banded implementation.

Each query sees itself and the `window - 1` tokens before it. Parameters, RoPE and
projections are exactly the softmax arm's (Tide's FusedQKV / SharedRoPE); only the
attention pattern differs, so the param counts of the two arms are identical.

Blockwise local attention on the fused SDPA kernel: the sequence is cut into blocks of
`window` queries, and each block attends to its own block plus the previous one, masked
to the exact band. Cost grows with T x 2 x window (not T^2), it supports autograd (S1
trains with it), and it needs no new dependency. At small windows the blocks are much
smaller than the kernel's tiles, so wall-clock time does not track T x window there:
E2 reports analytic FLOPs next to the times.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tidemark.models.packing import band_causal
from tidemark.models.softmax import FusedQKV, SharedRoPE, apply_rope, rope_at, sdpa_backend


def band_mask(n_blocks: int, w: int, device) -> torch.Tensor:
    """(n_blocks, w, 2w) bool: query i of a block may see key j of [previous block | own block]
    iff 0 <= (w + i) - j < w; block 0 has no previous block."""
    i = torch.arange(w, device=device)[:, None]
    j = torch.arange(2 * w, device=device)[None]
    band = (j > i) & (j <= i + w)
    first = band & (j >= w)
    return torch.where(torch.arange(n_blocks, device=device)[:, None, None] == 0, first, band)


def _blockify(q, k, v, w):
    B, H, T, d = q.shape
    pad = (-T) % w
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))
    nb = (T + pad) // w
    as_blocks = lambda t: t.view(B, H, nb, w, d).permute(0, 2, 1, 3, 4)          # noqa: E731
    qb, kb, vb = as_blocks(q), as_blocks(k), as_blocks(v)                          # (B, nb, H, w, d)
    prev = lambda t: F.pad(t[:, :-1], (0, 0, 0, 0, 0, 0, 1, 0))                    # noqa: E731
    kk = torch.cat([prev(kb), kb], dim=3).reshape(B * nb, H, 2 * w, d)
    vv = torch.cat([prev(vb), vb], dim=3).reshape(B * nb, H, 2 * w, d)
    mask = band_mask(nb, w, q.device)[None].expand(B, nb, w, 2 * w).reshape(B * nb, 1, w, 2 * w)
    return qb.reshape(B * nb, H, w, d), kk, vv, mask, nb


def banded_attention(q, k, v, w: int) -> torch.Tensor:
    """q, k, v (B, H, T, d) -> (B, H, T, d); each query attends to keys in [t - w + 1, t]."""
    B, H, T, d = q.shape
    qb, kk, vv, mask, nb = _blockify(q, k, v, w)
    o = F.scaled_dot_product_attention(qb, kk, vv, attn_mask=mask)
    return o.view(B, nb, H, w, d).permute(0, 2, 1, 3, 4).reshape(B, H, nb * w, d)[:, :, :T]


class SWAMixer(nn.Module):
    """(B, T, D) -> (B, T, D). Input is already normed; no residual is added here.

    masked=True (the Milestone 4 twins only, owner 2026-10-15): masked SDPA with the band
    (p - q) < window over the whole sequence, which also takes a Pack (same series AND band).
    Same maths as the banded path (tests/test_twin_packing.py checks it); at 256 tokens the
    dense masked call is faster than 8-token blocks, and the twins make no cost claim. Every
    cost measurement (E2, E3) uses the banded path."""

    def __init__(self, d_model: int, n_heads: int, window: int, masked: bool = False):
        super().__init__()
        d_kv = d_model // n_heads
        self.qkv = FusedQKV(d_model, d_kv, n_heads)
        self.out_proj = nn.Linear(n_heads * d_kv, d_model, bias=False)
        self.rope = SharedRoPE(d_kv)
        self.n_heads, self.d_kv, self.window = n_heads, d_kv, window
        self.masked = masked

    def qkv_at(self, h: torch.Tensor, pos0: int):
        q, k, v = self.qkv(h)
        pos = torch.arange(pos0, pos0 + h.shape[1], device=h.device)
        cos, sin = rope_at(self.rope, pos, q.dtype)
        q, k = apply_rope(q, k, cos, sin)
        return q, k, v

    def forward(self, h: torch.Tensor, pack=None) -> torch.Tensor:
        B, T, _ = h.shape
        q, k, v = self.qkv_at(h, 0)
        if pack is not None:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=pack.mask(self.window))
        elif self.masked:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=band_causal(T, h.device, self.window))
        else:
            o = banded_attention(q, k, v, self.window)
        return self.out_proj(o.transpose(1, 2).reshape(B, T, -1))

    # ---- bookkeeping (identical parameters to the softmax arm)
    def param_groups(self) -> dict[str, int]:
        main = sum(p.numel() for p in self.qkv.parameters()) + self.out_proj.weight.numel()
        return {"main": main, "gate": 0, "norm": 0}

    def info(self) -> dict:
        return {"key_dim_total": self.n_heads * self.d_kv, "head_k_dim": self.d_kv,
                "head_v_dim": self.d_kv, "state_size": None, "swa_window": self.window}

    # ---- streaming with a fixed-size KV cache: a ring buffer of `window` slots (E3)
    def init_state(self, batch: int, device, dtype, capacity: int | None = None, length: int = 0) -> dict:
        """Empty ring, or one holding min(length, window) random tokens (E3: the cost of an
        update does not depend on the cached values)."""
        shape = (batch, self.n_heads, self.window, self.d_kv)
        k, v = torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)
        if length:
            k.normal_()
            v.normal_()
        return {"k": k, "v": v, "len": length}

    def prefill(self, h: torch.Tensor, capacity: int | None = None):
        B, T, _ = h.shape
        q, k, v = self.qkv_at(h, 0)
        o = banded_attention(q, k, v, self.window)
        state = self.init_state(B, h.device, k.dtype)
        keep = torch.arange(max(0, T - self.window), T, device=h.device)
        state["k"][:, :, keep % self.window] = k[:, :, keep]
        state["v"][:, :, keep % self.window] = v[:, :, keep]
        state["len"] = T
        return self.out_proj(o.transpose(1, 2).reshape(B, T, -1)), state

    def step(self, h_t: torch.Tensor, state: dict):
        B = h_t.shape[0]
        pos = state["len"]
        q, k, v = self.qkv_at(h_t, pos)
        slot = pos % self.window
        state["k"][:, :, slot:slot + 1] = k.to(state["k"].dtype)
        state["v"][:, :, slot:slot + 1] = v.to(state["v"].dtype)
        state["len"] = pos + 1
        n = min(pos + 1, self.window)              # RoPE is in the keys: slot order is irrelevant
        o = F.scaled_dot_product_attention(q, state["k"][:, :, :n].to(q.dtype),
                                           state["v"][:, :, :n].to(q.dtype))
        return self.out_proj(o.transpose(1, 2).reshape(B, 1, -1)), state

    def sdpa_backend(self, batch: int, T: int, device, dtype) -> str:
        q = torch.zeros(batch, self.n_heads, T, self.d_kv, device=device, dtype=dtype)
        qb, kk, vv, mask, _ = _blockify(q, q, q, self.window)
        return sdpa_backend(qb, kk, vv, attn_mask=mask)
