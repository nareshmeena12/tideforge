"""Softmax arm: Tide's TimeSelfAttention, ported from
reference/tide_v5_pretrain_1T.py (section 2. MODEL).

SharedRoPE, rotate_half, apply_rope and FusedQKV are copied as written.
SoftmaxMixer is TimeSelfAttention with two changes, both approved:
  * its pre-norm and residual moved to models.backbone.Block, which is
    identical in every arm (same parameters, same maths);
  * attention-weight dropout is 0 (GLA has no equivalent; decision 2026-10-08).
Causal, as in Tide's unpacked path (is_causal=True).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tidemark.models.packing import band_causal


class SharedRoPE(nn.Module):
    """Rotary tables computed fresh from the (fixed) inv_freq buffer on every call.

    Deliberately does NOT cache cos/sin in a Python dict: that cache is fatal under
    torch.compile(mode="reduce-overhead") (CUDA-graph pool overwrites). Recomputing is
    mathematically identical and negligible in cost.
    """

    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: (B, H, T, D) ; cos,sin: (T, D)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class FusedQKV(nn.Module):
    def __init__(self, d_model, d_kv, n_heads, n_kv_heads=None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.d_kv = d_kv
        self.q_dim = n_heads * d_kv
        self.kv_dim = self.n_kv_heads * d_kv
        self.proj = nn.Linear(d_model, self.q_dim + 2 * self.kv_dim, bias=False)

    def forward(self, x):
        qkv = self.proj(x)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        B, T, _ = x.shape
        q = q.view(B, T, self.n_heads, self.d_kv).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.d_kv).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.d_kv).transpose(1, 2)
        return q, k, v


class SoftmaxMixer(nn.Module):
    """Causal softmax attention over the patch tokens, RoPE inside. (B, T, D) -> (B, T, D).
    Input is already normed; no residual is added here."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        d_kv = d_model // n_heads
        self.qkv = FusedQKV(d_model, d_kv, n_heads)
        self.out_proj = nn.Linear(n_heads * d_kv, d_model, bias=False)
        self.rope = SharedRoPE(d_kv)
        self.n_heads = n_heads
        self.d_kv = d_kv
        # local-window test (Milestone 4): attention limited to this many patches AT INFERENCE
        # ONLY (set by pretrain.model.restrict_window); None = full causal attention
        self.window: int | None = None

    def forward(self, h: torch.Tensor, pack=None) -> torch.Tensor:
        """pack (models.packing.Pack): packed rows, attention limited to the same series
        (Tide's masked SDPA). RoPE uses the row position: it is relative, so a series
        attends exactly as it would alone."""
        B, T, _ = h.shape
        q, k, v = self.qkv(h)
        cos, sin = self.rope(T, h.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)
        if pack is not None:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=pack.mask(self.window))
        elif self.window is not None:
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=band_causal(T, h.device, self.window))
        else:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, -1)
        return self.out_proj(o)

    # ---- bookkeeping for the G1 rule and the results rows
    def param_groups(self) -> dict[str, int]:
        main = sum(p.numel() for p in self.qkv.parameters()) + self.out_proj.weight.numel()
        return {"main": main, "gate": 0, "norm": 0}

    def info(self) -> dict:
        return {"key_dim_total": self.n_heads * self.d_kv, "head_k_dim": self.d_kv,
                "head_v_dim": self.d_kv, "state_size": None}

    # ---- streaming with a KV cache (E3, test 3). The cache is preallocated to `capacity`
    # so an update never copies the history; keys are stored after RoPE.
    def qkv_at(self, h: torch.Tensor, pos0: int):
        """q, k, v (B, H, T, d) for tokens at absolute positions pos0 .. pos0 + T - 1."""
        q, k, v = self.qkv(h)
        pos = torch.arange(pos0, pos0 + h.shape[1], device=h.device)
        cos, sin = rope_at(self.rope, pos, q.dtype)
        q, k = apply_rope(q, k, cos, sin)
        return q, k, v

    def init_state(self, batch: int, device, dtype, capacity: int, length: int = 0) -> dict:
        """Empty cache, or a cache that already holds `length` tokens of random keys / values
        (E3: the cost of an update depends on the cache length, not on its contents)."""
        shape = (batch, self.n_heads, capacity, self.d_kv)
        k, v = torch.zeros(shape, device=device, dtype=dtype), torch.zeros(shape, device=device, dtype=dtype)
        if length:
            k[:, :, :length].normal_()
            v[:, :, :length].normal_()
        return {"k": k, "v": v, "len": length}

    def prefill(self, h: torch.Tensor, capacity: int):
        """Full causal pass over h that also returns the cache it leaves behind."""
        B, T, _ = h.shape
        q, k, v = self.qkv_at(h, 0)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, -1)
        state = self.init_state(B, h.device, k.dtype, capacity)
        state["k"][:, :, :T], state["v"][:, :, :T], state["len"] = k, v, T
        return self.out_proj(o), state

    def step(self, h_t: torch.Tensor, state: dict):
        """One new token (B, 1, D) attends to every cached token and itself."""
        B = h_t.shape[0]
        pos = state["len"]
        q, k, v = self.qkv_at(h_t, pos)
        state["k"][:, :, pos:pos + 1] = k.to(state["k"].dtype)
        state["v"][:, :, pos:pos + 1] = v.to(state["v"].dtype)
        state["len"] = pos + 1
        o = F.scaled_dot_product_attention(q, state["k"][:, :, :pos + 1].to(q.dtype),
                                           state["v"][:, :, :pos + 1].to(q.dtype))
        return self.out_proj(o.transpose(1, 2).reshape(B, 1, -1)), state

    def sdpa_backend(self, batch: int, T: int, device, dtype) -> str:
        q = torch.zeros(batch, self.n_heads, T, self.d_kv, device=device, dtype=dtype)
        return sdpa_backend(q, q, q, is_causal=True)


def rope_at(rope: SharedRoPE, positions: torch.Tensor, dtype):
    """RoPE tables for arbitrary absolute positions: the same values as rows `positions` of
    SharedRoPE(seq_len) (same float32 outer product), without building every earlier row."""
    freqs = positions.to(torch.float32)[:, None] * rope.inv_freq.to(positions.device)[None]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def sdpa_backend(q, k, v, attn_mask=None, is_causal: bool = False) -> str:
    """Which kernel scaled_dot_product_attention picks for these shapes (logged by E2 / E3)."""
    try:
        from torch.nn.attention import SDPBackend
        choice = torch._fused_sdp_choice(q, k, v, attn_mask, 0.0, is_causal)
        return SDPBackend(choice).name
    except Exception as e:  # noqa: BLE001 - private API; never fail a benchmark over a log field
        return f"unknown ({type(e).__name__})"
