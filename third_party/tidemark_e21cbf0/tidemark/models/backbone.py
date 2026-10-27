"""Backbone, identical in every arm. Ported from reference/tide_v5_pretrain_1T.py:
rms_norm and FeedForward as written; PatchEmbed is CausalPatchDecoder's residual patch
embedding (values + validity-mask channels). Only the time mixer differs between arms.

Differences from Tide (approved, plan section 3):
  * per-window instance norm over observed points, reversed on the output
    (no asinh, no causal running statistics: the head reads only the last token);
  * a direct linear head instead of the next-patch quantile head: on the last token
    ("last") or over all token states ("flatten", PatchTST style);
  * no role embedding / known-covariate lookahead (every channel is a target).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tidemark.models.attention import build_time_mixer
from tidemark.models.group import LaneGroupAttention
from tidemark.utils.config import ModelConfig


def rms_norm(x, weight, eps):
    return F.rms_norm(x, (x.shape[-1],), weight.to(x.dtype), eps)


def window_stats(x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-window mean and std over observed points, (R, 1) each (as Tide's unpacked path).
    Used by the instance norm and by the C3 loss scale."""
    x = torch.where(mask, x, torch.zeros_like(x))
    maskf = mask.to(x.dtype)
    cnt = maskf.sum(dim=1, keepdim=True).clamp(min=1)
    mean = (x * maskf).sum(dim=1, keepdim=True) / cnt
    var = ((x - mean) ** 2 * maskf).sum(dim=1, keepdim=True) / cnt
    return mean, var.clamp(min=1e-6).sqrt()


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, eps):
        super().__init__()
        self.norm_w = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        h = rms_norm(x, self.norm_w, self.eps)
        return x + self.drop(self.wo(self.act(self.wi(h))))


class PatchEmbed(nn.Module):
    """residual patch embedding (Chronos-2 / t0 style): values + validity mask channels"""

    def __init__(self, patch_len: int, d_model: int, d_ff: int):
        super().__init__()
        self.emb_w1 = nn.Linear(2 * patch_len, d_ff)
        self.emb_w2 = nn.Linear(d_ff, d_model)
        self.emb_skip = nn.Linear(2 * patch_len, d_model)

    def forward(self, u):
        return self.emb_w2(F.relu(self.emb_w1(u))) + self.emb_skip(u)


class Block(nn.Module):
    """time attention -> (group attention) -> feed-forward. The pre-norm and residual of the
    time attention live here (moved out of Tide's TimeSelfAttention), so they are identical
    in every arm."""

    def __init__(self, cfg: ModelConfig, gla_backend: str = "fla", layer: int = 0):
        super().__init__()
        d, eps = cfg.d_model, cfg.eps
        self.eps = eps
        self.time_norm_w = nn.Parameter(torch.ones(d))
        self.mixer = build_time_mixer(cfg, gla_backend, layer)
        self.group = (LaneGroupAttention(d, d // cfg.n_heads, cfg.n_heads, cfg.dropout, eps)
                      if cfg.group_attn else None)
        self.ffn = FeedForward(d, cfg.ffn_ratio * d, cfg.dropout, eps)

    def forward(self, x, lanes: int = 1):
        x = x + self.mixer(rms_norm(x, self.time_norm_w, self.eps))
        if self.group is not None:
            x = self.group(x, lanes, None)
        return self.ffn(x)

    # ---- streaming (E3). Group attention is left out of E2 / E3: it is identical in every
    # arm and its cost does not grow with the context length.
    def _no_group(self):
        if self.group is not None:
            raise NotImplementedError("streaming is defined without group attention")

    def prefill(self, x, capacity: int):
        self._no_group()
        y, state = self.mixer.prefill(rms_norm(x, self.time_norm_w, self.eps), capacity)
        return self.ffn(x + y), state

    def step(self, x_t, state):
        self._no_group()
        y, state = self.mixer.step(rms_norm(x_t, self.time_norm_w, self.eps), state)
        return self.ffn(x_t + y), state


class Forecaster(nn.Module):
    """x, mask: (R, L) raw values / validity, L a multiple of patch_len. With group attention,
    rows come in blocks of `lanes` consecutive rows (the channels of one window).
    Returns the forecast (R, horizon) in the input scale."""

    def __init__(self, cfg: ModelConfig, gla_backend: str = "fla"):
        super().__init__()
        self.cfg = cfg
        d_ff = cfg.ffn_ratio * cfg.d_model
        self.patch_len = cfg.patch_len
        self.eps = cfg.eps
        self.embed = PatchEmbed(cfg.patch_len, cfg.d_model, d_ff)
        self.blocks = nn.ModuleList(Block(cfg, gla_backend, i) for i in range(cfg.n_layers))
        self.final_norm_w = nn.Parameter(torch.ones(cfg.d_model))
        self.head_kind = cfg.head
        n_in = cfg.d_model * (cfg.head_tokens if cfg.head == "flatten" else 1)
        self.head = nn.Linear(n_in, cfg.horizon)

    def embed_tokens(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        """Instance norm and patch embedding: (token states (R, P, D), mean, std)."""
        R, L = x.shape
        pl = self.patch_len
        if L % pl:
            raise ValueError(f"length {L} is not a multiple of the patch length {pl}")
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        # per-window instance norm over observed points (as Tide's unpacked path)
        mean, std = window_stats(x, mask)
        z = torch.where(mask, (x - mean) / std, torch.zeros_like(x))
        P = L // pl
        u = torch.cat([z.view(R, P, pl), mask.to(x.dtype).view(R, P, pl)], dim=-1)
        return self.embed(u), mean, std

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, lanes: int = 1):
        R = x.shape[0]
        h, mean, std = self.embed_tokens(x, mask)
        P = h.shape[1]
        for blk in self.blocks:
            h = blk(h, lanes)
        h = rms_norm(h, self.final_norm_w, self.eps)
        if self.head_kind == "flatten":
            if P != self.cfg.head_tokens:
                raise ValueError(f"flatten head built for {self.cfg.head_tokens} tokens, got {P}")
            y = self.head(h.reshape(R, -1)).float()
        else:
            y = self.head(h[:, -1]).float()
        return y * std.float() + mean.float()

    def forecast_from_tokens(self, h: torch.Tensor, mean: torch.Tensor, std: torch.Tensor,
                             lanes: int = 1) -> torch.Tensor:
        """Everything after embed_tokens: token states h (R, P, D) -> forecast (R, horizon) in
        the input scale. Same maths as forward (a test checks they agree); exists so A6 can
        drop or shuffle old tokens AFTER the window's instance norm, without touching forward."""
        R, P, _ = h.shape
        for blk in self.blocks:
            h = blk(h, lanes)
        h = rms_norm(h, self.final_norm_w, self.eps)
        if self.head_kind == "flatten":
            if P != self.cfg.head_tokens:
                raise ValueError(f"flatten head built for {self.cfg.head_tokens} tokens, got {P}")
            y = self.head(h.reshape(R, -1)).float()
        else:
            y = self.head(h[:, -1]).float()
        return y * std.float() + mean.float()

    def mixers(self) -> list[nn.Module]:
        return [b.mixer for b in self.blocks]
