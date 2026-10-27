"""The one attention interface. Every arm is a TimeMixer: causal mixing along the
patch-token axis, (B, T, D) -> (B, T, D), input already normed, no residual.
build_time_mixer() is the only place where the arm is chosen (per layer for the
F1 hybrids).

Each mixer also reports:
  param_groups() -> {"main", "gate", "norm"[, "other"]} parameter counts (G1 rule)
  info()         -> key_dim_total, head_k_dim, head_v_dim, state_size (results rows)
and streams (E3, tests 3 and 5):
  init_state(batch, device, dtype, capacity, length=0) -> state   (length > 0: a state of
                  that many tokens with random contents, for cost measurements)
  prefill(h, capacity) -> (y, state)       full causal pass that also returns the state
  step(h_t, state)     -> (y_t, state)     one new token (B, 1, D)
  sdpa_backend(batch, T, device, dtype) -> name of the kernel a forward pass uses
"""
from __future__ import annotations

import torch.nn as nn

from tidemark.models.gdn import GDNMixer
from tidemark.models.gla import GLAMixer
from tidemark.models.softmax import SoftmaxMixer
from tidemark.models.swa import SWAMixer
from tidemark.utils.config import ModelConfig


def build_time_mixer(cfg: ModelConfig, gla_backend: str = "fla", layer: int = 0) -> nn.Module:
    """The mixer of layer `layer` (hybrids choose the kind per layer, F1)."""
    kind = cfg.layer_attention(layer)
    if kind == "softmax":
        return SoftmaxMixer(cfg.d_model, cfg.n_heads)
    if kind == "gla":
        return GLAMixer(cfg.d_model, cfg.n_heads, cfg.gla_expand_k, cfg.gla_expand_v,
                        cfg.gla_gate_rank, backend=gla_backend)
    if kind == "swa":
        return SWAMixer(cfg.d_model, cfg.n_heads, cfg.swa_window)
    if kind == "gdn":
        return GDNMixer(cfg.d_model, cfg.n_heads, backend="reference" if gla_backend == "reference" else "fla")
    raise ValueError(f"unknown attention {kind!r}")
