"""Draws from a switch's own generator (never torch's global stream)."""
from __future__ import annotations

import torch


def u(g: torch.Generator, shape, lo: float, hi: float, device) -> torch.Tensor:
    return lo + (hi - lo) * torch.rand(shape, generator=g, device=device)


def bern(g: torch.Generator, shape, p: float, device) -> torch.Tensor:
    return torch.rand(shape, generator=g, device=device) < p


def randint(g: torch.Generator, lo: int, hi: int, shape, device) -> torch.Tensor:
    """Integers in [lo, hi]."""
    return torch.randint(lo, hi + 1, shape, generator=g, device=device)
