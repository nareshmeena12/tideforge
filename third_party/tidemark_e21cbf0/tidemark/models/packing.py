"""Packed rows: many series per row (Milestone 4 twins, LEAN_PLAN 4.2).

A packed batch is R rows x P patches; `seg` (R, P) holds the series id of every patch.
Series never cross a row, and every series is a run of consecutive patches. Each time
mixer reads the geometry it needs from one Pack:
  softmax : same series AND causal (Tide's masked SDPA)                 -> mask()
  SWA     : the same AND (p - q) < window                               -> mask(window)
  GLA     : rows flattened to one sequence, cu_seqlens at every series
            boundary and row start (state reset per series, risk 6)     -> cu_seqlens
The Pack is built once per forward pass and shared by every layer, so the masks and
cu_seqlens are computed once (fla caches its chunk indices by tensor identity).
"""
from __future__ import annotations

import torch


def band_causal(P: int, device, window: int | None = None) -> torch.Tensor:
    """(P, P) bool: query p may see key q iff q <= p (and p - q < window)."""
    p = torch.arange(P, device=device)
    d = p[:, None] - p[None, :]
    ok = d >= 0
    if window is not None:
        ok = ok & (d < window)
    return ok


class Pack:
    def __init__(self, seg: torch.Tensor):
        if seg.dim() != 2:
            raise ValueError("seg must be (rows, patches)")
        self.seg = seg
        self._masks: dict = {}
        self._cu = None

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.seg.shape)

    def mask(self, window: int | None = None) -> torch.Tensor:
        """(R, 1, P, P) bool, True = may attend: same series, causal, and within the window."""
        if window not in self._masks:
            same = self.seg.unsqueeze(2) == self.seg.unsqueeze(1)
            self._masks[window] = (same & band_causal(self.seg.shape[1], self.seg.device, window)).unsqueeze(1)
        return self._masks[window]

    @property
    def cu_seqlens(self) -> torch.Tensor:
        """int64 (n_series + 1,): start offsets of every series in the flattened (R * P)
        sequence, plus its total length. A boundary is placed wherever the series id
        changes AND at every row start (a series never crosses a row)."""
        if self._cu is None:
            R, P = self.seg.shape
            flat = self.seg.reshape(-1)
            start = torch.ones_like(flat, dtype=torch.bool)
            start[1:] = flat[1:] != flat[:-1]
            start[::P] = True
            idx = start.nonzero().squeeze(1)
            self._cu = torch.cat([idx, idx.new_tensor([R * P])]).to(torch.long)
        return self._cu

    def segments(self) -> list[tuple[int, int]]:
        """(start, end) of every series in the flattened sequence (tests, reference GLA)."""
        cu = self.cu_seqlens.tolist()
        return list(zip(cu[:-1], cu[1:]))
