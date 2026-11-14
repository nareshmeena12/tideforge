"""Predictor adapters: seasonal naive, ft3 checkpoints (ft3_main and tideforge runs). get(model) picks one by name or path."""
from __future__ import annotations

from pathlib import Path


def get(model: str, device=None, batch_size: int | None = None, naive_kind: str = "tfbench", rescale: bool = True):
    if model in ("seasonal_naive", "naive"):
        from tideforge.score.adapters.naive import SeasonalNaive
        return SeasonalNaive(naive_kind)
    if Path(model).suffix in (".pt", ".pth") or Path(model).exists():
        from tideforge.score.adapters.ft3 import Ft3
        return Ft3(model, device=device, batch_size=batch_size, rescale=rescale)
    raise ValueError(f"--model {model!r}: a checkpoint path (.pt) or seasonal_naive")
