"""Equal-weight-per-dataset means and their bootstrap CIs over datasets (shared by D2-D6; seeds from tideforge.eval.devmetrics)."""
from __future__ import annotations

import numpy as np

from tideforge.eval.devmetrics import BOOT_SEED, N_BOOT


def per_dataset(values, datasets) -> tuple[np.ndarray, np.ndarray]:
    """(dataset ids, mean of `values` per dataset)."""
    ids, inv = np.unique(np.asarray(datasets), return_inverse=True)
    v = np.asarray(values, np.float64)
    return ids, np.bincount(inv, weights=v, minlength=len(ids)) / np.bincount(inv, minlength=len(ids))


def mean_ci(values, datasets, n_boot: int = N_BOOT, seed: int = BOOT_SEED, alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean over datasets of the per-dataset mean (every dataset weighs the same), with a percentile CI from resampling datasets."""
    _, m = per_dataset(values, datasets)
    if len(m) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    b = m[rng.integers(0, len(m), size=(n_boot, len(m)))].mean(1)
    return float(m.mean()), float(np.quantile(b, alpha / 2)), float(np.quantile(b, 1 - alpha / 2))


def window_weights(datasets) -> np.ndarray:
    """1 / (windows of the window's dataset): sums to 1 per dataset."""
    _, inv, cnt = np.unique(np.asarray(datasets), return_inverse=True, return_counts=True)
    return 1.0 / cnt[inv]
