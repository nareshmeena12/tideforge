"""Probe metrics as per-series components plus an aggregate, so every number gets a bootstrap CI over series and two runs
on the same items can be compared pairwise.

A metric value = agg(comps) with comps (n_series, m). bootstrap() resamples series (1000 times, fixed seed) and returns the
95% percentile interval; paired() does the same for agg(A) - agg(B) with the same resampled series for both.
Quantile forecasts are (n, k, 9, H); targets / futures (n, k, H).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from tideforge.probes.common import Q9

N_BOOT = 1000
BOOT_SEED = 20261027


@dataclass
class Metric:
    name: str
    comps: np.ndarray                      # (n, m)
    agg_name: str = "mean"                 # a key of AGG (stored with the series files, so comparisons can re-aggregate)
    mode: str = "main"
    view: str = "main"

    @property
    def agg(self) -> Callable[[np.ndarray], float]:
        return AGG[self.agg_name]


def mean_agg(c: np.ndarray) -> float:
    return float(np.nanmean(c[:, 0]))


def ratio_agg(c: np.ndarray) -> float:
    """sum(numerator) / sum(denominator) over series."""
    return float(np.nansum(c[:, 0]) / max(np.nansum(c[:, 1]), 1e-300))


def one_minus_ratio_agg(c: np.ndarray) -> float:
    return 1.0 - ratio_agg(c)


def abs_diff_agg(c: np.ndarray) -> float:
    """|mean(column 0) - mean(column 1)| (e.g. predicted vs observed zero share)."""
    return float(abs(np.nanmean(c[:, 0]) - np.nanmean(c[:, 1])))


def _boot_values(comps: np.ndarray, agg_name: str, idx: np.ndarray) -> np.ndarray:
    """agg over each resample idx (B, n), vectorised for the built-in aggregates."""
    c = comps[idx]                                           # (B, n, m)
    with np.errstate(invalid="ignore", divide="ignore"):
        if agg_name == "mean":
            return np.nanmean(c[..., 0], axis=1)
        if agg_name in ("ratio", "one_minus_ratio"):
            r = np.nansum(c[..., 0], 1) / np.maximum(np.nansum(c[..., 1], 1), 1e-300)
            return r if agg_name == "ratio" else 1.0 - r
        if agg_name == "abs_diff":
            return np.abs(np.nanmean(c[..., 0], 1) - np.nanmean(c[..., 1], 1))
        if agg_name == "pit_tv":
            f = c.sum(1) / np.maximum(c.sum((1, 2)), 1)[:, None]
            return 0.5 * np.abs(f - 0.1).sum(1)
    return np.array([AGG[agg_name](comps[i]) for i in idx])


def bootstrap(m: Metric, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(m.comps), size=(n_boot, len(m.comps)))
    b = _boot_values(m.comps, m.agg_name, idx)
    return m.agg(m.comps), float(np.nanpercentile(b, 2.5)), float(np.nanpercentile(b, 97.5))


def paired(a: Metric, b: Metric, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """agg(A) - agg(B) on the same items, with the same resampled series for both."""
    if a.comps.shape[0] != b.comps.shape[0]:
        raise ValueError("paired comparison needs the same items")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a.comps), size=(n_boot, len(a.comps)))
    bd = _boot_values(a.comps, a.agg_name, idx) - _boot_values(b.comps, b.agg_name, idx)
    return float(a.agg(a.comps) - b.agg(b.comps)), float(np.nanpercentile(bd, 2.5)), float(np.nanpercentile(bd, 97.5))


def unpaired(a: Metric, b: Metric, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float, float]:
    """agg(A) / agg(B) for different items (e.g. a cell vs its control): each side resampled on its own."""
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a.comps), size=(n_boot, len(a.comps)))
    ib = rng.integers(0, len(b.comps), size=(n_boot, len(b.comps)))
    r = _boot_values(a.comps, a.agg_name, ia) / np.maximum(_boot_values(b.comps, b.agg_name, ib), 1e-300)
    return float(a.agg(a.comps) / max(b.agg(b.comps), 1e-300)), float(np.nanpercentile(r, 2.5)), float(np.nanpercentile(r, 97.5))


# ---------------------------------------------------------------------------------------------------- per-series pieces
def scale_seasonal(ctx: np.ndarray, S) -> np.ndarray:
    """(n, k, L) -> (n, k): mean |y_t - y_{t-S}| over the context (NaN-aware), floored. S: int or (n,) per series."""
    n, k, L = ctx.shape
    Ss = np.broadcast_to(np.asarray(S, int), (n,))
    out = np.empty((n, k))
    for s in np.unique(Ss):
        rows = Ss == s
        s_ = max(1, min(int(s), L - 1))
        c = ctx[rows]
        with np.errstate(invalid="ignore"):
            d = np.nanmean(np.abs(c[..., s_:] - c[..., :-s_]), axis=-1)
            alt = np.nanmean(np.abs(c - np.nanmean(c, -1, keepdims=True)), -1)
        out[rows] = np.where(np.isfinite(d) & (d > 0), d, alt)
    return np.maximum(np.nan_to_num(out, nan=1.0), 1e-8)


def vs_oracle(q, cell, scale, mode="main", view="main", mask=None) -> "Metric":
    """sum sql(model) / sum sql(oracle) over series (1 = as good as the oracle), paired by item."""
    y = cell.arrays["future"]
    return ratio("sql_vs_oracle", sql(q, y, scale, mask), sql(cell.arrays["oracle_q"], y, scale, mask), mode, view)


def scale_abs(ctx: np.ndarray) -> np.ndarray:
    """(n, k, L) -> (n, k): mean |y| over the context (counts), floored."""
    return np.maximum(np.nan_to_num(np.nanmean(np.abs(ctx), -1), nan=1.0), 1e-3)


def pinball(q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(n, k, 9, H), (n, k, H) -> (n, k, H): mean pinball loss over the 9 levels."""
    d = y[..., None, :] - q
    tau = Q9[:, None]
    return np.mean(np.maximum(tau * d, (tau - 1) * d), axis=-2)


def sql(q, y, scale, mask=None) -> np.ndarray:
    """Per series (n, 1): pinball / scale, averaged over targets and steps (or over mask (n, k, H) steps only)."""
    p = pinball(q, y) / scale[..., None]
    if mask is not None:
        p = np.where(mask, p, np.nan)
    return np.nanmean(p.reshape(len(p), -1), axis=1)[:, None]


def coverage(q, y, lo: int = 0, hi: int = 8) -> np.ndarray:
    inside = (y >= q[..., lo, :]) & (y <= q[..., hi, :])
    return inside.reshape(len(y), -1).mean(1)[:, None]


def width(q, scale, lo: int = 0, hi: int = 8) -> np.ndarray:
    w = (q[..., hi, :] - q[..., lo, :]) / scale[..., None]
    return w.reshape(len(w), -1).mean(1)[:, None]


def pit_counts(q, y) -> np.ndarray:
    """(n, 10): per series, how many (target, step) values fall below 0..9 of the quantiles (bins of 0.1 each)."""
    b = (y[..., None, :] > q).sum(-2)                       # (n, k, H) in 0..9
    return np.stack([(b == i).reshape(len(b), -1).sum(1) for i in range(10)], 1).astype(float)


def pit_tv(c: np.ndarray) -> float:
    """Total-variation distance of the pooled PIT histogram from uniform (0 = perfect, 1 = worst)."""
    f = c.sum(0) / max(c.sum(), 1)
    return float(0.5 * np.abs(f - 0.1).sum())


def p_zero(q: np.ndarray) -> np.ndarray:
    """P(y = 0) from 9 quantiles of a count forecast: the highest level whose quantile is <= 0.5, interpolated linearly to
    the next level (0 below the 0.1 quantile, capped at 0.9). Resolution about 0.1."""
    qs = np.sort(q, axis=-2)
    c = (qs <= 0.5).sum(-2)                                  # number of levels at or below 0.5
    lo_i = np.clip(c - 1, 0, 8)
    hi_i = np.clip(c, 0, 8)
    q_lo = np.take_along_axis(qs, lo_i[..., None, :], -2)[..., 0, :]
    q_hi = np.take_along_axis(qs, hi_i[..., None, :], -2)[..., 0, :]
    frac = np.clip((0.5 - q_lo) / np.where(q_hi > q_lo, q_hi - q_lo, 1.0), 0, 1)
    p = Q9[lo_i] + 0.1 * frac
    return np.where(c == 0, 0.0, np.where(c == 9, 0.9, p))


AGG = {"mean": mean_agg, "ratio": ratio_agg, "one_minus_ratio": one_minus_ratio_agg, "pit_tv": lambda c: pit_tv(c),
       "abs_diff": abs_diff_agg}


def standard(q, y, scale, mode="main", view="main", prefix="") -> list:
    """The metrics every cell reports: sql, 10-90 coverage, 10-90 width (scaled)."""
    return [Metric(prefix + "sql", sql(q, y, scale), "mean", mode, view),
            Metric(prefix + "cov80", coverage(q, y), "mean", mode, view),
            Metric(prefix + "width80", width(q, scale), "mean", mode, view)]


def ratio(name: str, num: np.ndarray, den: np.ndarray, mode: str = "main", view: str = "main") -> Metric:
    """sum(num) / sum(den) over series, paired by item (e.g. sql on / sql hidden)."""
    return Metric(name, np.concatenate([num, den], 1), "ratio", mode, view)


def median(q):
    return np.sort(q, axis=-2)[..., 4, :]
