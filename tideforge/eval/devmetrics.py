"""Dev-set metrics and aggregation, as TIME does it (zqiao11/TIME at c633e23, read 2026-10-27):

- per window: MASE and CRPS from timebench.evaluation.metrics.compute_per_window_metrics_from_quantiles, called directly
  (MASE: median forecast / mean |seasonal difference| of the context; CRPS: mean over the 9 quantiles of
  2 * sum(pinball) / sum|y|);
- per (dataset, term): the mean over its windows (NaN ignored), divided by seasonal naive's mean for the same pair
  (scripts/compute_local_leaderboard.py normalize_by_seasonal_naive);
- per dataset: geometric mean over its terms; overall: weighted geometric mean over datasets with the manifest weights
  (every dataset equal, the 10% domain cap applied as weights).
Slices and the bootstrap use the same per-(dataset, term) ratios: a slice keeps the windows in it, recomputes the pair ratios
on those windows and reweights the datasets it contains (their manifest weights, renormalised). The bootstrap resamples
datasets with replacement (each draw keeps its weight).

timebench is found as an installed package, else in $TF_TIME_REPO/src, else in $TM_EVAL/eval/data/TIME/src (the clone the
tidemark scorer fetches).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("MASE", "CRPS")
HIST_BANDS = ((1, 64, "1-64"), (65, 512, "65-512"), (513, 2048, "513-2048"), (2049, None, "2049+"))   # fev's bands
N_BOOT = 1000
BOOT_SEED = 20261027


def timebench_metrics():
    try:
        from timebench.evaluation.metrics import compute_per_window_metrics_from_quantiles as f
        return f
    except ImportError:
        pass
    for root in (os.environ.get("TF_TIME_REPO"), os.environ.get("TM_EVAL") and str(Path(os.environ["TM_EVAL"]) / "eval" / "data" / "TIME")):
        if root and (Path(root) / "src" / "timebench").is_dir():
            if str(Path(root) / "src") not in sys.path:
                sys.path.insert(0, str(Path(root) / "src"))
            from timebench.evaluation.metrics import compute_per_window_metrics_from_quantiles as f
            return f
    raise ImportError("timebench not found: install it, or set TF_TIME_REPO (a clone of github.com/zqiao11/TIME)")


def window_metrics(q: np.ndarray, y: np.ndarray, ctx: np.ndarray, S: int) -> tuple[float, float]:
    """(MASE, CRPS) of one window: q (9, H) quantiles 0.1..0.9, y (H,), ctx (L,) the full history before the cutoff."""
    f = timebench_metrics()
    q, y, ctx = (np.asarray(a, dtype=np.float64) for a in (q, y, ctx))
    m = f(q[None, None, :, None, :], y[None, None, None, :], ctx[None, None, None, :], seasonality=int(S))
    return float(m["MASE"][0, 0, 0]), float(m["CRPS"][0, 0, 0])


def hist_band(n: int) -> str:
    for lo, hi, name in HIST_BANDS:
        if n >= lo and (hi is None or n <= hi):
            return name
    return "0"


def pair_ratios(win: pd.DataFrame, by: tuple = ()) -> pd.DataFrame:
    """Per (dataset_id, term, *by): mean over windows (NaN ignored) of the model's and naive's MASE / CRPS and their ratios
    r_MASE, r_CRPS (NaN where naive's mean is 0, NaN or the ratio is not finite, as TIME)."""
    keys = ["dataset_id", "term", *by]
    cols = [*METRICS, *(f"{m}_naive" for m in METRICS)]
    g = win.groupby(keys, dropna=False)[cols].mean().reset_index()
    for m in METRICS:
        with np.errstate(divide="ignore", invalid="ignore"):
            r = g[m] / g[f"{m}_naive"]
        g[f"r_{m}"] = r.where(np.isfinite(r) & (g[f"{m}_naive"] != 0))
    return g


def _gmean(x: pd.Series) -> float:
    x = x[np.isfinite(x) & (x > 0)]
    return float(np.exp(np.log(x).mean())) if len(x) else float("nan")


def dataset_scores(pairs: pd.DataFrame, by: tuple = ()) -> pd.DataFrame:
    """Per (dataset_id, *by): geometric mean over terms of r_MASE / r_CRPS."""
    return pairs.groupby(["dataset_id", *by], dropna=False)[[f"r_{m}" for m in METRICS]].agg(_gmean).reset_index()


def weighted_gmean(values: np.ndarray, weights: np.ndarray) -> float:
    v, w = np.asarray(values, float), np.asarray(weights, float)
    ok = np.isfinite(v) & (v > 0) & (w > 0)
    if not ok.any():
        return float("nan")
    return float(np.exp((w[ok] * np.log(v[ok])).sum() / w[ok].sum()))


def boot_ci(values, weights, n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> tuple[float, float]:
    """95% CI of the weighted geometric mean, resampling datasets with replacement."""
    v, w = np.asarray(values, float), np.asarray(weights, float)
    if len(v) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    bs = np.array([weighted_gmean(v[i], w[i]) for i in idx])
    return float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))


def overall(ds: pd.DataFrame, weights: dict[str, float], ci: bool = True) -> dict:
    """{metric: (score, lo, hi)} over the datasets in ds, weights = manifest weights (renormalised to the datasets present)."""
    w = ds["dataset_id"].map(weights).astype(float).to_numpy()
    out = {"datasets": int(len(ds))}
    for m in METRICS:
        v = ds[f"r_{m}"].to_numpy()
        out[m] = (weighted_gmean(v, w), *(boot_ci(v, w) if ci else (float("nan"),) * 2))
    return out


def slices(win: pd.DataFrame, weights: dict[str, float], keys=("domain", "freq", "term", "hist", "cov")) -> pd.DataFrame:
    """One row per (slice key, value): the overall score of the windows in that slice."""
    rows = []
    for k in keys:
        if k not in win.columns:
            continue
        by = () if k == "term" else (k,)
        pairs = pair_ratios(win, by)
        ds = dataset_scores(pairs, by + (("term",) if k == "term" else ()))
        for val, g in ds.groupby(k, dropna=False):
            o = overall(g, weights)
            rows.append(dict(slice=k, value=val, datasets=o["datasets"],
                             **{f"{m}{s}": o[m][i] for m in METRICS for i, s in enumerate(("", "_lo", "_hi"))}))
    return pd.DataFrame(rows)


def score(win: pd.DataFrame, weights: dict[str, float]) -> dict:
    """The headline numbers of a per-window table: overall (with CIs), per-dataset scores and slices."""
    ds = dataset_scores(pair_ratios(win))
    return dict(overall=overall(ds, weights), datasets=ds, slices=slices(win, weights))


def paired(a: pd.DataFrame, b: pd.DataFrame, weights: dict[str, float], metric: str = "MASE",
           n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """A vs B on the datasets both have: the weighted geometric mean of the per-dataset ratios A / B (< 1 = A better), a
    95% bootstrap CI over datasets, and the per-dataset table."""
    j = a[["dataset_id", f"r_{metric}"]].merge(b[["dataset_id", f"r_{metric}"]], on="dataset_id", suffixes=("_a", "_b"))
    j["ratio"] = j[f"r_{metric}_a"] / j[f"r_{metric}_b"]
    w = j["dataset_id"].map(weights).astype(float).to_numpy()
    lo, hi = boot_ci(j["ratio"].to_numpy(), w, n_boot, seed)
    return dict(ratio=weighted_gmean(j["ratio"].to_numpy(), w), lo=lo, hi=hi, datasets=int(len(j)), table=j)
