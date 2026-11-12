"""Seasonal naive on every task (CPU), from the packaged fev tables: the baseline column of BUILD_REPORT.md and a check
that every task scores (NaN or infinite scores are flagged).

Windows as fev cuts them: cutoffs -H - (n-1-i) H from each series' end; a series enters a window when it has
min_context_length points before the cutoff. Forecast: the last season repeated (last finite value where the season has
NaN, or when the context is shorter than one season); every quantile = the point forecast.
MASE = mean over series of mean|e| / seasonal-difference scale of the context; WQL = sum of quantile losses x 2 / sum |y|
/ number of levels; SQL = mean over series of the quantile loss x 2 / scale. Each averaged over windows (fev's default).
"""
from __future__ import annotations

import numpy as np
import pyarrow.parquet as pq


def sn_forecast(past: np.ndarray, H: int, m: int) -> np.ndarray:
    fin = past[np.isfinite(past)]
    last = fin[-1] if len(fin) else 0.0
    if len(past) < m:
        return np.full(H, last)
    season = past[-m:].copy()
    season[~np.isfinite(season)] = last
    return season[np.arange(H) % m]


def _scale(past: np.ndarray, m: int) -> float:
    m = m if len(past) > m else 1
    d = np.abs(past[m:] - past[:-m])
    d = d[np.isfinite(d)]
    return float(d.mean()) if len(d) else np.nan


def score(table_path, task: dict) -> dict:
    t = pq.read_table(table_path).to_pydict()
    H, n, m, ctx = task["horizon"], task["num_windows"], task["seasonality"], task["min_context_length"]
    tg = task.get("generate_univariate_targets_from") or (task["target"] if isinstance(task["target"], list) else [task["target"]])
    q = np.asarray(task.get("quantile_levels") or [0.5], dtype=float)
    res = {"MASE": [], "WQL": [], "SQL": []}
    used = 0
    for i in range(n):
        cut = -H - (n - 1 - i) * H
        mase, sql, num, den = [], [], 0.0, 0.0
        for col in tg:
            for y in t[col]:
                y = np.asarray(y, dtype=np.float64)
                if len(y) + cut < ctx:
                    continue
                past, true = y[:len(y) + cut], y[len(y) + cut: len(y) + cut + H]
                f = sn_forecast(past, H, m)
                e = true - f
                pin = np.where(e[:, None] >= 0, e[:, None] * q[None, :], -e[:, None] * (1 - q[None, :]))
                sc = _scale(past, m)
                with np.errstate(divide="ignore", invalid="ignore"):
                    mase.append(np.nanmean(np.abs(e)) / sc)
                    sql.append(2 * np.nanmean(pin) / sc)
                num += 2 * np.nansum(pin) / len(q)
                den += np.nansum(np.abs(true))
                used += 1
        if mase:
            res["MASE"].append(float(np.mean(mase)))
            res["SQL"].append(float(np.mean(sql)))
            with np.errstate(divide="ignore", invalid="ignore"):
                res["WQL"].append(float(num / den) if den else float("inf") if num else float("nan"))
    out = {k: (float(np.mean(v)) if v else float("nan")) for k, v in res.items()}
    out["series_windows"] = used
    out["flag"] = "; ".join(f"{k} {'NaN' if np.isnan(v) else 'inf'}" for k, v in out.items()
                            if k in res and not np.isfinite(v))
    return out
