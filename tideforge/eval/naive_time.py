"""SeasonalNaiveTIME: TIME's seasonal-naive baseline, made deterministic.

TIME (zqiao11/TIME c633e23, timebench/models/statsforecast_predictor.py) fits statsforecast's SeasonalNaive with
level=[20, 40, 60, 80], draws 100 unseeded samples from that band by inverse-CDF interpolation and takes np.quantile of the
samples. Here the 9 quantiles are read straight off the band (no sampling, owner 2026-10-27), so our naive's CRPS differs
slightly from TIME-Output's. The band is reimplemented (statsforecast is a test-only extra, test-sf):
    mean_t   = y[n - S + (t mod S)]                       (the last season repeated)
    sigma    = sqrt(sum_{t>=S} (y_t - y_{t-S})^2 / (n - S))
    q_t(p)   = mean_t + Phi^-1(p) * sigma * sqrt(floor(t / S) + 1)
History preparation as TIME's predictor: NaN carried forward, leading NaN set to the first valid value, an all-NaN history
to 0. A history shorter than S gives NaN (statsforecast returns NaN there).
"""
from __future__ import annotations

import numpy as np

from tideforge.probes.common import Z9
from tideforge.probes.models import Model


def prepare(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, np.float64).copy()
    ok = np.isfinite(y)
    if ok.all():
        return y
    if not ok.any():
        return np.zeros_like(y)
    idx = np.where(ok, np.arange(len(y)), 0)
    np.maximum.accumulate(idx, out=idx)
    y = y[idx]
    y[:np.argmax(ok)] = y[np.argmax(ok)]
    return y


def band(y: np.ndarray, H: int, S: int) -> np.ndarray:
    """(9, H) quantiles 0.1..0.9 of statsforecast's SeasonalNaive band for history y."""
    y = prepare(y)
    n, S = len(y), max(1, int(S))
    if n < S or n == 0:
        return np.full((9, H), np.nan)
    t = np.arange(H)
    mean = y[n - S + (t % S)]
    sigma = np.sqrt(np.sum((y[S:] - y[:-S]) ** 2) / (n - S)) if n > S else 0.0
    sh = sigma * np.sqrt(np.floor(t / S) + 1)
    return mean[None, :] + Z9[:, None] * sh[None, :]


class SeasonalNaiveTIME(Model):
    """Rows of the cell may be left-padded with NaN to a common length: arrays["ctx_len"] (n,) gives each row's true length."""
    name = "naive_time"
    max_context = None                                     # the full history, as statsforecast sees it

    def predict_cell(self, cell, mode="main", view="main"):
        T = cell.targets(view)
        n, k, L = T.shape
        lens = cell.arrays.get("ctx_len", np.full(n, L))
        out = np.empty((n, k, 9, cell.H))
        for i, s in enumerate(cell.seasons()):
            for j in range(k):
                out[i, j] = band(T[i, j, L - int(lens[i]):], cell.H, int(s))
        return out
