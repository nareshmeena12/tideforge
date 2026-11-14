"""Seasonal naive, two definitions (every quantile = the point forecast):
  fev      score_twins.naive_preds, the baseline behind our fev-bench numbers (43.78 / 42.09): NaN dropped from the past,
           the last min(season, n) finite values repeated (zeros when nothing is finite).
  tfbench  tfbench.baseline.sn_forecast, the baseline column of BUILD_REPORT.md: the last season repeated, a NaN in it
           replaced by the last finite value; a past shorter than one season repeats the last finite value.
fev-bench runs use `fev`, tfbench runs `tfbench` (run.py picks it)."""
from __future__ import annotations

import numpy as np

from tideforge.score.adapters.base import Predictor


def fev_naive(past: np.ndarray, H: int, season: int) -> np.ndarray:
    y = np.asarray(past, dtype=np.float64)
    y = y[np.isfinite(y)]
    if len(y) == 0:
        y = np.zeros(1)
    sl = max(1, min(int(season), len(y)))
    return np.tile(y[-sl:], int(np.ceil(H / sl)))[:H]


class SeasonalNaive(Predictor):
    groups = False
    needs_season = True                  # the engine calls predict once per seasonality

    def __init__(self, kind: str = "tfbench"):
        if kind not in ("fev", "tfbench"):
            raise ValueError(f"seasonal naive kind {kind!r}: fev or tfbench")
        self.kind, self.name = kind, "seasonal_naive"

    def predict(self, contexts, horizon, freq="", past_covariates=None, known_covariates=None, season=1):
        if past_covariates is not None or known_covariates is not None:
            raise ValueError("seasonal naive is univariate")
        from tideforge.tfbench.baseline import sn_forecast
        f = [fev_naive(c, horizon, season) if self.kind == "fev" else sn_forecast(np.asarray(c, np.float64), horizon, season)
             for c in contexts]
        out = np.asarray(f, dtype=np.float64).reshape(len(contexts), 1, horizon)
        return np.repeat(out, len(self.quantile_levels), axis=1)
