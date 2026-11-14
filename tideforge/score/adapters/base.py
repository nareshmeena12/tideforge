"""The predictor interface every adapter implements.

predict(contexts, horizon, freq, past_covariates=None, known_covariates=None, season=1) -> quantiles
  univariate call (both covariate arguments None): contexts = list of 1-D float arrays (the model's input, already prepared
      by prepare()); returns (n, Q, horizon) float array, sorted along Q.
  group call (covariates given, possibly empty lists): contexts = list of groups, each a list of 1-D target arrays (sibling
      variates of one item); past_covariates[g] = list of 1-D arrays; known_covariates[g] = list of (context, future) pairs;
      returns a list of (n_targets, Q, horizon) arrays, sorted along Q.
Attributes: name, quantile_levels, max_context (None = the full past), groups (the architecture has group / lane attention:
the default mode is groups ON), and groups_ok(horizon) (a group call is possible at this horizon; else the engine forecasts
the targets one by one, as ft3_core.forecast_groups itself falls back).
"""
from __future__ import annotations

import numpy as np

QL = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class Predictor:
    name = "predictor"
    quantile_levels = QL
    max_context: int | None = None
    groups = False
    needs_season = False                 # True: predict's season argument matters (the engine groups calls by it)

    def groups_ok(self, horizon: int) -> bool:
        return self.groups

    def prepare(self, past: np.ndarray, season: int) -> np.ndarray:
        """The row's past as the model receives it (default: the last max_context points, unchanged)."""
        return past if self.max_context is None else past[-self.max_context:]

    def predict(self, contexts, horizon: int, freq: str = "", past_covariates=None, known_covariates=None, season: int = 1):
        raise NotImplementedError
