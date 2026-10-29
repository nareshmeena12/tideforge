"""Timestral-5M / -28M (mistralai, timestral @ a534bdb): multivariate with past and known covariates, NaN = missing.
TimestralForecaster.forecast(context (B, V, T), prediction_length, past_covariates (B, P, T), known_covariates (B, F, T + H))
-> Forecast.quantiles (B, V, H, 21) at Forecast.quantile_levels (0.01 ... 0.99, containing 0.1 ... 0.9). Categorical lanes
arrive already as numeric codes (the scorers encode them) and go in as numeric covariates, as for every other model.
from_pretrained / forecast defaults otherwise (fev's published run names only the model path in model_kwargs)."""
from __future__ import annotations

import numpy as np

from tideforge.eval.public_models.base import PublicModel, snapshot


class Adapter(PublicModel):
    def load(self):
        from timestral import TimestralForecaster
        return TimestralForecaster.from_pretrained(snapshot(self.spec.repo, self.spec.revision), device=self.device)

    def forecast(self, items, H):
        kw = {}
        if items[0]["past"]:
            kw["past_covariates"] = np.stack([np.stack(it["past"]) for it in items])
        if items[0]["known"]:
            kw["known_covariates"] = np.stack([np.stack([np.concatenate([c, f]) for c, f in it["known"]]) for it in items])
        fc = self.backend.forecast(np.stack([it["targets"] for it in items]), prediction_length=H, **kw)
        self.levels = tuple(float(x) for x in fc.quantile_levels)
        q = np.asarray(fc.quantiles, np.float64)                           # (B, V, H, n_levels)
        return list(np.transpose(q, (0, 1, 3, 2)))                         # B x (V, n_levels, H)
