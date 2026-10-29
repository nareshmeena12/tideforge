"""Chronos-2 (amazon/chronos-2, chronos-forecasting 2.3.2): Chronos2Pipeline.predict on a list of dicts
{"target": (k, L), "past_covariates": {...}, "future_covariates": {...}} -> per item (k, 21, H) at pipeline.quantiles (which
contain 0.1 ... 0.9). Past lanes go in past_covariates only; known lanes in both (context values) and future_covariates.
batch_size 100 as in fev's published run (model_kwargs of chronos-2.csv); cross_learning off (the default)."""
from __future__ import annotations

import numpy as np

from tideforge.eval.public_models.base import PublicModel, snapshot, to_numpy


class Adapter(PublicModel):
    def load(self):
        from chronos import Chronos2Pipeline
        return Chronos2Pipeline.from_pretrained(snapshot(self.spec.repo, self.spec.revision), device_map=self.device)

    def forecast(self, items, H):
        b = self.backend
        self.levels = tuple(float(x) for x in b.quantiles)
        inputs = []
        for it in items:
            d = {"target": it["targets"]}
            if it["past"] or it["known"]:
                d["past_covariates"] = {**{f"p{j}": p for j, p in enumerate(it["past"])},
                                        **{f"k{j}": c for j, (c, _) in enumerate(it["known"])}}
            if it["known"]:
                d["future_covariates"] = {f"k{j}": f for j, (_, f) in enumerate(it["known"])}
            inputs.append(d)
        out = b.predict(inputs, prediction_length=H, batch_size=self.spec.kwargs["batch_size"])
        return [to_numpy(o) for o in out]
