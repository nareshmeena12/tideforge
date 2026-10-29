"""TiRex-2 (NX-AI/TiRex-2, tirex-2 @ 12aa6dd; own venv $TF_ROOT/venvs/tirex2: torch >= 2.8, numpy >= 2.1.3, flashrnn,
mlstm_kernels). Multivariate with past and known covariates: TimeseriesType(target (k, L), past_covariates (P, L),
future_covariates (F, L + H)); forecast(list, prediction_length, output_type="numpy") -> per item (k, 9, H) at the
checkpoint's quantiles (0.1 ... 0.9, config.json). batch_size 512 as in tirex-2's fev submission
(examples/fevbench/submission.py)."""
from __future__ import annotations

import numpy as np

from tideforge.eval.public_models.base import PublicModel, snapshot, to_numpy


class Adapter(PublicModel):
    def load(self):
        from tirex2 import load_model
        return load_model(snapshot(self.spec.repo, self.spec.revision), device=self.device)

    def forecast(self, items, H):
        import torch
        TimeseriesType = getattr(self.backend, "TimeseriesType", None)       # a test backend brings its own
        if TimeseriesType is None:
            from tirex2 import TimeseriesType

        def t(x):
            return torch.tensor(np.asarray(x), dtype=torch.float32)
        ts = [TimeseriesType(target=t(it["targets"]),
                             past_covariates=t(np.stack(it["past"])) if it["past"] else None,
                             future_covariates=t(np.stack([np.concatenate([c, f]) for c, f in it["known"]])) if it["known"] else None)
              for it in items]
        out = self.backend.forecast(ts, prediction_length=H, output_type="numpy", batch_size=self.spec.kwargs["batch_size"])
        return [to_numpy(o)[..., :H] for o in out]
