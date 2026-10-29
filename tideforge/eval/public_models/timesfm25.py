"""TimesFM-2.5 (google/timesfm-2.5-200m-pytorch, timesfm 3.0.2): univariate, hidden mode only (covariates exist only through
XReg, which needs jax: not installed, owner 2026-10-29). forecast(horizon, inputs) -> point (B, H), quantiles (B, H, 10) with
column 0 the mean and 1..9 the levels 0.1 ... 0.9. ForecastConfig as on the model card (normalize_inputs, continuous quantile
head, flip invariance, infer_is_positive, fix_quantile_crossing); max_context 16384 (the 2.5 limit); max_horizon the next
multiple of 128 >= H (recompiled when a longer horizon comes)."""
from __future__ import annotations

import numpy as np

from tideforge.eval.public_models.base import PublicModel, regroup, snapshot, univariate


class Adapter(PublicModel):
    _horizon = 0

    def load(self):
        import timesfm
        return timesfm.TimesFM_2p5_200M_torch.from_pretrained(snapshot(self.spec.repo, self.spec.revision))

    def _compile(self, H):
        mh = -(-H // 128) * 128
        if mh > self._horizon:
            cfg = getattr(self.backend, "ForecastConfig", None)               # a test backend brings its own
            if cfg is None:
                import timesfm
                cfg = timesfm.ForecastConfig
            self.backend.compile(cfg(
                max_context=self.max_context, max_horizon=mh, normalize_inputs=True, use_continuous_quantile_head=True,
                force_flip_invariance=True, infer_is_positive=True, fix_quantile_crossing=True))
            self._horizon = mh

    def forecast(self, items, H):
        self._compile(H)
        ser, ks = univariate(items)
        _, q = self.backend.forecast(horizon=H, inputs=[np.asarray(s, np.float64) for s in ser])
        q = np.asarray(q, np.float64)[:len(ser), :H, 1:]              # drop the mean column -> (B, H, 9) at 0.1 ... 0.9
        return regroup(np.transpose(q, (0, 2, 1)), ks)
