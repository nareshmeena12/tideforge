"""Fake backends for the tests: tiny stand-ins with each library's call signature and output layout (shapes, quantile axis,
level list), so the adapters' glue (inputs, covariates, quantile selection, reshaping) is tested without any model package.

Every fake forecasts quantile level a at step h as: last finite target value + (a - 0.5) + 0.01 h + the sum of the known
covariates' future values at h (0 when none are given). So a test can check the levels, the horizon order and that the
known covariates reached the model.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

L21 = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99)
Q9 = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def _last(x):
    x = np.asarray(x, np.float64)
    f = x[np.isfinite(x)]
    return float(f[-1]) if len(f) else 0.0


def _q(targets, H, levels, known_fut=None):
    """(k, n_levels, H) for one item; targets (k, L); known_fut (F, H) or None."""
    targets = np.atleast_2d(np.asarray(targets, np.float64))
    eff = np.zeros(H) if known_fut is None or not len(known_fut) else np.asarray(known_fut, np.float64).sum(0)[:H]
    lv = np.asarray(levels)[None, :, None]
    base = np.array([_last(t) for t in targets])[:, None, None]
    return base + (lv - 0.5) + 0.01 * np.arange(H)[None, None, :] + eff[None, None, :]


class Chronos2:
    quantiles = list(L21)

    def predict(self, inputs, prediction_length, batch_size=256, **kw):
        out = []
        for d in inputs:
            fut = d.get("future_covariates")
            out.append(_q(d["target"], prediction_length, L21, np.stack(list(fut.values())) if fut else None))
        return out


class TimesFM:
    def __init__(self):
        self.ForecastConfig = lambda **kw: SimpleNamespace(**kw)
        self.config = None

    def compile(self, cfg):
        self.config = cfg

    def forecast(self, horizon, inputs):
        assert self.config is not None and self.config.max_horizon >= horizon
        q = np.stack([_q(x, horizon, Q9)[0] for x in inputs])           # (B, 9, H)
        q = np.transpose(q, (0, 2, 1))                                   # (B, H, 9)
        mean = q[..., 4:5]
        return q[..., 4], np.concatenate([mean, q], axis=-1)             # point (B, H), (B, H, 10): mean first


class Toto2:
    def forecast(self, inputs, horizon, **kw):
        import torch
        x, m = inputs["target"].cpu().numpy(), inputs["target_mask"].cpu().numpy()
        x = np.where(m, x, np.nan)
        q = np.stack([_q(b, horizon, Q9) for b in x])                    # (B, V, 9, H)
        return torch.tensor(np.transpose(q, (2, 0, 1, 3)))              # (9, B, V, H)


class TiRex:
    backend = "cuda"

    def forecast(self, context, prediction_length, output_type="torch", **kw):
        q = np.stack([_q(c.numpy(), prediction_length, Q9)[0] for c in context])   # (B, 9, H)
        q = np.transpose(q, (0, 2, 1))                                              # (B, H, 9)
        return q, q[..., 4]


class TiRex2:
    TimeseriesType = staticmethod(lambda target, past_covariates, future_covariates:
                                  SimpleNamespace(target=target, past=past_covariates, future=future_covariates))

    def forecast(self, ts, prediction_length, output_type="torch", batch_size=512, **kw):
        out = []
        for t in ts:
            L = t.target.shape[-1]
            fut = None if t.future is None else t.future.numpy()[:, L:L + prediction_length]
            out.append(_q(t.target.numpy(), prediction_length, Q9, fut))   # (k, 9, H)
        return out


class Timestral:
    def forecast(self, context, prediction_length, past_covariates=None, known_covariates=None, **kw):
        T = context.shape[-1]
        q = np.stack([_q(c, prediction_length, L21, None if known_covariates is None else known_covariates[i][:, T:])
                      for i, c in enumerate(context)])                   # (B, V, 21, H)
        return SimpleNamespace(quantiles=np.transpose(q, (0, 1, 3, 2)), quantile_levels=L21)


BACKENDS = {"chronos2": Chronos2, "timesfm25": TimesFM, "toto2": Toto2, "tirex": TiRex, "tirex2": TiRex2, "timestral": Timestral}


def backend_for(name: str):
    from tideforge.eval.public_models import SPECS
    return BACKENDS[SPECS[name].adapter]()


def expected(targets, H, known_fut=None) -> np.ndarray:
    """What every fake returns at our 9 levels for one item: (k, 9, H)."""
    return _q(targets, H, Q9, known_fut)
