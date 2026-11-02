"""Inference-only flip averaging (--flip-avg): each forecast is run on the targets as they are and on the negated targets, and
combined per level as q_tau = 0.5 * (f_tau(x) - f_{1-tau}(-x)), so g_tau(-x) = -g_{1-tau}(x) for any model. Covariates, NaN positions
and masks are not touched (-NaN is NaN), and padding is added after the negation, so it stays zero. Costs two forecasts.

use_flip(True) wraps ft3_core.forecast_left_padded and forecast_groups, both looked up by attribute at call time in the scorers, on
top of whatever the NaN variant swapped in, and restores them on exit. use_flip(False) swaps nothing. Nothing under third_party/ is
edited.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from tideforge.eval import thirdparty


def check_levels(model, C) -> None:
    lv = list(getattr(model, "quantile_levels", C.QUANTILE_LEVELS))
    if any(abs(a + b - 1) > 1e-9 for a, b in zip(lv, reversed(lv))):
        raise ValueError(f"--flip-avg needs symmetric quantile levels, got {lv}")


def combine(a: tuple, b: tuple) -> tuple:
    """a, b: (point (N, H), quant (N, n_q, H)) from x and -x."""
    return 0.5 * (a[0] - b[0]), 0.5 * (a[1] - b[1][:, ::-1, :])


def flip_group(g: dict) -> dict:
    return {**g, "targets": [-np.asarray(t, dtype=np.float32) for t in g["targets"]]}


def wrap(left_padded, groups_fn, C):
    depth = [0]                                  # forecast_groups can fall back to forecast_left_padded: flip once, not twice

    def flip_left_padded(model, series_list, horizon, device, **kw):
        if depth[0]:
            return left_padded(model, series_list, horizon, device, **kw)
        check_levels(model, C)
        neg = [-np.asarray(y, dtype=np.float32) for y in series_list]
        return combine(left_padded(model, series_list, horizon, device, **kw), left_padded(model, neg, horizon, device, **kw))

    def flip_groups(model, groups, horizon, device, **kw):
        check_levels(model, C)
        depth[0] += 1
        try:
            a = groups_fn(model, groups, horizon, device, **kw)
            b = groups_fn(model, [flip_group(g) for g in groups], horizon, device, **kw)
        finally:
            depth[0] -= 1
        return [combine(x, y) for x, y in zip(a, b)]

    return flip_left_padded, flip_groups


@contextmanager
def use_flip(on: bool):
    if not on:
        yield
        return
    thirdparty.add_path()
    import tidemark.real.ft3_core as C
    saved = C.forecast_left_padded, C.forecast_groups
    try:
        C.forecast_left_padded, C.forecast_groups = wrap(*saved, C)
        yield
    finally:
        C.forecast_left_padded, C.forecast_groups = saved
