"""--flip-avg: off swaps nothing, on is antisymmetric and monotone, and only the targets change in the flipped call."""
from types import SimpleNamespace

import numpy as np
import pytest

import tidemark.real.ft3_core as C
from tideforge.eval import flip, score

MODEL = SimpleNamespace(quantile_levels=C.QUANTILE_LEVELS)
H = 4
OFFSETS = np.array([-4, -2, -1, -.5, 0, 1, 3, 6, 10], dtype=np.float32)


def fake_plain(model, series, horizon, device, **kw):
    """An asymmetric forecaster with sorted quantiles."""
    c = np.array([np.nansum(np.asarray(y) ** 2) ** 0.5 + 2 * np.nanmean(y) for y in series], dtype=np.float32)
    q = c[:, None, None] + OFFSETS[None, :, None] * np.ones((1, 1, horizon), np.float32)
    return q[:, 4, :], q


def wrapped(calls=None):
    def groups_fn(model, groups, horizon, device, **kw):
        if calls is not None:
            calls.append(groups)
        return [fake_plain(model, g["targets"], horizon, device) for g in groups]
    return flip.wrap(fake_plain, groups_fn, C)


def test_off_swaps_nothing():
    before = C.forecast_left_padded, C.forecast_groups
    with flip.use_flip(False):
        assert (C.forecast_left_padded, C.forecast_groups) == before
    with flip.use_flip(True):
        assert C.forecast_left_padded is not before[0]
    assert (C.forecast_left_padded, C.forecast_groups) == before


def test_tags(monkeypatch):
    seen = []
    monkeypatch.setattr(score.thirdparty.score_groups(), "main", lambda argv: seen.append(argv) or 0)
    base = ["score", "--root", "R", "--ckpt", "r14=p", "--modes", "on"]
    assert score.main(base) == 0 and score.main(base + ["--flip-avg"]) == 0
    assert "r14base=p" in seen[0] and "r14flipbase=p" in seen[1]


def test_antisymmetry_and_order():
    plain, groups = wrapped()
    x = [np.array([1, 2, np.nan, 4], np.float32), np.array([3, -1, .5], np.float32)]
    p, q = plain(MODEL, x, H, None)
    pn, qn = plain(MODEL, [-y for y in x], H, None)
    assert np.array_equal(qn, -q[:, ::-1, :]) and np.array_equal(pn, -p)
    assert (np.diff(q, axis=1) >= 0).all() and np.array_equal(p, q[:, 4, :])
    g = dict(targets=x, past=[np.ones(3)], known=[])
    (_, qg), = groups(MODEL, [g], H, None)
    (_, qgn), = groups(MODEL, [flip.flip_group(g)], H, None)
    assert np.array_equal(qgn, -qg[:, ::-1, :])


def test_flipped_call_changes_only_the_targets():
    calls = []
    t = np.array([1, np.nan, 3], np.float32)
    past, cx, fut = np.array([np.nan, 1, 0], np.float32), np.array([1, 0, 1], np.float32), np.array([0, 1, 1, 0], np.float32)
    g = dict(targets=[t], past=[past], known=[(cx, fut)])
    wrapped(calls)[1](MODEL, [g], H, None)
    plain, neg = calls[0][0], calls[1][0]
    assert plain is g and neg["past"][0] is past and neg["known"][0][0] is cx and neg["known"][0][1] is fut
    assert np.array_equal(neg["targets"][0], -t, equal_nan=True)


def test_group_fallback_flips_once():
    n = []

    def lp(model, series, horizon, device, **kw):
        n.append(1)
        return fake_plain(model, series, horizon, device)

    def groups_fn(model, groups, horizon, device, **kw):
        return [plain(model, g["targets"], horizon, device) for g in groups]
    plain, grp = flip.wrap(lp, groups_fn, C)
    grp(MODEL, [dict(targets=[np.ones(3, np.float32)])], H, None)
    assert len(n) == 2


def test_asymmetric_levels_fail():
    with pytest.raises(ValueError, match="symmetric"):
        wrapped()[0](SimpleNamespace(quantile_levels=[.1, .5, .8]), [np.ones(3)], H, None)
