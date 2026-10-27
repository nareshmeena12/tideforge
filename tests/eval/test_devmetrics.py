"""Dev metrics: timebench's per-window definitions (hand-computed and against timebench itself), TIME's aggregation, the
domain cap, the deterministic seasonal-naive band (against statsforecast when installed)."""
import numpy as np
import pandas as pd
import pytest

from tideforge.devdata import build as B
from tideforge.eval import devmetrics as M
from tideforge.eval import naive_time as N


def test_window_metrics_hand_computed(tb):
    ctx = np.array([1.0, 2.0, 3.0, 4.0])          # S = 1: mean |diff| = 1
    y = np.array([5.0, 7.0])
    q = np.full((9, 2), 6.0)                       # a point forecast: CRPS = sum|e| / sum|y| = 2 / 12
    mase, crps = M.window_metrics(q, y, ctx, 1)
    assert mase == pytest.approx(1.0) and crps == pytest.approx(2 / 12)


def test_window_metrics_equal_timebench(tb):
    rng = np.random.default_rng(0)
    ctx, y = rng.normal(size=200), rng.normal(size=24)
    ctx[[5, 50]] = np.nan
    q = np.sort(rng.normal(size=(9, 24)), axis=0)
    ref = tb(q[None, None, :, None, :], y[None, None, None, :], ctx[None, None, None, :], seasonality=24)
    assert M.window_metrics(q, y, ctx, 24) == (float(ref["MASE"][0, 0, 0]), float(ref["CRPS"][0, 0, 0]))


def _win(vals, naive=1.0):
    return pd.DataFrame([dict(dataset_id=d, term=t, domain=dom, MASE=v, CRPS=v, MASE_naive=naive, CRPS_naive=naive)
                         for d, t, dom, v in vals])


def test_aggregation_as_time():
    # per (dataset, term): mean over windows / naive's mean; per dataset: gmean over terms
    w = _win([("a", "short", "x", 1.0), ("a", "short", "x", 3.0), ("a", "long", "x", 8.0), ("b", "short", "y", 0.5)])
    full = M.dataset_scores(M.pair_ratios(w))
    ds = full.set_index("dataset_id")["r_MASE"]
    assert ds["a"] == pytest.approx(np.sqrt(2.0 * 8.0)) and ds["b"] == pytest.approx(0.5)
    o = M.overall(full, {"a": 0.25, "b": 0.75}, ci=False)
    assert o["MASE"][0] == pytest.approx(np.exp(0.25 * np.log(4.0) + 0.75 * np.log(0.5)))


def test_baseline_identical_scores_exactly_one():
    w = _win([(d, t, "x", 2.0) for d in "abc" for t in ("short", "long")], naive=2.0)
    o = M.overall(M.dataset_scores(M.pair_ratios(w)), {"a": .2, "b": .3, "c": .5})
    assert o["MASE"] == (1.0, 1.0, 1.0) and o["CRPS"][0] == 1.0


def test_domain_cap_holds_in_the_score():
    n = {f"d{i}": 1 for i in range(14)} | {"big": 30}
    dw = B.domain_weights(n)
    ids = [(f"d{i}", f"d{i}") for i in range(14)] + [(f"big{j}", "big") for j in range(30)]
    weights = {d: dw[dom] / n[dom] for d, dom in ids}
    base = _win([(d, "short", dom, 1.0) for d, dom in ids])
    worse = _win([(d, "short", dom, 2.0 if dom == "big" else 1.0) for d, dom in ids])
    o0 = M.overall(M.dataset_scores(M.pair_ratios(base)), weights, ci=False)["MASE"][0]
    o1 = M.overall(M.dataset_scores(M.pair_ratios(worse)), weights, ci=False)["MASE"][0]
    assert o0 == pytest.approx(1.0) and np.log(o1) == pytest.approx(B.DOMAIN_CAP * np.log(2.0))   # 30 datasets weigh 10%
    s = M.slices(worse, weights, keys=("domain",)).set_index("value")
    assert s.loc["big", "MASE"] == pytest.approx(2.0) and s.loc["d0", "MASE"] == pytest.approx(1.0)


def test_hist_bands():
    assert [M.hist_band(n) for n in (1, 64, 65, 512, 513, 2048, 2049)] == ["1-64", "1-64", "65-512", "65-512", "513-2048",
                                                                          "513-2048", "2049+"]


def test_naive_band_hand_computed():
    y = np.array([1.0, 2.0, 1.0, 4.0, 1.0, 2.0])  # S = 2: diffs y_t - y_{t-2} = [0, 2, 0, -2] -> sigma = sqrt(8 / 4)
    q = N.band(y, 3, 2)
    s = np.sqrt(2.0)
    assert np.allclose(q[4], [1.0, 2.0, 1.0])
    assert np.allclose(q[:, 2] - q[4, 2], N.Z9 * s * np.sqrt(2.0)) and np.allclose(q[:, 0] - q[4, 0], N.Z9 * s)
    assert np.isnan(N.band(np.ones(3), 2, 5)).all()


def test_naive_band_equals_statsforecast():
    sf = pytest.importorskip("statsforecast")
    from statsforecast.models import SeasonalNaive
    rng = np.random.default_rng(3)
    for S, n, H in ((24, 500, 48), (7, 100, 30), (1, 50, 5)):
        y = 5 + np.sin(np.arange(n) * 2 * np.pi / max(S, 2)) + rng.normal(0, 0.3, n)
        r = SeasonalNaive(season_length=S).forecast(y=y, h=H, level=[20, 40, 60, 80])
        cols = [r["lo-80"], r["lo-60"], r["lo-40"], r["lo-20"], r["mean"], r["hi-20"], r["hi-40"], r["hi-60"], r["hi-80"]]
        assert np.allclose(N.band(y, H, S), np.stack(cols), rtol=1e-5, atol=1e-5), (S, sf.__version__)
