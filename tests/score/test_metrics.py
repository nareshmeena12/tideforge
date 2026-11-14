"""Our metrics equal fev's evaluation_summary on the same forecasts; seasonal naive matches the BUILD_REPORT baseline."""
import numpy as np
import pytest

from tideforge.score import metrics as MT
from tideforge.score import release as RL
from tideforge.score import windows as W
from tideforge.score.adapters.naive import SeasonalNaive


def to_fev(cache, q):
    """(R, D, Q, H) -> fev's predictions_per_window."""
    levels = [str(x) for x in cache.info["quantile_levels"]]
    out = []
    for w in range(cache.info["n_windows"]):
        sel = np.flatnonzero(cache.r_win == w)
        d = {}
        for di, col in enumerate(cache.info["target_columns"]):
            d[col] = [{"predictions": q[r, di, 4].tolist(), **{lv: q[r, di, j].tolist() for j, lv in enumerate(levels)}}
                      for r in sel]
        out.append(d)
    return out


@pytest.fixture
def caches(release, tmp_path):
    specs = RL.Release(release).tasks("full", "tuning")
    return specs, W.build(specs, tmp_path / "cache", workers=1)


def test_metrics_equal_fev(caches):
    import fev
    specs, cs = caches
    rng = np.random.default_rng(0)
    for s, c in zip(specs, cs):
        R, D, H = c.info["n_rows"], len(c.info["target_columns"]), c.info["horizon"]
        base = np.asarray(c.y_true).transpose(0, 2, 1)[:, :, None, :]
        base = np.nan_to_num(base, nan=20.0) + rng.normal(0, 2, (R, D, 1, H))
        q = np.sort(base + rng.normal(0, 1, (R, D, 9, H)), axis=2)
        ours = MT.score_task(c, q)
        t = fev.Task(**s["kw"])
        ref = t.evaluation_summary(to_fev(c, q), model_name="x")
        for k in ("MASE", "WQL", "SQL", "test_error"):
            assert np.isclose(ours[k], ref[k], rtol=1e-6, atol=0), (c.name, k, ours[k], ref[k])
        assert ours["num_forecasts"] == ref["num_forecasts"]


def naive_q(c, kind):
    nv = SeasonalNaive(kind)
    D, R, H, S = len(c.info["target_columns"]), c.info["n_rows"], c.info["horizon"], c.info["seasonality"]
    q = np.zeros((R, D, 9, H))
    for d in range(D):
        q[:, d] = nv.predict([c.past(r, d) for r in range(R)], H, season=S)
    return q


def test_naive_matches_fev_and_build_report(caches):
    import fev
    from tideforge.eval import thirdparty
    from tideforge.tfbench.baseline import score as bl_score
    S_ = thirdparty.score_twins()
    specs, cs = caches
    for s, c in zip(specs, cs):
        t = fev.Task(**s["kw"])
        t.load_full_dataset(num_proc=0)
        ref = t.evaluation_summary(S_.naive_preds(t), model_name="seasonal_naive")
        ours = MT.score_task(c, naive_q(c, "fev"))
        for k in ("MASE", "WQL", "SQL"):
            assert np.isclose(ours[k], ref[k], rtol=1e-9), (c.name, k)
        if c.name in ("b__h/short/target", "c__d/short/target"):        # no NaN: fev's and the report's averaging agree
            b = bl_score(s["kw"]["dataset_path"], s["kw"])
            mine = MT.score_task(c, naive_q(c, "tfbench"))
            for k in ("MASE", "WQL", "SQL"):
                assert np.isclose(mine[k], b[k], rtol=1e-6), (c.name, k, mine[k], b[k])


def test_refuses_metrics_needing_more_past():
    with pytest.raises(NotImplementedError):
        MT.metric_objects({"task_name": "t", "eval_metric": "RMSSE", "extra_metrics": []})
