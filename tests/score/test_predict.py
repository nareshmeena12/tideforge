"""Batching changes nothing: the pooled engine gives the old per-task scorer's forecasts (score_twins.fev_predictions for
groups OFF, score_groups.fev_groups_predictions for groups ON), on CPU in fp32 with a tiny random group model, for a plain,
a covariate (numeric past, numeric + string known), a univariate-targets and a joint task; batch size 1 equals the default
batch; a crashed run resumes to the same result."""
import numpy as np
import pytest
import torch

from tideforge.eval import thirdparty
from tideforge.score import predict as PR
from tideforge.score import release as RL
from tideforge.score import windows as W
from tideforge.score.adapters.ft3 import Ft3
from tideforge.score.adapters.naive import SeasonalNaive

TOL = 2e-4          # fp32 on CPU, different batch composition / packing: float noise only


@pytest.fixture
def setup(release, tmp_path, tiny_ckpt):
    specs = RL.Release(release).tasks("full", "tuning")
    caches = W.build(specs, tmp_path / "cache", workers=1)
    return specs, caches, Ft3(tiny_ckpt, device="cpu", name="tiny", rescale=False), tmp_path     # the old scorer's path


def old_q(spec, cache, model, mode):
    """The old scorer's quantiles for this task, as (R, D, Q, H) in cache row order."""
    import fev
    S_, G, C = thirdparty.score_twins(), thirdparty.score_groups(), thirdparty.ft3_core()
    t = fev.Task(**spec["kw"])
    t.load_full_dataset(num_proc=0)
    cpu = torch.device("cpu")
    if mode == "off":
        s, i = S_.fev_inputs(t)
        pr = S_.fev_predictions(model, t, s, i, cpu, C.forecast_left_padded)
    else:
        pr = G.fev_groups_predictions(model, t, cpu, C.forecast_groups, C.forecast_left_padded, S_.fill_season, S_.MAX_CTX,
                                      S_.BATCH)
    levels = [str(x) for x in t.quantile_levels]
    rows = []
    for w in pr:
        n = len(next(iter(w.values())))
        for k in range(n):
            rows.append([[w[col][k][lv] for lv in levels] for col in cache.info["target_columns"]])
    return np.asarray(rows, dtype=np.float64)


@pytest.mark.parametrize("mode", ["off", "on"])
def test_pooled_equals_old_scorer(setup, mode):
    specs, caches, pred, tmp = setup
    q, st = PR.run(caches, pred, mode, tmp / "out", "tiny")
    assert st["series"] > 0
    for s, c in zip(specs, caches):
        ref = old_q(s, c, pred.model, mode)
        assert ref.shape == q[c.name].shape, c.name
        np.testing.assert_allclose(q[c.name], ref, rtol=TOL, atol=TOL * 10, err_msg=f"{c.name} {mode}")
    by = {c.name: c for c in caches}
    assert PR.use_groups(by["c__d/short/joint"], pred, "on") and PR.use_groups(by["b__h/short/cov"], pred, "on")
    assert not PR.use_groups(by["b__h/short/target"], pred, "on") and not PR.use_groups(by["c__d/short/joint"], pred, "off")


def test_joint_univariate_model_scores_each_target(setup):
    specs, caches, _, tmp = setup
    by = {c.name: c for c in caches}
    nv = SeasonalNaive("fev")
    assert not PR.use_groups(by["c__d/short/joint"], nv, "on")
    q, _ = PR.run(caches, nv, "on", tmp / "nv", "naive")
    c = by["c__d/short/joint"]
    from tideforge.score.adapters.naive import fev_naive
    for r in range(c.info["n_rows"]):
        for d in range(2):
            assert np.allclose(q[c.name][r, d, 4], fev_naive(c.past(r, d), c.info["horizon"], c.info["seasonality"]))


def test_batch_size_one_equals_default(setup, tiny_ckpt):
    specs, caches, pred, tmp = setup
    one = Ft3(tiny_ckpt, device="cpu", batch_size=1, rescale=False)
    for mode in ("off", "on"):
        a, _ = PR.run(caches, pred, mode, tmp / "a", "tiny")
        b, _ = PR.run(caches, one, mode, tmp / "b", "tiny1")
        for n in a:
            np.testing.assert_allclose(a[n], b[n], rtol=TOL, atol=TOL * 10)


def test_resume_after_crash(setup, monkeypatch):
    specs, caches, pred, tmp = setup
    full, _ = PR.run(caches, pred, "off", tmp / "ref", "tiny", chunk_points=500)
    calls = {"n": 0}
    orig = pred.predict

    def crash(*a, **k):
        calls["n"] += 1
        if calls["n"] == 2:
            raise KeyboardInterrupt("node died")
        return orig(*a, **k)
    monkeypatch.setattr(pred, "predict", crash)
    with pytest.raises(KeyboardInterrupt):
        PR.run(caches, pred, "off", tmp / "crash", "tiny", chunk_points=500)
    monkeypatch.setattr(pred, "predict", orig)
    again, st = PR.run(caches, pred, "off", tmp / "crash", "tiny", chunk_points=500)
    assert st["reused"] >= 1
    for n in full:
        assert np.array_equal(full[n], again[n])


def test_readonly_inputs_with_nan(tiny_ckpt):
    """The window cache hands out read-only memmap views; forecast_groups fills NaN in place (Perlmutter: 'assignment
    destination is read-only'). The adapter must pass copies and leave the inputs untouched, in both paths."""
    pred = Ft3(tiny_ckpt, device="cpu", name="tiny")
    rng = np.random.default_rng(0)

    def ro(n, nan=True):
        x = rng.normal(size=n).astype(np.float32)
        if nan:
            x[n // 2: n // 2 + 3] = np.nan
        x.setflags(write=False)
        return x
    tg = [[ro(200), ro(200)], [ro(150)]]
    past = [[ro(200)], [ro(150)]]
    known = [[(ro(200), ro(24))], [(ro(150), ro(24))]]        # NaN in the known future: forecast_groups forward-fills it
    assert pred.groups, "tiny_ckpt must have group attention, or forecast_groups is never reached"
    every = [x for g in tg for x in g] + [x for g in known for c in g for x in c]
    before = [x.copy() for x in every]
    out = pred.predict(tg, 24, past_covariates=past, known_covariates=known)
    assert len(out) == 2 and all(np.isfinite(q).all() for q in out)
    q = pred.predict([ro(200), ro(120)], 24)
    assert np.isfinite(q).all()
    for a, b in zip(every, before):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("a,b", [(1e12, 3e12), (1e-6, 2e-6), (50.0, -7.0)])
def test_rescale_is_scale_invariant(tiny_ckpt, a, b):
    """Rescale on: forecasting a * y + b gives a * forecast(y) + b (a > 0), on both paths, for
    the extreme scales that broke the raw path (thermobench ~1e12, GOES / WDI ~1e-6)."""
    pred = Ft3(tiny_ckpt, device="cpu", name="tiny")
    rng = np.random.default_rng(1)
    y = [np.sin(np.arange(n) / 5) + 0.1 * rng.normal(size=n) for n in (300, 200)]
    y[0][40:45] = np.nan
    fut = rng.normal(size=24)

    def lin(q):
        q = a * q + b
        return np.sort(q, axis=-2)
    q0 = pred.predict(y, 24)
    q1 = pred.predict([a * v + b for v in y], 24)
    np.testing.assert_allclose(q1, lin(q0), rtol=1e-4, atol=1e-4 * abs(a))
    g0 = pred.predict([y], 24, past_covariates=[[y[1]]], known_covariates=[[(y[0], fut)]])
    g1 = pred.predict([[a * v + b for v in y]], 24, past_covariates=[[y[1]]], known_covariates=[[(y[0], fut)]])
    np.testing.assert_allclose(g1[0], lin(g0[0]), rtol=1e-4, atol=1e-4 * abs(a))


def test_affine_edge_cases():
    from tideforge.score.adapters.ft3 import affine
    z, c, s = affine([5.0, 5.0, 5.0])
    assert c == 5.0 and s > 0 and np.all(z == 0)
    z, c, s = affine([0.0, 0.0])
    assert s == 1.0
    z, c, s = affine([np.nan, np.nan])
    assert np.isnan(z).all()
    z, c, s = affine([0, 0, 0, 0, 1, 0, 0, 3.0])                # intermittent: MAD 0 -> std
    assert np.isclose(s, np.std([0, 0, 0, 0, 1, 0, 0, 3.0]))
