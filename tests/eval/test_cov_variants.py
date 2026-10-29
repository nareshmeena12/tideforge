"""E0.10 covariate variants: which columns count as categorical, target encoding reads the context only, the order is by
|correlation| within each role, the swaps restrict fev to covariate tasks and are always restored (fakes on CPU)."""
from types import SimpleNamespace

import numpy as np
import pytest

import tidemark.eval.score_twins as S
import tidemark.real.score_groups as G
from tideforge.eval import cov_variants as V
from tideforge.eval.nan_variants import CODE, VARIANTS, use_variant, variant_of


def test_registered_with_the_variant_switch():
    for v in V.VARIANTS:
        assert v in VARIANTS and CODE[v] == V.CODE[v]
    assert variant_of("ft3maintencocorron_s0") == "both" and variant_of("ft3mainocorr_s0") == "order_corr"
    assert variant_of("ft3maintenc_s0") == "target_enc"


def test_is_categorical():
    assert V.is_categorical([np.array(["a", "b"]), np.array(["c"])])
    assert V.is_categorical([np.array([0, 1, 2, 2])])
    assert not V.is_categorical([np.array([0, 1, 1, 0])])            # two values: encoding is affine, left alone
    assert not V.is_categorical([np.array([0.5, 1.0, 2.0])])
    assert not V.is_categorical([np.arange(40)])


def test_target_encoding_uses_the_context_only():
    codes_ctx = np.array([0, 1, 2, 0, 1, 2], np.float32)
    y = np.array([1, 10, 100, 3, 12, 102], np.float64)
    fut = np.array([2, 0, 5, np.nan], np.float32)                  # 5 unseen in the context
    c, f = V.target_encode(codes_ctx, fut, y)
    assert np.allclose(c, [2, 11, 101, 2, 11, 101])
    assert np.allclose(f[:3], [101, 2, y.mean()]) and np.isnan(f[3])
    # changing the target's future cannot change anything: the function never sees it
    c2, f2 = V.target_encode(codes_ctx, fut, y)
    assert np.array_equal(f, f2, equal_nan=True)


def test_order_by_correlation_within_role():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    noise, weak, strong = rng.normal(size=200), y + 3 * rng.normal(size=200), y + 0.1 * rng.normal(size=200)
    g = dict(targets=[y], known=[(noise, np.zeros(4)), (strong, np.ones(4)), (weak, np.ones(4) * 2)],
             past=[np.zeros(200), -strong])
    o = V.order_lanes(g)
    assert [k[1][0] for k in o["known"]] == [1.0, 2.0, 0.0]
    assert o["past"][0] is g["past"][1]                            # constant lane last
    assert o["targets"] is g["targets"]


def fake_window():
    """A fev-like window: one target, a string known covariate (3 values) and a numeric past covariate."""
    T, H, n = 30, 4, 2
    rng = np.random.default_rng(1)
    cat = np.array(["hi", "lo", "mid"])
    k = [cat[rng.integers(0, 3, T)] for _ in range(n)]
    kf = [cat[rng.integers(0, 3, H)] for _ in range(n)]
    tgt = [np.where(k[i] == "hi", 100.0, np.where(k[i] == "mid", 10.0, 1.0)) for i in range(n)]
    past = {"y": tgt, "k": k, "p": [rng.normal(size=T) for _ in range(n)]}
    fut = {"k": kf}
    return SimpleNamespace(target_columns=["y"], known_dynamic_columns=["k"], past_dynamic_columns=["p"],
                           get_input_data=lambda: (past, fut))


def test_window_groups_swap_and_restore():
    orig_wg, orig_load = G.window_groups, S.load_fev_tasks
    w = fake_window()
    fill = lambda y, s: np.asarray(y, np.float32)                  # noqa: E731
    base = G.window_groups(w, 64, 1, fill)
    assert set(np.unique(base[0]["known"][0][0])) <= {0.0, 1.0, 2.0}  # baseline: alphabetical codes
    with use_variant("target_enc"):
        enc = G.window_groups(w, 64, 1, fill)
    assert set(np.unique(enc[0]["known"][0][0])) <= {1.0, 10.0, 100.0}
    assert np.array_equal(enc[0]["past"][0], base[0]["past"][0])   # numeric covariate untouched
    assert G.window_groups is orig_wg and S.load_fev_tasks is orig_load
    with pytest.raises(RuntimeError):
        with use_variant("both"):
            raise RuntimeError("boom")
    assert G.window_groups is orig_wg and S.load_fev_tasks is orig_load


def test_cov_tasks_only(monkeypatch):
    tasks = [SimpleNamespace(known_dynamic_columns=[], past_dynamic_columns=[]),
             SimpleNamespace(known_dynamic_columns=["a"], past_dynamic_columns=[]),
             SimpleNamespace(known_dynamic_columns=[], past_dynamic_columns=["b"])]
    monkeypatch.setattr(S, "load_fev_tasks", lambda paths: tasks)
    with use_variant("order_corr"):
        got = S.load_fev_tasks(None)
    assert got == tasks[1:]


def test_report_rows():
    w = fake_window()
    t = SimpleNamespace(task_name="fake", known_dynamic_columns=["k"], past_dynamic_columns=["p"], iter_windows=lambda: iter([w]))
    t0 = SimpleNamespace(task_name="plain", known_dynamic_columns=[], past_dynamic_columns=[], iter_windows=lambda: iter([w]))
    (r,) = V.report_rows([t, t0])
    assert r["task"] == "fake" and r["categorical"] == 1 and r["max_categories"] == 3 and r["dropped"] == 0
