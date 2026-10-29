"""D4 rules find a planted effect (and nothing in noise), D5 finds an uncovered region and the config that covers it, D6 tells a
planted difference apart and not identical data, synth windows keep the lengths and family labels (synthetic fixtures)."""
import numpy as np
import pandas as pd
import pytest

import tideforge.gen as G
from tideforge.diagnose import coverage as CV, features as FE, realism as RL, rules as RU, synth as SY


def planted(rng, effect=1.0, n_ds=30, n_w=40):
    rows = []
    for d in range(n_ds):
        x = rng.uniform(0, 1, n_w)
        fix = effect * (x > 0.8) + 0.2 * rng.normal(size=n_w)
        rows += [dict(dataset_id=f"d{d}", x=xi, z=rng.normal(), fix=fi) for xi, fi in zip(x, fix)]
    return pd.DataFrame(rows)


def test_rules_find_the_planted_effect():
    df = planted(np.random.default_rng(0))
    t = RU.binned(df, "x")
    top = t.iloc[-1]
    assert top.effect > 0.5 and top.lo > 0 and t.iloc[0].hi < 0
    lines = RU.report([t, RU.binned(df, "z")])
    assert lines and lines[0].startswith("we lose more when x in")
    assert not any(" z in " in s for s in lines[:2])
    node = RU.tree(df, ["x", "z"], depth=1)
    assert node.feature == "x" and 0.75 < node.thr < 0.85 and node.right.value > node.left.value + 0.5


def test_rules_quiet_on_noise():
    df = planted(np.random.default_rng(1), effect=0.0)
    lines = RU.report([RU.binned(df, f) for f in ("x", "z")])
    assert len(lines) <= 1                                         # 10 bins at 95%: at most a chance hit


def two_clusters(rng, n=300):
    cols = FE.SYNTH_COMPARABLE
    a = pd.DataFrame(rng.normal(0, 0.05, (n, len(cols))) + 0.2, columns=cols)
    b = pd.DataFrame(rng.normal(0, 0.05, (n, len(cols))) + 0.8, columns=cols)
    real = pd.concat([a, b], ignore_index=True)
    real["dataset_id"] = [f"a{i % 10}" for i in range(n)] + [f"b{i % 10}" for i in range(n)]
    real["domain"] = ["A"] * n + ["B"] * n
    real["series_id"], real["window"] = np.arange(2 * n), 0
    return real, a.copy(), pd.concat([a, b], ignore_index=True)


def test_coverage_finds_the_empty_region():
    rng = np.random.default_rng(2)
    real, only_a, both = two_clusters(rng)
    cov, info = CV.coverage(real, {"b0": only_a + rng.normal(0, 0.01, only_a.shape), "new": both + rng.normal(0, 0.01, both.shape)})
    unc = cov["uncovered:b0"].to_numpy()
    assert unc[300:].mean() > 0.95 and unc[:300].mean() < 0.05
    assert cov["uncovered:new"].mean() < 0.05
    reg = CV.regions(cov, info)
    assert set(reg["domain"]) == {"B"} and (reg["best_config"] == "new").all()
    assert reg["covered_by:new"].min() > 0.9


def test_realism_planted_and_null():
    rng = np.random.default_rng(3)
    cols = ["f1", "f2", "f3"]
    real = pd.DataFrame(rng.normal(size=(600, 3)), columns=cols)
    real["dataset_id"] = [f"d{i % 20}" for i in range(600)]
    syn = pd.DataFrame(rng.normal(size=(600, 3)), columns=cols)
    syn["family"] = [f"f{i % 4}" for i in range(600)]
    null = RL.logistic(real, syn, cols)
    assert abs(null["auc"] - 0.5) < 0.08
    syn2 = syn.copy()
    syn2.loc[syn2.family == "f0", "f2"] += 3.0                      # one family is off in one feature
    lg = RL.logistic(real, syn2, cols)
    assert lg["coef"].iloc[0]["feature"] == "f2"
    fam = RL.by_family(lg["score"], lg["label"], syn2["family"])
    assert fam.iloc[0]["family"] == "f0" and fam.iloc[0]["auc"] > 0.9


def test_auc():
    assert RL.auc([0, 1, 2, 3], [0, 0, 1, 1]) == 1.0 and RL.auc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5
    assert RL.auc([3, 2, 1, 0], [0, 0, 1, 1]) == 0.0


def test_cnn_tells_sines_from_noise():
    rng = np.random.default_rng(4)
    real = [np.sin(np.arange(n) * 2 * np.pi / 24) + 0.1 * rng.normal(size=n) for n in rng.integers(100, 400, 160)]
    syn = [rng.normal(size=n).cumsum() for n in rng.integers(100, 400, 160)]
    grp = np.r_[[f"r{i % 8}" for i in range(160)], [f"s{i % 8}" for i in range(160)]]
    res = RL.cnn(real, syn, grp)
    print(f"\ncnn: pooled AUC {res['auc']:.3f}, per-fold mean {res['auc_fold_mean']:.3f}")
    assert res["auc_fold_mean"] > 0.9


def test_synth_lengths_and_families():
    gc = G.b0()
    xs = SY.windows(gc, [40, 200, 1000], 12, seed=0)
    assert len(xs) == 12 and {len(x) for x in xs} <= {40, 200, 1000} and all(np.isfinite(x).all() for x in xs)
    xs, fams = SY.by_family(gc, [64, 300], 3, seed=0)
    prof = G.configure(gc)
    want = {f for f, w in G.GEN_PROFILES[prof]["weights"].items() if w > 0}
    assert set(fams) == want and len(xs) == 3 * len(want)
    f = FE.comparable_frame(xs[:5], [dict(family=x) for x in fams[:5]])
    assert set(FE.SYNTH_COMPARABLE) <= set(f.columns) and list(f["family"]) == fams[:5]
