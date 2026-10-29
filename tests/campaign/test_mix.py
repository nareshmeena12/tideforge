"""X3: the regression recovers a planted optimum from noisy synthetic short runs; sampling is deterministic and inside the
box; amounts scale and cap; the plan writes 28 mixes + 4 anchors; the rank check judges anchors and milestones together."""
import json
import math

import numpy as np
import pytest

from tideforge import expcfg
from tideforge.campaign import mix as X
from tideforge.campaign import stack as S


def test_regression_recovers_a_planted_optimum():
    rng = np.random.default_rng(3)
    d, n = 4, 28
    xs = np.log([0.5, 2.0, 1.0, 0.3])                       # the planted best multipliers
    X_ = rng.uniform(math.log(X.LO), math.log(X.HI), size=(n, d))
    y = math.log(0.7) + 0.05 * ((X_ - xs) ** 2).sum(1) + rng.normal(0, 0.005, n)   # noise ~ a 0.5% seed range
    m = X.fit(X_, y)
    x, v = X.best_mix(m)
    assert np.abs(x - xs).max() < 0.25 and abs(math.exp(v) - 0.7) < 0.01 and m["loo_r2"] > 0.8


def test_sampling_is_deterministic_and_in_the_box():
    a, b = X.sample(["covariates", "cycles"], 28, seed=0), X.sample(["covariates", "cycles"], 28, seed=0)
    assert a == b and all(X.LO <= v <= X.HI for m in a for v in m.values())
    assert a != X.sample(["covariates", "cycles"], 28, seed=1)


def test_apply_scales_amounts_and_caps():
    d = dict(base="b0", switches={"sc_load": {"share": 0.05}, "gaps_block": True, "no_ffill": True})
    out = X.apply(d, {"covariates": 2.0, "breaks_gaps": 10.0}, S.roles())
    assert out["switches"]["sc_load"]["share"] == 0.1
    assert out["switches"]["gaps_block"] == {"target_lanes": 0.95} and out["switches"]["no_ffill"] is True


def test_plan_writes_mixes_and_anchors(tmp_path):
    rec = X.plan("G1.8", ["E0.7", "G1.8", "G4.3", "G6.1"], n=32, points=1e10, seed=0, out_dir=tmp_path, campaigns=tmp_path)
    assert rec["groups"] == ["covariates"] and len(rec["runs"]) == 32
    e = expcfg.load_exp(tmp_path / "X3.r01.yaml")
    m = rec["runs"]["X3.r01"]["covariates"]
    assert e.train["target_points"] == 1e10
    assert dict(dict(e.gen.switches)["sc_load"])["share"] == round(min(0.95, 0.05 * m), 6)
    a4 = expcfg.load_exp(tmp_path / "X3.a4.yaml")
    assert a4.train["target_points"] == 1e10 and "gaps_scattered" in dict(a4.gen.switches)
    assert json.loads((tmp_path / "x3_mix.json").read_text())["runs"]["X3.a1"] == {"anchor": "E0.7"}
    with pytest.raises(ValueError, match="too few"):
        X.plan("G1.8", ["E0.7"], n=4, out_dir=tmp_path / "b", campaigns=tmp_path)


def test_rank_check():
    good = [(0.80, 0.70), (0.82, 0.72), (0.85, 0.75), (0.90, 0.80)]
    r = X.rank_check(good, [(0.81, 0.71), (0.95, 0.79)])
    assert r["anchors"]["rho"] == 1.0 and r["decision"] == "keep 10B" and "CHEAP EXTRA SIGNAL" in r["milestones"]["note"]
    bad = [(0.80, 0.80), (0.82, 0.72), (0.85, 0.75), (0.90, 0.70)]
    assert X.rank_check(bad, [])["decision"] == "rerun the short runs at 20B"
    assert X.rank_check([(1, 1)], [])["decision"].startswith("unknown")
