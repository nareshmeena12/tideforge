"""P1: the oracle gains from the covariate in every mechanism cell and not in the decoy; naive is far from the oracle; the
build is deterministic."""
from helpers import same_sha_twice, scores

import tideforge.probes.p01_covariates as P


def test_p1_oracle_naive_control():
    ps, s = scores(P)
    T = P.THRESHOLDS
    for c in ps.cells:
        mech = c.name.split("@")[0]
        r = s[("oracle", c.name, "sql_on_vs_hidden", "on", "main")]
        if c.control:
            lo, hi = T["decoy_ratio"]
            assert lo <= r <= hi, (c.name, r)
        else:
            assert r < T["oracle_on_hidden_max"].get(mech, T["oracle_on_hidden_max"]["default"]), (c.name, r)
        ratio = s[("naive", c.name, "sql", "hidden", "main")] / s[("oracle", c.name, "sql", "on", "main")]
        assert ratio > T["naive_measured_below"].get(mech, T["naive_vs_oracle_min"]), (c.name, ratio)
        assert abs(s[("oracle", c.name, "cov80", "on", "main")] - 0.8) < 0.05


def test_p1_deterministic():
    assert same_sha_twice(P)
