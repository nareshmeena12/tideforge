"""P6: the oracle's gaps/clean ratio is exactly 1 (it knows the state); the control's gaps view equals the clean view for any
model; naive is off by the measured ratios; the gaps really are NaN in the main view; deterministic."""
import numpy as np
from helpers import same_sha_twice, scores

import tideforge.probes.p06_gaps as P


def test_p6_oracle_naive_control():
    ps, s = scores(P)
    T = P.THRESHOLDS
    for c in ps.cells:
        lo, hi = T["oracle_ratio"]
        assert lo <= s[("oracle", c.name, "sql_gaps_vs_clean", "main", "main")] <= hi
        r = s[("naive", c.name, "sql_vs_oracle", "main", "clean")]
        assert r > T["naive_measured_below"].get(c.name, T["naive_vs_oracle_min"]), (c.name, r)
        share = np.isnan(c.arrays["targets"]).mean()
        assert (share == 0) == c.control
    assert s[("naive", "control", "sql_gaps_vs_clean", "main", "main")] == 1.0


def test_p6_deterministic():
    assert same_sha_twice(P)
