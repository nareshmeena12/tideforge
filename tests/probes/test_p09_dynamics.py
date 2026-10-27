"""P9: finite trajectories, a calibrated oracle, naive far off (measured 3.8-13.1 on n = 48); deterministic."""
import numpy as np
from helpers import same_sha_twice, scores

import tideforge.probes.p09_dynamics as P


def test_p9_oracle_naive():
    ps, s = scores(P, n=24)
    for c in ps.cells:
        assert np.isfinite(c.arrays["targets"]).all() and np.isfinite(c.arrays["future"]).all()
        assert s[("naive", c.name, "sql_vs_oracle", "main", "main")] > P.THRESHOLDS["naive_vs_oracle_min"], c.name
        assert abs(s[("oracle", c.name, "cov80", "main", "main")] - 0.8) < 0.07


def test_p9_deterministic():
    assert same_sha_twice(P, n=4)
