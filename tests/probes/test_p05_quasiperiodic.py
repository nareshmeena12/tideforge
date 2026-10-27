"""P5: naive is >= 1.5x the oracle in every cell (measured 1.66-3.10 on n = 48); the oracle is calibrated; deterministic."""
from helpers import same_sha_twice, scores

import tideforge.probes.p05_quasiperiodic as P


def test_p5_oracle_naive():
    ps, s = scores(P)
    for c in ps.cells:
        assert s[("naive", c.name, "sql_vs_oracle", "main", "main")] > P.THRESHOLDS["naive_vs_oracle_min"], c.name
        assert abs(s[("oracle", c.name, "cov80", "main", "main")] - 0.8) < 0.05


def test_p5_deterministic():
    assert same_sha_twice(P)
