"""P3: the oracle brings back the seen-once / seen-twice event; naive is off by the measured ratios; deterministic."""
from helpers import same_sha_twice, scores

import tideforge.probes.p03_cycles as P


def test_p3_oracle_naive():
    ps, s = scores(P)
    T = P.THRESHOLDS
    for c in ps.cells:
        if c.name.startswith("year"):
            assert s[("oracle", c.name, "event_score", "main", "main")] > T["oracle_event_min"], c.name
        assert s[("oracle", c.name, "sql_vs_oracle", "main", "main")] == 1.0
        r = s[("naive", c.name, "sql_vs_oracle", "main", "main")]
        assert r > T["naive_measured_below"].get(c.name, T["naive_vs_oracle_min"]), (c.name, r)
        assert abs(s[("oracle", c.name, "cov80", "main", "main")] - 0.8) < 0.06


def test_p3_deterministic():
    assert same_sha_twice(P)
