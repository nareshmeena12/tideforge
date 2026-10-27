"""P4: the oracle adopts a step (1) and ignores a transient (0); naive is off by the measured ratios; deterministic."""
from helpers import same_sha_twice, scores

import tideforge.probes.p04_breaks as P


def test_p4_oracle_naive():
    ps, s = scores(P)
    T = P.THRESHOLDS
    for c in ps.cells:
        if not c.control:
            lo, hi = T["oracle_adoption"][c.name]
            assert lo <= s[("oracle", c.name, "adoption", "main", "main")] <= hi, c.name
            assert ("oracle", c.name, "width80_ratio_vs_control", "main", "main") in s
        r = s[("naive", c.name, "sql_vs_oracle", "main", "main")]
        assert r > T["naive_measured_below"].get(c.name, T["naive_vs_oracle_min"]), (c.name, r)


def test_p4_deterministic():
    assert same_sha_twice(P)
