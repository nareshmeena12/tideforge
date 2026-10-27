"""P2: the flag helps the oracle on the event days in every cell but the zero-effect control; naive is far off."""
from helpers import same_sha_twice, scores

import tideforge.probes.p02_calendar as P


def test_p2_oracle_naive_control():
    ps, s = scores(P)
    T = P.THRESHOLDS
    for c in ps.cells:
        r = s[("oracle", c.name, "event_sql_on_vs_hidden", "on", "main")]
        if c.control:
            assert T["control_ratio"][0] <= r <= T["control_ratio"][1], (c.name, r)
        else:
            assert r < T["oracle_event_on_hidden_max"], (c.name, r)
        ratio = s[("naive", c.name, "sql", "hidden", "main")] / s[("oracle", c.name, "sql", "on", "main")]
        assert ratio > T["naive_measured_below"].get(c.name, T["naive_vs_oracle_min"]), (c.name, ratio)


def test_p2_deterministic():
    assert same_sha_twice(P)
