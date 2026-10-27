"""P8: the oracle is coherent and calibrated; parent = sum of children in the data; naive is off by the measured ratios."""
import numpy as np
from helpers import same_sha_twice, scores

import tideforge.probes.p08_hierarchy as P


def test_p8_oracle_naive():
    ps, s = scores(P)
    T = P.THRESHOLDS
    for c in ps.cells:
        t, f = c.arrays["targets"].astype(float), c.arrays["future"]
        assert np.allclose(t[:, 0], t[:, 1:].sum(1), atol=1e-3) and np.allclose(f[:, 0], f[:, 1:].sum(1))
        for mode in c.modes:
            assert s[("oracle", c.name, "coherence", mode, "main")] <= T["oracle_coherence_max"]
            r = s[("naive", c.name, "sql_vs_oracle", mode, "main")]
            assert r > T["naive_measured_below"].get(c.name, T["naive_vs_oracle_min"]), (c.name, r)


def test_p8_deterministic():
    assert same_sha_twice(P)
