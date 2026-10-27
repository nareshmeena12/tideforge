"""P7 (zero mass, not sql): the oracle's zero-mass error is small and naive's is not; the oracle quantiles are integers;
the 9-quantile estimate tracks the exact P(y=0) to about its 0.1 resolution; deterministic."""
import numpy as np
from helpers import same_sha_twice, scores

import tideforge.probes.p07_counts as P
from tideforge.probes import metrics as M


def test_p7_zero_mass():
    ps, s = scores(P)
    T = P.THRESHOLDS
    for c in ps.cells:
        assert s[("oracle", c.name, "zero_mass_err", "main", "main")] <= T["oracle_zero_err_max"], c.name
        assert s[("naive", c.name, "zero_mass_err", "main", "main")] > T["naive_zero_err_min"], c.name
        q = c.arrays["oracle_q"]
        assert np.array_equal(q, np.round(q))
        est = M.p_zero(q)
        assert np.abs(est - c.arrays["p_zero_true"]).max() <= 0.1 + 1e-9


def test_p7_deterministic():
    assert same_sha_twice(P)
