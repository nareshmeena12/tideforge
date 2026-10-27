"""P10 (coverage and PIT, not sql): the oracle is calibrated, naive is not; deterministic."""
from helpers import same_sha_twice, scores

import tideforge.probes.p10_tails as P


def test_p10_calibration():
    ps, s = scores(P)
    T = P.THRESHOLDS
    for c in ps.cells:
        g = lambda m, k: s[(m, c.name, k, "main", "main")]
        assert T["oracle_cov"][0] <= g("oracle", "cov80") <= T["oracle_cov"][1], c.name
        assert g("oracle", "pit_tv") <= T["oracle_pit_tv_max"], c.name
        assert g("naive", "pit_tv") >= T["naive_pit_tv_min"], c.name
        assert abs(g("naive", "cov80") - 0.8) >= T["naive_cov_off_min"], c.name


def test_p10_deterministic():
    assert same_sha_twice(P)
