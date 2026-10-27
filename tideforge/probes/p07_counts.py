"""P7 counts and intermittency (gap G7). Daily, S = 7, L = 512, H = 28.
Intensity lambda_t = mean x weekly factor (mean 1, +-30%); half the series Poisson, half negative binomial (r 1-5,
variance lambda + lambda^2 / r). Cells by the mean rate:
    lam_lt_0.3   0.05-0.3   (mostly zeros)
    lam_0.3_1    0.3-1
    lam_1_3      1-3
    control      40-60      (high counts, nearly Gaussian; no zero mass)
oracle_q: the exact Poisson / NB quantiles per step; p_zero_true: the exact P(y = 0) per step.
Metrics: standard with the error scale = mean |y| of the context; sql_vs_oracle; zero_mass_err = |mean predicted P(y=0) -
observed zero share| over series and steps, with P(y=0) read from the 9 quantiles (the highest level whose quantile is
<= 0.5, interpolated to the next level; resolution about 0.1; metrics.p_zero); p_zero_pred and zero_share for reference.
The runner keeps the raw quantiles of every run, so a finer zero-mass estimate can be computed later.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from tideforge.probes import metrics as M
from tideforge.probes.common import Q9, Cell, ProbeSet, periodic, rng_for

PROBE, VERSION, SEED, N_PER_CELL = "P7", 1, 707, 256
L, H, S = 512, 28, 7
RATES = {"lam_lt_0.3": (0.05, 0.3), "lam_0.3_1": (0.3, 1.0), "lam_1_3": (1.0, 3.0), "control": (40.0, 60.0)}
# Measured (n = 48, 2026-10-27): zero_mass_err oracle 0.000-0.010, naive 0.062-0.132 (naive / oracle sql 1.52-1.89).
THRESHOLDS = dict(oracle_zero_err_max=0.05, naive_zero_err_min=0.05)


def _cell(name: str, n: int, seed: int) -> Cell:
    rng = rng_for(seed, name)
    lo, hi = RATES[name]
    mean = rng.uniform(lo, hi, size=(n, 1))
    lam = mean * (1 + 0.3 * periodic(rng, n, L + H, S, harmonics=2))
    lam = np.maximum(lam, 1e-3)
    nb = rng.random(n) < 0.5
    r = rng.uniform(1, 5, size=(n, 1))
    y = np.where(nb[:, None], rng.negative_binomial(r, r / (r + lam)), rng.poisson(lam)).astype(float)
    lf, rf = lam[:, L:], np.broadcast_to(r, (n, H))
    q = np.where(nb[:, None, None], stats.nbinom.ppf(Q9[None, :, None], rf[:, None, :], (rf / (rf + lf))[:, None, :]),
                 stats.poisson.ppf(Q9[None, :, None], lf[:, None, :]))
    p0 = np.where(nb[:, None], stats.nbinom.pmf(0, rf, rf / (rf + lf)), stats.poisson.pmf(0, lf))
    a = dict(targets=y[:, None, :L].astype(np.float32), future=y[:, None, L:], oracle_q=q[:, None].astype(float),
             p_zero_true=p0[:, None])
    return Cell(name, a, S, control=name == "control", desc=f"counts, mean rate {lo}-{hi}")


def build(n_per_cell: int = N_PER_CELL, seed: int = SEED) -> ProbeSet:
    return ProbeSet(PROBE, VERSION, seed, n_per_cell, [_cell(c, n_per_cell, seed) for c in RATES],
                    "counts: Poisson / NB at low rates, high-count control")


def metrics(cell: Cell, preds: dict) -> list:
    q = preds[("main", "main")]
    y, sc = cell.arrays["future"], M.scale_abs(cell.arrays["targets"])
    pz = M.p_zero(q)
    zero = (y == 0).astype(float)
    per = lambda a: a.reshape(len(a), -1).mean(1)[:, None]
    return M.standard(q, y, sc) + [
        M.vs_oracle(q, cell, sc),
        M.Metric("zero_mass_err", np.concatenate([per(pz), per(zero)], 1), "abs_diff"),
        M.Metric("p_zero_pred", per(pz), "mean"),
        M.Metric("zero_share", per(zero), "mean")]
