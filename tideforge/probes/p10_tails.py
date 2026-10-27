"""P10 heavy tails (gap G10): calibration of intervals, not point skill. Returns, L = 1024, H = 32, S = 1.
    garch_t        GARCH(1,1) (beta 0.80-0.90, alpha 0.05-0.15 with alpha + beta <= 0.97, unconditional variance 1) with
                   Student-t innovations, nu 3-5, scaled to unit variance; 500 steps of burn-in
    garch_t_jumps  the same plus jumps: with probability 0.01 per step an extra N(0, 5^2)
    control        independent N(0, 1)
oracle_q: per step, the quantiles of 4000 Monte Carlo paths of the true process from its true state (the control: exact
Gaussian quantiles).
Metrics: standard (the sql is reported, but this probe is about calibration), pit_tv (total-variation distance of the
pooled PIT histogram from uniform; 0 = calibrated), width_vs_oracle = sum of the 10-90 widths / the oracle's (paired).
"""
from __future__ import annotations

import numpy as np

from tideforge.probes import metrics as M
from tideforge.probes.common import Q9, Z9, Cell, ProbeSet, rng_for

PROBE, VERSION, SEED, N_PER_CELL = "P10", 1, 1010, 256
L, H, BURN, N_MC = 1024, 32, 500, 4000
CELLS = ("garch_t", "garch_t_jumps", "control")
# Measured (n = 48 / 256, 2026-10-27): coverage oracle 0.786-0.813, naive 0.974-0.991; pit_tv oracle 0.008-0.035, naive 0.45-0.51.
THRESHOLDS = dict(oracle_cov=(0.75, 0.85), oracle_pit_tv_max=0.05, naive_pit_tv_min=0.3, naive_cov_off_min=0.1)


def _t(rng, nu, size):
    return rng.standard_t(nu, size=size) / np.sqrt(nu / (nu - 2))


def _cell(name: str, n: int, seed: int) -> Cell:
    rng = rng_for(seed, name)
    T = BURN + L + H
    if name == "control":
        y = rng.standard_normal((n, L + H))
        q = np.broadcast_to(Z9[None, :, None], (n, 9, H)).copy()
    else:
        b = rng.uniform(0.80, 0.90, (n, 1))
        a = rng.uniform(0.05, np.minimum(0.15, 0.97 - b))      # alpha + beta <= 0.97: stationary, omega > 0
        w = 1 - a - b
        nu = rng.uniform(3, 5, (n, 1))
        jumps = name == "garch_t_jumps"
        r = np.zeros((n, T))
        s2 = np.ones((n, 1))
        for t in range(T):
            eps = _t(rng, nu, (n, 1))
            r[:, t:t + 1] = np.sqrt(s2) * eps + (jumps * (rng.random((n, 1)) < 0.01) * 5 * rng.standard_normal((n, 1)))
            s2 = w + a * r[:, t:t + 1] ** 2 + b * s2
            if t == BURN + L - 1:
                s2_state = s2.copy()                            # sigma^2 of the first horizon step
        y = r[:, BURN:]
        mc = np.zeros((n, N_MC, H))
        s2m = np.repeat(s2_state, N_MC, 1)
        for h in range(H):
            eps = _t(rng, np.repeat(nu, N_MC, 1), (n, N_MC))
            x = np.sqrt(s2m) * eps
            if jumps:
                x = x + (rng.random((n, N_MC)) < 0.01) * 5 * rng.standard_normal((n, N_MC))
            mc[:, :, h] = x
            s2m = w + a * x ** 2 + b * s2m
        q = np.quantile(mc, Q9, axis=1).transpose(1, 0, 2)    # (n, 9, H)
    a_ = dict(targets=y[:, None, :L].astype(np.float32), future=y[:, None, L:], oracle_q=q[:, None])
    return Cell(name, a_, 1, control=name == "control", desc=name)


def build(n_per_cell: int = N_PER_CELL, seed: int = SEED) -> ProbeSet:
    return ProbeSet(PROBE, VERSION, seed, n_per_cell, [_cell(c, n_per_cell, seed) for c in CELLS],
                    "heavy tails: GARCH-t, GARCH-t with jumps, Gaussian control")


def metrics(cell: Cell, preds: dict) -> list:
    q = preds[("main", "main")]
    y, sc = cell.arrays["future"], M.scale_seasonal(cell.arrays["targets"], 1)
    w = lambda qq: (qq[:, :, 8] - qq[:, :, 0]).reshape(len(qq), -1).sum(1)[:, None]
    return M.standard(q, y, sc) + [M.Metric("pit_tv", M.pit_counts(q, y), "pit_tv"),
                                   M.ratio("width_vs_oracle", w(q), w(cell.arrays["oracle_q"]))]
