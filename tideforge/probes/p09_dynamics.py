"""P9 nonlinear dynamics (gap G9). L = 1024, H = 64. Trajectories from fixed-step RK4 (vectorised over series, 10 sub-steps
per observation, a burn-in so each series sits on its attractor), observed with Gaussian noise of 2% of the series' sd.
    lotka_volterra  prey of dx = a x - b x y, dy = d x y - g y (a, g 0.8-1.2; b 0.3-0.6; d 0.2-0.4); dt 0.15
    forced_sir      I(t) x 1e4 of a seasonally forced SIR with births (beta = beta0 (1 + 0.2 cos(2 pi t / 52)), beta0 8-12,
                    gamma 1, mu 1/260; weekly steps), recurrent epidemics
    van_der_pol     x of x'' - mu (1 - x^2) x' + x = 0, mu 0.5-3; dt 0.15
    control         a sine with period 40-60 and the same relative noise
The oracle knows the state: oracle_q = the true future + the observation noise quantiles. The naive season of each series is
the dominant period of its context (periodogram peak).
Metrics: standard, sql_vs_oracle.
"""
from __future__ import annotations

import numpy as np

from tideforge.probes import metrics as M
from tideforge.probes.common import Cell, ProbeSet, normal_q, rng_for

PROBE, VERSION, SEED, N_PER_CELL = "P9", 1, 909, 256
L, H = 1024, 64
CELLS = ("lotka_volterra", "forced_sir", "van_der_pol", "control")
NOISE = 0.02
THRESHOLDS = dict(naive_vs_oracle_min=1.5)


def rk4(f, z0: np.ndarray, t0: float, dt: float, n_obs: int, sub: int = 10) -> np.ndarray:
    """z (n, d) observed every dt for n_obs steps (sub-steps of dt / sub)."""
    h = dt / sub
    z, t = z0.copy(), t0
    out = np.empty((n_obs,) + z.shape)
    for i in range(n_obs):
        for _ in range(sub):
            k1 = f(t, z)
            k2 = f(t + h / 2, z + h / 2 * k1)
            k3 = f(t + h / 2, z + h / 2 * k2)
            k4 = f(t + h, z + h * k3)
            z = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            t += h
        out[i] = z
    return out.transpose(1, 0, 2)                                   # (n, n_obs, d)


def trajectories(name: str, rng, n: int, T: int) -> np.ndarray:
    if name == "lotka_volterra":
        a, g = rng.uniform(0.8, 1.2, (n, 1)), rng.uniform(0.8, 1.2, (n, 1))
        b, d = rng.uniform(0.3, 0.6, (n, 1)), rng.uniform(0.2, 0.4, (n, 1))
        f = lambda t, z: np.concatenate([a * z[:, :1] - b * z[:, :1] * z[:, 1:], d * z[:, :1] * z[:, 1:] - g * z[:, 1:]], 1)
        z0 = np.concatenate([rng.uniform(1, 3, (n, 1)), rng.uniform(0.5, 2, (n, 1))], 1)
        return rk4(f, z0, 0.0, 0.15, T + 200)[:, 200:, 0]
    if name == "van_der_pol":
        mu = rng.uniform(0.5, 3.0, (n, 1))
        f = lambda t, z: np.concatenate([z[:, 1:], mu * (1 - z[:, :1] ** 2) * z[:, 1:] - z[:, :1]], 1)
        z0 = rng.uniform(-2, 2, (n, 2))
        return rk4(f, z0, 0.0, 0.15, T + 200)[:, 200:, 0]
    if name == "forced_sir":
        b0, gam, mu = rng.uniform(8, 12, (n, 1)), 1.0, 1 / 260
        ph = rng.uniform(0, 2 * np.pi, (n, 1))

        def f(t, z):
            beta = b0 * (1 + 0.2 * np.cos(2 * np.pi * t / 52 + ph))
            s, i = z[:, :1], z[:, 1:]
            return np.concatenate([mu - beta * s * i - mu * s, beta * s * i - (gam + mu) * i], 1)
        z0 = np.concatenate([rng.uniform(0.08, 0.12, (n, 1)), rng.uniform(1e-4, 1e-3, (n, 1))], 1)
        return rk4(f, z0, 0.0, 1.0, T + 2000, sub=20)[:, 2000:, 1] * 1e4
    P = rng.uniform(40, 60, (n, 1))
    return np.sin(2 * np.pi * np.arange(T)[None, :] / P + rng.uniform(0, 2 * np.pi, (n, 1)))


def dominant_period(x: np.ndarray) -> np.ndarray:
    xc = x - x.mean(1, keepdims=True)
    p = np.abs(np.fft.rfft(xc, axis=1)) ** 2
    p[:, :2] = 0
    k = np.argmax(p, axis=1)
    return np.clip(np.round(x.shape[1] / np.maximum(k, 1)), 2, x.shape[1] // 2).astype(np.int64)


def _cell(name: str, n: int, seed: int) -> Cell:
    rng = rng_for(seed, name)
    z = trajectories(name, rng, n, L + H)
    sd = NOISE * z[:, :L].std(1, keepdims=True)
    y = z + sd * rng.standard_normal(z.shape)
    a = dict(targets=y[:, None, :L].astype(np.float32), future=y[:, None, L:],
             oracle_q=normal_q(z[:, L:], np.broadcast_to(sd, (n, H)))[:, None], season=dominant_period(y[:, :L]))
    return Cell(name, a, int(np.median(a["season"])), control=name == "control", desc=name)


def build(n_per_cell: int = N_PER_CELL, seed: int = SEED) -> ProbeSet:
    return ProbeSet(PROBE, VERSION, seed, n_per_cell, [_cell(c, n_per_cell, seed) for c in CELLS],
                    "dynamics: Lotka-Volterra, forced SIR, van der Pol, sine control")


def metrics(cell: Cell, preds: dict) -> list:
    q = preds[("main", "main")]
    y, sc = cell.arrays["future"], M.scale_seasonal(cell.arrays["targets"], cell.seasons())
    return M.standard(q, y, sc) + [M.vs_oracle(q, cell, sc)]
