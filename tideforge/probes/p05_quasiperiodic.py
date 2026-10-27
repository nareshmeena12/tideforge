"""P5 drifting and quasi-periodic cycles (gap G5), at L = 1024 and L = 4096; H = 64. Noise: AR(1) (phi 0.6, sigma 0.3).
    am       amplitude modulation: 2 (1 + 0.5 sin(2 pi t / T_A + phi)) x shape(period P), P 20-40, T_A 300-800
    fm       frequency modulation: the period wanders +-3% (f(t) = (1 + 0.03 sin(2 pi t / T_f + phi)) / P, T_f 400-1200),
             waveform sin(theta) + 0.4 sin(2 theta)
    beats    two close periods, 12.42 and 12 (tides: spring-neap), amplitudes 1 and 0.5-1
    control  a stationary periodic shape, P 20-40, same noise
The modulations are deterministic, so the oracle knows the future up to the AR noise.
Metrics: standard, sql_vs_oracle.
"""
from __future__ import annotations

import numpy as np

from tideforge.probes import metrics as M
from tideforge.probes.common import Cell, ProbeSet, ar1, ar1_forecast, normal_q, periodic, rng_for

PROBE, VERSION, SEED, N_PER_CELL = "P5", 1, 505, 256
LENGTHS = (1024, 4096)
H = 64
CELLS = ("am", "fm", "beats", "control")
# Measured (v1, n = 48, 2026-10-27): naive / oracle sql 1.66-3.10 (am 2.13-2.18, fm 1.93-1.98, beats 3.04-3.10, control 1.66-1.79).
THRESHOLDS = dict(naive_vs_oracle_min=1.5)


def _cell(name: str, L: int, n: int, seed: int) -> Cell:
    rng = rng_for(seed, name, L)
    T = L + H
    t = np.arange(T)[None, :]
    P = rng.uniform(20, 40, size=(n, 1))
    ph = rng.uniform(0, 2 * np.pi, size=(n, 1))
    if name == "am":
        TA = rng.uniform(300, 800, size=(n, 1))
        det = 2 * (1 + 0.5 * np.sin(2 * np.pi * t / TA + ph)) * periodic(rng, n, T, P[:, 0])
    elif name == "fm":
        Tf = rng.uniform(400, 1200, size=(n, 1))
        f = (1 + 0.03 * np.sin(2 * np.pi * t / Tf + ph)) / P
        theta = 2 * np.pi * np.cumsum(f, axis=1) + rng.uniform(0, 2 * np.pi, size=(n, 1))
        det = 2 * (np.sin(theta) + 0.4 * np.sin(2 * theta))
    elif name == "beats":
        a = rng.uniform(0.5, 1.0, size=(n, 1))
        det = 2 * (np.sin(2 * np.pi * t / 12.42 + ph) + a * np.sin(2 * np.pi * t / 12 + rng.uniform(0, 2 * np.pi, size=(n, 1))))
        P = np.full((n, 1), 12.0)
    else:
        det = 2 * periodic(rng, n, T, P[:, 0])
    e = ar1(rng, n, T, 0.6, 0.3)
    y = 5 + det + e
    m, sd = ar1_forecast(e[:, L - 1], 0.6, 0.3, H)
    a = dict(targets=y[:, None, :L].astype(np.float32), future=y[:, None, L:], oracle_q=normal_q(5 + det[:, L:] + m, sd)[:, None],
             season=np.round(P[:, 0]).astype(np.int64))
    return Cell(f"{name}@{L}", a, int(np.median(np.round(P))), control=name == "control", desc=f"{name}, L={L}", extra=dict(L=L))


def build(n_per_cell: int = N_PER_CELL, seed: int = SEED) -> ProbeSet:
    return ProbeSet(PROBE, VERSION, seed, n_per_cell, [_cell(c, L, n_per_cell, seed) for L in LENGTHS for c in CELLS],
                    "quasi-periodic: AM, FM, beats, control; two context lengths")


def metrics(cell: Cell, preds: dict) -> list:
    q = preds[("main", "main")]
    y, sc = cell.arrays["future"], M.scale_seasonal(cell.arrays["targets"], cell.seasons())
    return M.standard(q, y, sc) + [M.vs_oracle(q, cell, sc)]
