"""P8 hierarchies (gap G8). Daily, S = 7, L = 512, H = 28. Targets per item: [parent, child 1, ..., child c]; every child =
level + weekly shape + AR(1) noise, independent of the others; parent = the exact sum of the children.
    similar   4 children of comparable size (levels 5-15, noise sigma 1)
    dominant  4 children, one carrying most of the total (level 40, sigma 3; the others 2-4, sigma 0.5)
    control   1 child (the parent equals it)
oracle_q: Gaussian, exact (the parent's mean and variance are the sums of the children's), so its medians are coherent.
Modes: on = all targets forecast together as one group; hidden = each target on its own.
Metrics per mode: standard over all targets; sql_parent, sql_children; coherence = mean over the horizon of
|sum of the children's medians - the parent's median| / the parent's scale (0 = coherent).
"""
from __future__ import annotations

import numpy as np

from tideforge.probes import metrics as M
from tideforge.probes.common import Cell, ProbeSet, normal_q, rng_for, seasonal_ar_base

PROBE, VERSION, SEED, N_PER_CELL = "P8", 1, 808, 256
L, H, S = 512, 28, 7
CELLS = ("similar", "dominant", "control")
# Measured (n = 48, 2026-10-27; reported, not tuned): naive / oracle sql similar 1.52, dominant 1.52, control 1.46 -> > 1.2 there;
# coherence oracle 0, naive 0.05-0.07 (its residual-quantile medians do not add up).
THRESHOLDS = dict(oracle_coherence_max=1e-6, naive_vs_oracle_min=1.5, naive_measured_below={"control": 1.2})


def _cell(name: str, n: int, seed: int) -> Cell:
    rng = rng_for(seed, name)
    c = 1 if name == "control" else 4
    ys, ms, vs = [], [], []
    for j in range(c):
        if name == "similar":
            lev, sig = rng.uniform(5, 15, size=(n, 1)), 1.0
        elif name == "dominant":
            lev, sig = (np.full((n, 1), 40.0), 3.0) if j == 0 else (rng.uniform(2, 4, size=(n, 1)), 0.5)
        else:
            lev, sig = rng.uniform(5, 15, size=(n, 1)), 1.0
        y, m, sd = seasonal_ar_base(rng, n, L, H, S, season_amp=sig, sigma=sig, level=lev)
        ys.append(y)
        ms.append(m)
        vs.append(sd ** 2)
    parent, pm, pv = sum(ys), sum(ms), sum(vs)
    Y = np.stack([parent] + ys, 1)                                      # (n, 1 + c, L + H)
    q = np.stack([normal_q(pm, np.sqrt(pv))] + [normal_q(m, np.sqrt(v)) for m, v in zip(ms, vs)], 1)
    a = dict(targets=Y[..., :L].astype(np.float32), future=Y[..., L:], oracle_q=q)
    return Cell(name, a, S, control=name == "control", modes=("on", "hidden"), desc=f"parent + {c} children")


def build(n_per_cell: int = N_PER_CELL, seed: int = SEED) -> ProbeSet:
    return ProbeSet(PROBE, VERSION, seed, n_per_cell, [_cell(c, n_per_cell, seed) for c in CELLS],
                    "hierarchy: parent = sum of children; similar, dominant, one-child control")


def coherence(q, scale_parent) -> np.ndarray:
    med = M.median(q)                                                   # (n, k, H)
    return (np.abs(med[:, 1:].sum(1) - med[:, 0]).mean(1) / scale_parent)[:, None]


def metrics(cell: Cell, preds: dict) -> list:
    y, sc = cell.arrays["future"], M.scale_seasonal(cell.arrays["targets"], cell.S)
    out = []
    for mode in cell.modes:
        q = preds[(mode, "main")]
        out += M.standard(q, y, sc, mode) + [M.vs_oracle(q, cell, sc, mode)]
        out.append(M.Metric("sql_parent", M.sql(q[:, :1], y[:, :1], sc[:, :1]), "mean", mode))
        out.append(M.Metric("sql_children", M.sql(q[:, 1:], y[:, 1:], sc[:, 1:]), "mean", mode))
        out.append(M.Metric("coherence", coherence(q, sc[:, 0]), "mean", mode))
    return out
