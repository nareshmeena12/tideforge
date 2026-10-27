"""P4 breaks (gap G4). S = 24, L = 1024, H = 64. Base: seasonal shape + AR(1) noise (phi 0.6, sigma 1).
    step_up / step_down  a level shift of size 4-7 (about 3-6 noise sd) at 70-90% of the context, lasting to the end
    transient            a shift of the same size starting at 60-80% of the context, lasting 5-20% of it, back to the old
                         level >= 16 steps before the context ends (correct adoption = 0)
    control              no break
Metrics: standard, sql_vs_oracle, adoption = mean over the horizon of (median - level without the shift) / shift
(1 = takes the new level, 0 = stays at the old one; step cells should be 1, transient 0; a model that adopts every jump
looks good on step and bad on transient), and width80_ratio_vs_control (cross-cell, unpaired bootstrap).
"""
from __future__ import annotations

import numpy as np

from tideforge.probes import metrics as M
from tideforge.probes.common import Cell, ProbeSet, normal_q, rng_for, seasonal_ar_base

PROBE, VERSION, SEED, N_PER_CELL = "P4", 1, 404, 256
L, H, S = 1024, 64, 24
CELLS = ("step_up", "step_down", "transient", "control")
THRESHOLDS = dict(oracle_adoption={"step_up": (0.98, 1.02), "step_down": (0.98, 1.02), "transient": (-0.02, 0.02)},
                  naive_vs_oracle_min=1.5, naive_measured_below={"step_down": 1.2, "transient": 1.2, "control": 1.2})
# Measured (n = 48, 2026-10-27; reported, not tuned): naive / oracle sql step_up 1.54, step_down 1.47, transient 1.48, control
# 1.46. Seasonal naive repeats the last season, so it also adopts correctly (1.01 / 1.02 on steps, -0.01 on transient):
# adoption separates models that react to old jumps, not this baseline.


def _cell(name: str, n: int, seed: int) -> Cell:
    rng = rng_for(seed, name)
    base, bm, bsd = seasonal_ar_base(rng, n, L, H, S, level=10.0)
    size = rng.uniform(4, 7, size=n) * (-1 if name == "step_down" else rng.choice([-1.0, 1.0], size=n) if name == "transient" else 1)
    shift = np.zeros((n, L + H))
    for i in range(n):
        if name.startswith("step"):
            shift[i, int(rng.integers(int(0.7 * L), int(0.9 * L))):] = size[i]
        elif name == "transient":
            d = int(rng.integers(int(0.05 * L), int(0.2 * L)))
            b = min(int(rng.integers(int(0.6 * L), int(0.8 * L))), L - 16 - d)
            shift[i, b:b + d] = size[i]
    y = base + shift
    a = dict(targets=y[:, None, :L].astype(np.float32), future=y[:, None, L:],
             oracle_q=normal_q(bm + shift[:, L:], bsd)[:, None], level_without=bm[:, None],
             shift=(size if name != "control" else np.zeros(n))[:, None])
    return Cell(name, a, S, control=name == "control", desc=name)


def build(n_per_cell: int = N_PER_CELL, seed: int = SEED) -> ProbeSet:
    return ProbeSet(PROBE, VERSION, seed, n_per_cell, [_cell(c, n_per_cell, seed) for c in CELLS],
                    "breaks: step up / down, transient, control")


def adoption(q, cell) -> np.ndarray:
    med = M.median(q)[:, 0]
    d = cell.arrays["shift"][:, 0]
    return ((med - cell.arrays["level_without"][:, 0]).mean(1) / np.where(d != 0, d, np.nan))[:, None]


def metrics(cell: Cell, preds: dict) -> list:
    q = preds[("main", "main")]
    y, sc = cell.arrays["future"], M.scale_seasonal(cell.arrays["targets"], cell.S)
    out = M.standard(q, y, sc) + [M.vs_oracle(q, cell, sc)]
    if not cell.control:
        out.append(M.Metric("adoption", adoption(q, cell), "mean"))
    return out


def cross(cells, per_cell):
    ctrl = next(m for m in per_cell["control"] if m.name == "width80")
    return [(c, next(m for m in per_cell[c.name] if m.name == "width80"), ctrl) for c in cells if not c.control]
