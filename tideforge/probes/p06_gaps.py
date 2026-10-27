"""P6 true missing values (gap G6). S = 24, L = 1024, H = 64. Base: seasonal shape + AR(1) noise (phi 0.6, sigma 1).
    random    10% of the context points missing at random
    block     one block of 5-15% of the context, ending in its last 20% (at least 8 steps before the end)
    periodic  hours 0-5 of every day missing (night outages, nightly market closures)
    control   no gaps (both views identical)
Views: main = with gaps (NaN), clean = the same series without gaps. The gaps reach the model as NaN through the scorer's
path; the runner's --nan-variant (baseline, nofill_delete, nofill_mask of tideforge.eval.nan_variants) selects how they are
handled and is part of every result row.
Metrics: standard per view, sql_gaps_vs_clean = sum sql(gaps) / sum sql(clean) on the same items (1 = the gaps cost nothing).
The oracle knows the true state, so its ratio is exactly 1 everywhere.
"""
from __future__ import annotations

import numpy as np

from tideforge.probes import metrics as M
from tideforge.probes.common import Cell, ProbeSet, normal_q, rng_for, seasonal_ar_base

PROBE, VERSION, SEED, N_PER_CELL = "P6", 1, 606, 256
L, H, S = 1024, 64, 24
CELLS = ("random", "block", "periodic", "control")
# Measured (n = 48, 2026-10-27; reported, not tuned): naive / oracle sql on the clean view 1.48, 1.48, 1.42, 1.48 (seasonal naive
# is a fair forecaster for seasonal + AR noise; the same base as P4) -> checked at > 1.2.
THRESHOLDS = dict(oracle_ratio=(0.999, 1.001), naive_vs_oracle_min=1.5,
                  naive_measured_below={"random": 1.2, "block": 1.2, "periodic": 1.2, "control": 1.2})


def _cell(name: str, n: int, seed: int) -> Cell:
    rng = rng_for(seed, name)
    y, bm, bsd = seasonal_ar_base(rng, n, L, H, S, level=10.0)
    miss = np.zeros((n, L), bool)
    if name == "random":
        miss = rng.random((n, L)) < 0.10
    elif name == "block":
        for i in range(n):
            w = int(rng.integers(int(0.05 * L), int(0.15 * L) + 1))
            end = int(rng.integers(int(0.8 * L), L - 8))
            miss[i, max(0, end - w):end] = True
    elif name == "periodic":
        for i in range(n):
            hour = (np.arange(L) + int(rng.integers(0, 24))) % 24
            miss[i] = hour < 6
    ctx = y[:, :L].astype(np.float32)
    a = dict(targets=np.where(miss, np.nan, ctx)[:, None].astype(np.float32), view_clean=ctx[:, None],
             future=y[:, None, L:], oracle_q=normal_q(bm, bsd)[:, None], missing=miss[:, None])
    return Cell(name, a, S, control=name == "control", views=("main", "clean"), desc=f"{name} gaps")


def build(n_per_cell: int = N_PER_CELL, seed: int = SEED) -> ProbeSet:
    return ProbeSet(PROBE, VERSION, seed, n_per_cell, [_cell(c, n_per_cell, seed) for c in CELLS],
                    "gaps: random, block, periodic, control; scored against the same series without gaps")


def metrics(cell: Cell, preds: dict) -> list:
    y = cell.arrays["future"]
    sc = M.scale_seasonal(cell.arrays["view_clean"], cell.S)          # one scale for both views: the clean series'
    out = []
    for view in cell.views:
        q = preds[("main", view)]
        out += M.standard(q, y, sc, view=view) + [M.vs_oracle(q, cell, sc, view=view)]
    out.append(M.ratio("sql_gaps_vs_clean", M.sql(preds[("main", "main")], y, sc), M.sql(preds[("main", "clean")], y, sc)))
    return out
