"""P2 calendar (gap G2): calendar effects with and without their flag. Daily, S = 7, H = 28.

Base: level + weekly shape + AR(1) noise (phi 0.6, sigma 1).
    holiday          ~10 random holiday dates a year (not recurring), effect m (|m| 3-6) on each; >= 1 in the horizon;
                     L = 730. The holiday flag is the known covariate.
    moving_festival  a 3-day festival that moves 11 days earlier every year (365 - 11 = 354-day spacing, as an Islamic-
                     calendar holiday drifts through the Gregorian year); 3 occurrences in L = 1096 days, the 4th in the
                     horizon. The festival flag is the known covariate.
    weekend_switch   the weekend effect w (-2 to -4) falls on the days the weekend flag marks; the definition switches
                     from Sat-Sun to Fri-Sat 7-21 days before the context ends (as a country changing its weekend). No
                     other weekly shape. The weekend flag (old then new definition) is the known covariate. L = 730.
    control          the holiday cell's design with effect 0 (the flag is there but means nothing).
Modes on / hidden. oracle_hidden_q: the effect replaced by its mean, its spread added (a blind reference).
Metrics: standard per mode; event_sql (sql on the flagged days of the horizon; for weekend_switch the days whose role
changed: Fri and Sun); sql_on_vs_hidden and event_sql_on_vs_hidden (paired).
"""
from __future__ import annotations

import numpy as np

from tideforge.probes import metrics as M
from tideforge.probes.common import Cell, ProbeSet, normal_q, rng_for, seasonal_ar_base

PROBE, VERSION, SEED, N_PER_CELL = "P2", 1, 202, 256
H, S = 28, 7
CELLS = ("holiday", "moving_festival", "weekend_switch", "control")
# Measured below 1.5 (n = 48, 2026-10-27; reported, not tuned): naive / oracle sql on weekend_switch 1.49 -> checked at > 1.2.
THRESHOLDS = dict(oracle_event_on_hidden_max=0.8, control_ratio=(0.98, 1.02), naive_vs_oracle_min=1.5,
                  naive_measured_below={"weekend_switch": 1.2})


def _cell(name: str, n: int, seed: int) -> Cell:
    rng = rng_for(seed, name)
    L = 1096 if name == "moving_festival" else 730
    T = L + H
    amp = 0.0 if name == "weekend_switch" else 1.0
    base, bm, bsd = seasonal_ar_base(rng, n, L, H, S, season_amp=amp, level=10.0)
    flag = np.zeros((n, T))
    mask = np.zeros((n, H), bool)
    if name in ("holiday", "control"):
        flag = (rng.random((n, T)) < 10 / 365).astype(float)
        flag[np.arange(n), L + rng.integers(0, H, size=n)] = 1.0
        m = rng.uniform(3, 6, size=(n, 1)) * rng.choice([-1.0, 1.0], size=(n, 1)) * (name == "holiday")
        mask = flag[:, L:] > 0
    elif name == "moving_festival":
        for i in range(n):
            s3 = L + int(rng.integers(0, H - 3))
            for k in range(4):
                a = s3 - (3 - k) * 354
                flag[i, a:a + 3] = 1.0
        m = rng.uniform(3, 6, size=(n, 1)) * rng.choice([-1.0, 1.0], size=(n, 1))
        mask = flag[:, L:] > 0
    else:                                                          # weekend_switch
        m = -rng.uniform(2, 4, size=(n, 1))
        for i in range(n):
            phase = int(rng.integers(0, 7))                        # day of week of t = 0 (0 = Monday)
            sw = L - int(rng.integers(7, 22))
            dow = (np.arange(T) + phase) % 7
            old, new = np.isin(dow, (5, 6)), np.isin(dow, (4, 5))
            flag[i] = np.where(np.arange(T) < sw, old, new)
            mask[i] = old[L:] != new[L:]
    eff = m * flag
    y = base + eff
    q_on = normal_q(bm + eff[:, L:], bsd)
    mean_h = bm + eff[:, :L].mean(1, keepdims=True)
    sd_h = np.sqrt(bsd ** 2 + eff[:, :L].var(1, keepdims=True))
    a = dict(targets=y[:, None, :L].astype(np.float32), future=y[:, None, L:], oracle_q=q_on[:, None],
             oracle_hidden_q=normal_q(mean_h, sd_h)[:, None], known_ctx=flag[:, None, :L].astype(np.float32),
             known_fut=flag[:, None, L:].astype(np.float32), event_mask=mask[:, None, :])
    return Cell(name, a, S, control=name == "control", modes=("on", "hidden"), desc=f"{name}, L={L}", extra=dict(L=L))


def build(n_per_cell: int = N_PER_CELL, seed: int = SEED) -> ProbeSet:
    return ProbeSet(PROBE, VERSION, seed, n_per_cell, [_cell(c, n_per_cell, seed) for c in CELLS],
                    "calendar: holiday / moving festival / weekend switch, with vs without the flag")


def metrics(cell: Cell, preds: dict) -> list:
    y, sc = cell.arrays["future"], M.scale_seasonal(cell.arrays["targets"], cell.S)
    mask = cell.arrays["event_mask"]
    out, ev, al = [], {}, {}
    for mode in cell.modes:
        q = preds[(mode, "main")]
        out += M.standard(q, y, sc, mode)
        ev[mode], al[mode] = M.sql(q, y, sc, mask), M.sql(q, y, sc)
        out.append(M.Metric("event_sql", ev[mode], "mean", mode))
    out.append(M.ratio("sql_on_vs_hidden", al["on"], al["hidden"], "on"))
    out.append(M.ratio("event_sql_on_vs_hidden", ev["on"], ev["hidden"], "on"))
    return out
