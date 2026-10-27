"""P3 few cycles and seen-once history (gap G3).

year_once / year_twice: tidemark's year probe (third_party/tidemark_e21cbf0/tidemark/pretrain/probes.py L139-L181,
_year_set, YEAR_SETS, YEAR_CYCLES), ported here: per long period P in (52, 120, 365) one dated event per cycle whose next
occurrence falls in the horizon, plus two other dated events that never touch it; 1.4 cycles of history (event seen once
before) or 2.4 (seen twice). One cell per (P, cycles). Change vs the source: observation noise 0.1 x the series scale is added
to the context AND the future (the source is noise-free in the future), so the oracle has a proper predictive distribution;
the event score compares with the same noisy future minus the event.
long_period: a smooth periodic shape with period 0.3-0.45 x L (2.2-3.3 cycles in the context) + AR(1) noise; L = 2048, H = 96.
control: the same shapes with periods 20-60 (>= 34 cycles).

Metrics: standard; sql_vs_oracle; event_score (year cells: per series 1 - sum over the event steps (median - future)^2 /
sum (future without the event - future)^2, clipped to [-1, 1], as tidemark's year probe; 1 = the event forecast exactly,
0 = as if it were not there).
"""
from __future__ import annotations

import numpy as np

from tideforge.probes import metrics as M
from tideforge.probes.common import Cell, ProbeSet, Z9, ar1, ar1_forecast, normal_q, periodic, rng_for

PROBE, VERSION, SEED, N_PER_CELL = "P3", 1, 303, 256
YEAR_SETS = [(52, 0), (120, 7), (365, 7)]      # (long period, short period inside it; 0 = none)   [tidemark L139]
YEAR_CYCLES = [1.4, 2.4]                        # cycles of history the model is shown               [tidemark L140]
NOISE = 0.1
# Measured (n = 48, 2026-10-27; reported, not tuned): the seasonal naive model is given the true period, and the year cells repeat
# exactly, so naive brings the event back (event score 0.83-1.0) and its sql vs oracle is 1.48 (year_once_P52), 1.38
# (year_twice_P52), 9.2-9.7 at P = 120 / 365; long_period 1.42, control 1.58. Checked at > 1.2 where measured below 1.5.
THRESHOLDS = dict(oracle_event_min=0.9, naive_vs_oracle_min=1.5,
                  naive_measured_below={"year_once_P52": 1.2, "year_twice_P52": 1.2, "long_period": 1.2})


def _year_cell(P: int, ps: int, cyc: float, n: int, seed: int) -> Cell:
    """tidemark _year_set for one (P, cycles), plus observation noise on context and future."""
    rng = rng_for(seed, "year", P, cyc)
    w = max(1, P // 50)
    H = max(8, min(64, P // 6))
    n_ctx = int(round(cyc * P))
    t = np.arange(n_ctx + H)
    ctx, fut, base, mask, clean = [], [], [], [], []
    sd_obs = []
    for _ in range(n):
        y = 0.6 * np.sin(2 * np.pi * (t + int(rng.integers(0, P))) / P)
        if ps:
            y = y + 0.7 * rng.standard_normal(ps)[(t + int(rng.integers(0, ps))) % ps]
        te = n_ctx + int(rng.integers(1, H - w))
        ev, oth = np.zeros(len(t)), np.zeros(len(t))
        m = float(rng.choice([-1.0, 1.0])) * rng.uniform(2.0, 4.0)
        for a in range(te % P - P, len(t), P):
            ev[max(a, 0):max(a + w, 0)] += m
        for _k in range(2):
            off = int(rng.integers(H + w, P - H - w))
            m2 = float(rng.choice([-1.0, 1.0])) * rng.uniform(1.5, 3.0)
            for a in range((te - off) % P - P, len(t), P):
                oth[max(a, 0):max(a + w, 0)] += m2
        full, nob = y + oth + ev, y + oth
        mu_, sd_ = full[:n_ctx].mean(), full[:n_ctx].std() + 1e-9
        sc_, sh_ = rng.uniform(0.5, 5.0), rng.normal(0.0, 3.0)
        f2, b2 = sc_ * (full - mu_) / sd_ + sh_, sc_ * (nob - mu_) / sd_ + sh_
        eps = NOISE * sc_ * rng.standard_normal(len(t))
        ctx.append(f2[:n_ctx] + eps[:n_ctx])
        fut.append(f2[n_ctx:] + eps[n_ctx:])
        base.append(b2[n_ctx:] + eps[n_ctx:])
        clean.append(f2[n_ctx:])
        sd_obs.append(NOISE * sc_)
        mk = np.zeros(H, bool)
        mk[te - n_ctx:te - n_ctx + w] = True
        mask.append(mk)
    sd = np.asarray(sd_obs)[:, None] * np.ones((1, H))
    a = dict(targets=np.asarray(ctx, np.float32)[:, None], future=np.asarray(fut)[:, None],
             oracle_q=normal_q(np.asarray(clean), sd)[:, None], without_event=np.asarray(base)[:, None],
             event_mask=np.asarray(mask)[:, None])
    name = f"year_{'once' if cyc < 2 else 'twice'}_P{P}"
    return Cell(name, a, P, desc=f"tidemark year probe P={P} (inner {ps}), {cyc} cycles of history",
                extra=dict(P=P, cycles=cyc, H=H))


def _long_cell(name: str, n: int, seed: int) -> Cell:
    rng = rng_for(seed, name)
    L, H = 2048, 96
    P = rng.uniform(0.3, 0.45, n) * L if name == "long_period" else rng.uniform(20, 60, n)
    shape = 2.0 * periodic(rng, n, L + H, P)
    e = ar1(rng, n, L + H, 0.6, 0.5)
    y = 5.0 + shape + e
    m, sd = ar1_forecast(e[:, L - 1], 0.6, 0.5, H)
    a = dict(targets=y[:, None, :L].astype(np.float32), future=y[:, None, L:],
             oracle_q=normal_q(5.0 + shape[:, L:] + m, sd)[:, None], season=np.round(P).astype(np.int64))
    return Cell(name, a, int(np.median(np.round(P))), control=name == "control",
                desc="period 0.3-0.45 L" if name == "long_period" else "period 20-60 (many cycles)")


def build(n_per_cell: int = N_PER_CELL, seed: int = SEED) -> ProbeSet:
    cells = [_year_cell(P, ps, cyc, n_per_cell, seed) for P, ps in YEAR_SETS for cyc in YEAR_CYCLES]
    cells += [_long_cell(c, n_per_cell, seed) for c in ("long_period", "control")]
    return ProbeSet(PROBE, VERSION, seed, n_per_cell, cells, "few cycles: year once / twice, long period")


def event_score(q, cell) -> np.ndarray:
    med, y = M.median(q)[:, 0], cell.arrays["future"][:, 0]
    b, mk = cell.arrays["without_event"][:, 0], cell.arrays["event_mask"][:, 0]
    num = (((med - y) ** 2) * mk).sum(1)
    den = np.maximum((((b - y) ** 2) * mk).sum(1), 1e-12)
    return np.clip(1.0 - num / den, -1.0, 1.0)[:, None]


def metrics(cell: Cell, preds: dict) -> list:
    q = preds[("main", "main")]
    y, sc = cell.arrays["future"], M.scale_seasonal(cell.arrays["targets"], cell.seasons())
    out = M.standard(q, y, sc) + [M.vs_oracle(q, cell, sc)]
    if "event_mask" in cell.arrays:
        out.append(M.Metric("event_score", event_score(q, cell), "mean"))
    return out
