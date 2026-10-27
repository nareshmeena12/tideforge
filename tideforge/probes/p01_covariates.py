"""P1 covariates (gap G1): groups where a covariate drives the target, one cell per mechanism, at L = 1024 and L = 4096.

Base (all cells): hourly-like, S = 24: seasonal shape + AR(1) noise (phi 0.6, sigma 1). Covariate x: AR(1), phi 0.97,
unit variance, independent of the base (so the target's own history cannot stand in for it). H = 64.
    lagged_linear   y = base + b x(t - lag), lag 0-8, |b| = 2.5; x known (future given)
    nonlinear_u     y = base + a ((temp - 18) / 5)^2, temp = 15 + 6 daily cycle + 3 x; temp known
    promo           y = (10 + base) (1 + u p(t) - d post(t)): promotion windows p (12-36 steps, one starts in the horizon),
                    uplift u 0.5-1.5, a dip d 0.2-0.4 for 12-24 steps after each window; the promotion flag known
    events          y = base + m sum_j 0.7^j e(t - j): sparse event flags e (1%, one in the horizon), |m| 3-6; flags known
    lead_past       y = base + b x(t - lag), lag H..H+16: the driver is past-only and fully informative
    lag0_as_past    y = base + b x(t), x given as past-only: only its persistence helps
    decoy (control) y = base; an unrelated AR covariate given as known
Modes: on (covariate shown, groups ON) / hidden (target only). oracle_q = true predictive quantiles with the covariate;
oracle_hidden_q = a covariate-blind reference (the covariate's effect replaced by its mean, its spread added).
Metrics: standard per mode + sql_on_vs_hidden = sum sql(on) / sum sql(hidden) (paired by item; decoy ~ 1).
"""
from __future__ import annotations

import numpy as np

from tideforge.probes import metrics as M
from tideforge.probes.common import Cell, ProbeSet, ar1, normal_q, periodic, rng_for, seasonal_ar_base

PROBE, VERSION, SEED, N_PER_CELL = "P1", 1, 101, 256
LENGTHS = (1024, 4096)
H, S = 64, 24
MECHS = ("lagged_linear", "nonlinear_u", "promo", "events", "lead_past", "lag0_as_past", "decoy")
# The oracle shows the effect (sql on / hidden below the max) in every non-control cell; the decoy shows none. Naive is at least
# 1.5x the oracle's sql (on) except where measured lower (n = 48 per cell, 2026-10-27; reported, the probe is not tuned):
# lag0_as_past 1.37 (L=1024) / 1.49 (L=4096), decoy 1.49 / 1.48 -> checked at > 1.2 there.
THRESHOLDS = dict(oracle_on_hidden_max={"default": 0.9, "lag0_as_past": 0.95}, decoy_ratio=(0.98, 1.02), naive_vs_oracle_min=1.5,
                  naive_measured_below={"lag0_as_past": 1.2, "decoy": 1.2})


def _cell(mech: str, L: int, n: int, seed: int) -> Cell:
    rng = rng_for(seed, mech, L)
    T = L + H
    base, bm, bsd = seasonal_ar_base(rng, n, L, H, S)
    phi = 0.97
    lagmax = H + 16
    x_all = ar1(rng, n, T + lagmax, phi, np.sqrt(1 - phi ** 2))
    x = x_all[:, lagmax:]                                          # x(t) for t = 0..T-1
    sign = rng.choice([-1.0, 1.0], size=(n, 1))
    role, cov = "known", x
    mean_on, sd_on = bm.copy(), bsd.copy()
    mean_h, var_h = bm.copy(), bsd ** 2
    y = base.copy()
    if mech in ("lagged_linear", "lead_past"):
        lag = rng.integers(0, 9, size=n) if mech == "lagged_linear" else rng.integers(H, H + 17, size=n)
        b = 2.5 * sign
        eff = np.stack([x_all[i, lagmax - lag[i]:lagmax - lag[i] + T] for i in range(n)]) * b
        y = base + eff
        mean_on = bm + eff[:, L:]
        var_h = var_h + b ** 2
        role = "known" if mech == "lagged_linear" else "past"
    elif mech == "lag0_as_past":
        b = 2.5 * sign
        y = base + b * x
        h = np.arange(1, H + 1)
        mean_on = bm + b * phi ** h * x[:, L - 1:L]
        sd_on = np.sqrt(bsd ** 2 + b ** 2 * (1 - phi ** (2 * h)))
        var_h = var_h + b ** 2
        role = "past"
    elif mech == "nonlinear_u":
        temp = 15 + 6 * periodic(rng, n, T, S, harmonics=1) + 3 * x
        a = rng.uniform(1.0, 2.0, size=(n, 1))
        eff = a * ((temp - 18) / 5) ** 2
        y = base + eff
        mean_on = bm + eff[:, L:]
        mean_h = bm + eff[:, :L].mean(1, keepdims=True)
        var_h = var_h + eff[:, :L].var(1, keepdims=True)
        cov = temp
    elif mech == "promo":
        p = np.zeros((n, T))
        post = np.zeros((n, T))
        for i in range(n):
            starts, t = [], int(rng.integers(0, 150))
            while t < L - 60:
                starts.append(t)
                t += int(rng.integers(150, 300))
            starts.append(L + int(rng.integers(0, H - 12)))
            k = int(rng.integers(12, 25))
            for s in starts:
                w = int(rng.integers(12, 37))
                p[i, s:s + w] = 1
                post[i, s + w:s + w + k] = 1
        post = post * (1 - p)
        u, d = rng.uniform(0.5, 1.5, size=(n, 1)), rng.uniform(0.2, 0.4, size=(n, 1))
        c = 1 + u * p - d * post
        y = (10 + base) * c
        q_on = normal_q(c[:, L:] * (10 + bm), c[:, L:] * bsd)
        q_h = normal_q(10 + bm, bsd)
        cov = p
        return _finish(mech, L, y, cov, role, q_on, q_h, S)
    elif mech == "events":
        e = (rng.random((n, T)) < 0.01).astype(float)
        e[np.arange(n), L + rng.integers(0, H - 8, size=n)] = 1.0
        kern = 0.7 ** np.arange(30)
        pulse = np.stack([np.convolve(r, kern)[:T] for r in e])
        m = rng.uniform(3, 6, size=(n, 1)) * sign
        eff = m * pulse
        y = base + eff
        mean_on = bm + eff[:, L:]
        mean_h = bm + eff[:, :L].mean(1, keepdims=True)
        var_h = var_h + eff[:, :L].var(1, keepdims=True)
        cov = e
    else:                                                            # decoy: an unrelated covariate
        cov = ar1(rng, n, T, phi, np.sqrt(1 - phi ** 2))
        y = base
        return _finish(mech, L, y, cov, role, normal_q(bm, bsd), normal_q(bm, bsd), S, control=True)
    return _finish(mech, L, y, cov, role, normal_q(mean_on, sd_on), normal_q(mean_h, np.sqrt(var_h)), S)


def _finish(mech, L, y, cov, role, q_on, q_h, S_, control=False) -> Cell:
    a = dict(targets=y[:, None, :L].astype(np.float32), future=y[:, None, L:], oracle_q=q_on[:, None],
             oracle_hidden_q=q_h[:, None])
    if role == "past":
        a["past"] = cov[:, None, :L].astype(np.float32)
    else:
        a["known_ctx"] = cov[:, None, :L].astype(np.float32)
        a["known_fut"] = cov[:, None, L:].astype(np.float32)
    return Cell(f"{mech}@{L}", a, S_, control=control, modes=("on", "hidden"), desc=f"{mech}, covariate {role}, L={L}",
                extra=dict(role=role, L=L))


def build(n_per_cell: int = N_PER_CELL, seed: int = SEED) -> ProbeSet:
    cells = [_cell(m, L, n_per_cell, seed) for L in LENGTHS for m in MECHS]
    return ProbeSet(PROBE, VERSION, seed, n_per_cell, cells, "covariates: one cell per mechanism, two context lengths")


def metrics(cell: Cell, preds: dict) -> list:
    y, sc = cell.arrays["future"], M.scale_seasonal(cell.arrays["targets"], cell.S)
    out = []
    for mode in cell.modes:
        out += M.standard(preds[(mode, "main")], y, sc, mode)
    on, hid = (M.sql(preds[(m, "main")], y, sc) for m in ("on", "hidden"))
    out.append(M.ratio("sql_on_vs_hidden", on, hid, "on"))
    return out
