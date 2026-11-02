"""P11 over-trust (docs/notes/g1_redesign_20261101.md, owner OK 2026-11-03): covariates a model should discount. P1 v1 is unchanged.

L = 1024, H = 64, P1's base (hourly-like, S = 24, AR(1) noise); covariate x: AR(1), phi 0.97, unit variance. b = 2.5 with a random sign.
    weak            y = base + 0.1 b x; known                     on about equal to hidden
    noisy_known     y = base + b x; the known future is a poor forecast (persistence plus noise)     discount the future values
    noisy_obs       y = base + b x; x seen with noise at SNR 1    partial use
    broke           the effect exists in the first 70% of the context, then stops                    stop using it
    sign_flip       the sign reverses at 70-90% of the context    follow the recent sign
    spurious        unrelated; the draw of 16 most correlated with the target's context              ignore it
    seasonal_decoy  the target's seasonal shape plus noise, no extra information                     ignore it
    control         y = base + b x; known                         use it
Modes on / hidden. Main metric: harm = sum sql(on) / sum sql(hidden); above 1, showing the covariate made the forecast worse. Oracles
are Gaussian; noisy_obs treats each point's noise alone (it ignores the smoothing a model could do), so it slightly understates what is
possible. Cell.extra["strength"] is the true effect in the horizon, in units of the base's noise sd (b x / 1), for trust vs strength.
Summary for the board: harm_summary, the mean of max(0, harm - 1) over the non-control cells, read beside control's harm.
"""
from __future__ import annotations

import numpy as np

from tideforge.probes import metrics as M
from tideforge.probes.common import Cell, ProbeSet, ar1, normal_q, rng_for, seasonal_ar_base

PROBE, VERSION, SEED, N_PER_CELL = "P11", 1, 1111, 256
L, H, S = 1024, 64, 24
PHI, B, K = 0.97, 2.5, 16
CELLS = ("weak", "noisy_known", "noisy_obs", "broke", "sign_flip", "spurious", "seasonal_decoy", "control")
STRENGTH = dict(weak=0.25, noisy_known=2.5, noisy_obs=2.5, broke=0.0, sign_flip=2.5, spurious=0.0, seasonal_decoy=0.0, control=2.5)
THRESHOLDS = dict(oracle_harm_max=1.01, oracle_control_max=0.9)


def _corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a - a.mean(-1, keepdims=True), b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a ** 2).sum(-1) * (b ** 2).sum(-1) + 1e-12)


def _cell(name: str, n: int, seed: int) -> Cell:
    rng = rng_for(seed, name)
    T = L + H
    base, bm, bsd = seasonal_ar_base(rng, n, L, H, S)
    sig = np.sqrt(1 - PHI ** 2)
    x = ar1(rng, n, T, PHI, sig)
    b = B * rng.choice([-1.0, 1.0], size=(n, 1))
    role, cov, cov_fut = "known", x[:, :L], x[:, L:]
    h = np.arange(1, H + 1)
    y = base + b * x
    mean_on, sd_on = bm + b * x[:, L:], bsd
    mean_h, sd_h = bm, np.sqrt(bsd ** 2 + b ** 2)
    if name == "weak":
        b = 0.1 * b
        y = base + b * x
        mean_on, sd_h = bm + b * x[:, L:], np.sqrt(bsd ** 2 + b ** 2)
    elif name == "noisy_known":
        persist = PHI ** h * x[:, L - 1:L]
        cov_fut = persist + 0.5 * rng.normal(size=(n, H))
        mean_on, sd_on = bm + b * persist, np.sqrt(bsd ** 2 + b ** 2 * (1 - PHI ** (2 * h)))
    elif name == "noisy_obs":
        seen = x + rng.normal(size=(n, T))
        cov, cov_fut = seen[:, :L], seen[:, L:]
        mean_on, sd_on = bm + 0.5 * b * seen[:, L:], np.sqrt(bsd ** 2 + 0.5 * b ** 2)
    elif name == "broke":
        y = base + b * x * (np.arange(T) < int(0.7 * L))
        mean_on, sd_on, mean_h, sd_h = bm, bsd, bm, bsd
    elif name == "sign_flip":
        tb = rng.integers(int(0.7 * L), int(0.9 * L), size=(n, 1))
        y = base + b * x * np.where(np.arange(T)[None, :] < tb, 1.0, -1.0)
        mean_on = bm - b * x[:, L:]
    elif name in ("spurious", "seasonal_decoy"):
        y = base
        mean_on, sd_on, mean_h, sd_h = bm, bsd, bm, bsd
        if name == "spurious":
            draws = np.stack([ar1(rng, n, T, PHI, sig) for _ in range(K)], 1)
            c = np.abs(_corr(draws[:, :, :L], base[:, None, :L]))
            pick = draws[np.arange(n), c.argmax(1)]
            cov, cov_fut = pick[:, :L], pick[:, L:]
        else:
            phase = np.arange(T) % S
            shape = np.stack([[base[i, :L][phase[:L] == s].mean() for s in range(S)] for i in range(n)])
            seas = shape[:, phase] + 0.3 * rng.normal(size=(n, T))
            cov, cov_fut = seas[:, :L], seas[:, L:]
    a = dict(targets=y[:, None, :L].astype(np.float32), future=y[:, None, L:], oracle_q=normal_q(mean_on, sd_on)[:, None],
             oracle_hidden_q=normal_q(mean_h, sd_h)[:, None], known_ctx=cov[:, None].astype(np.float32),
             known_fut=cov_fut[:, None].astype(np.float32))
    return Cell(name, a, S, control=name == "control", modes=("on", "hidden"), desc=f"{name}, covariate {role}",
                extra=dict(role=role, L=L, strength=STRENGTH[name]))


def build(n_per_cell: int = N_PER_CELL, seed: int = SEED) -> ProbeSet:
    return ProbeSet(PROBE, VERSION, seed, n_per_cell, [_cell(c, n_per_cell, seed) for c in CELLS], "over-trust: covariates to discount")


def metrics(cell: Cell, preds: dict) -> list:
    y, sc = cell.arrays["future"], M.scale_seasonal(cell.arrays["targets"], cell.S)
    out = []
    for mode in cell.modes:
        out += M.standard(preds[(mode, "main")], y, sc, mode)
    on, hid = (M.sql(preds[(m, "main")], y, sc) for m in ("on", "hidden"))
    out.append(M.ratio("harm", on, hid, "on"))
    return out


def harm_summary(harm: dict) -> float:
    """harm: {cell: value}; the mean of max(0, harm - 1) over the non-control cells."""
    return float(np.mean([max(0.0, v - 1) for k, v in harm.items() if k != "control"]))
