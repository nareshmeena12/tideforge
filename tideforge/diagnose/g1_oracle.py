"""G1 oracle check: does a forecast that uses a scenario's true covariates beat one that does not (P1-style learning is
possible)? Per group, ridge regression fitted on the context, scored on the last H points (MAE).

  without  intercept + seasonal design from the row's own clock (hour harmonics 1-3 below daily steps, day of week one-hot,
           yearly harmonic)
  with     the same + per covariate c, c^2, c^3 (known: at t; past: at t - H, the latest value seen at the origin) + the
           first two known covariates times the first two hour harmonics and the weekend flag + products of the first three
           known covariates
The ratio is MAE(with) / MAE(without); a check, not a target (owner 2026-10-28: report it, do not tune to it).
Graph: per parent (single covariate) against intercept + y(t - H), bucketed by role x lag.
Numpy, one group at a time (tests and tools only, never in generation).
"""
from __future__ import annotations

import math

import numpy as np

from tideforge.gen.groups import ROLE_KNOWN, ROLE_PAST

H = 128


def _std(c):
    return (c - c.mean()) / (c.std() + 1e-9)


def _shift(c, k):
    return np.concatenate([np.full(k, c[0]), c[:-k]]) if k else c


def base_design(gr: dict, i: int) -> np.ndarray:
    hour, dow, ang = (np.asarray(gr[k][i].cpu(), np.float64) for k in ("hour", "dow", "ang"))
    cols = [np.ones_like(hour)]
    if gr["dt"] < 24:
        for k in (1, 2, 3):
            cols += [np.cos(2 * math.pi * k * hour / 24), np.sin(2 * math.pi * k * hour / 24)]
    cols += [(dow == d).astype(float) for d in range(1, 7)]
    cols += [np.cos(ang), np.sin(ang)]
    return np.stack(cols, 1)


def cov_design(covs: np.ndarray, roles: np.ndarray, Xb: np.ndarray, hourly: bool) -> np.ndarray:
    cols, known = [], []
    for c, r in zip(covs, roles):
        if np.std(c) < 1e-12:
            continue
        s = _std(c)
        if r == ROLE_PAST:
            s = _shift(s, H)
        elif r == ROLE_KNOWN:
            known.append(s)
        else:
            continue
        cols += [s, s * s, s ** 3]
    for s in known[:2]:
        if hourly:
            cols += [s * Xb[:, 1], s * Xb[:, 2], s * Xb[:, 3], s * Xb[:, 4]]
        cols += [s * (Xb[:, -3] + Xb[:, -4])]                 # dow 5, 6 (weekend) one-hot columns
    for a in range(min(3, len(known))):
        for b in range(a + 1, min(3, len(known))):
            cols.append(known[a] * known[b])
    return np.stack(cols, 1) if cols else np.zeros((Xb.shape[0], 0))


def _fit_mae(X, y, lam=1e-3):
    tr = slice(H, len(y) - H)
    te = slice(len(y) - H, len(y))
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    sd[0] = 1.0 if np.all(X[:, 0] == 1) else sd[0]
    mu[0] = 0.0 if np.all(X[:, 0] == 1) else mu[0]
    Z = (X - mu) / sd
    A = Z[tr].T @ Z[tr] + lam * len(y) * np.eye(Z.shape[1])
    beta = np.linalg.solve(A, Z[tr].T @ y[tr])
    return float(np.mean(np.abs(Z[te] @ beta - y[te])))


def scenario_ratios(tgt, covs, roles, gr) -> np.ndarray:
    """Per group MAE(with true covariates) / MAE(without); tgt (n, L), covs (n, C, L), roles (n, C) torch."""
    y, c, r = (np.asarray(v.cpu(), np.float64) for v in (tgt, covs, roles))
    out = []
    for i in range(y.shape[0]):
        Xb = base_design(gr, i)
        Xc = np.concatenate([Xb, cov_design(c[i], r[i], Xb, gr["dt"] < 24)], 1)
        mb = _fit_mae(Xb, y[i])
        out.append(_fit_mae(Xc, y[i]) / mb if mb > 1e-12 else np.nan)
    return np.array(out)


def lag_bucket(lag: int) -> str:
    return "lag 0" if lag == 0 else f"lag 1-{H - 1}" if lag < H else f"lag >= {H}"


def graph_parent_ratios(X, info, role_known: np.ndarray, strong_only=True) -> dict:
    """One ratio per (group, parent): intercept + y(t - H) with / without the parent (known: x(t - lag); past: x(t -
    max(lag, H))). role_known (n, K) bool: the role each parent got. Returns {(role, bucket): [ratios]}."""
    X = np.asarray(X.cpu(), np.float64)
    A, lag, weak, tg = (np.asarray(info[k].cpu()) for k in ("A", "lag", "weak", "target"))
    res: dict = {}
    for i in range(X.shape[0]):
        t = int(tg[i])
        y = X[i, t]
        Xb = np.stack([np.ones_like(y), _shift(y, H)], 1)
        mb = _fit_mae(Xb, y)
        for p in np.nonzero(A[i, :, t])[0]:
            if strong_only and weak[i, p, t]:
                continue
            ell = int(lag[i, p, t])
            kn = bool(role_known[i, p])
            x = _shift(X[i, p], ell if kn else max(ell, H))
            f = np.stack([x, x * x, x ** 3, (x > 0).astype(float), np.tanh(x)], 1)
            res.setdefault(("known" if kn else "past", lag_bucket(ell)), []).append(
                _fit_mae(np.concatenate([Xb, f], 1), y) / mb)
    return res


def graph_decoy_gain(X, info) -> np.ndarray:
    """Per group: MAE(parents + decoys) / MAE(parents), all parents given as known (best case); decoys: non-parent nodes at
    t, descendants at t - H (they are past-only)."""
    X = np.asarray(X.cpu(), np.float64)
    A, lag, tg, desc, k = (np.asarray(info[n].cpu()) for n in ("A", "lag", "target", "desc", "k"))
    out = []
    for i in range(X.shape[0]):
        t = int(tg[i])
        y = X[i, t]
        cols = [np.ones_like(y), _shift(y, H)]
        par = np.nonzero(A[i, :, t])[0]
        for p in par:
            x = _shift(X[i, p], int(lag[i, p, t]))
            cols += [x, x * x, x ** 3, (x > 0).astype(float), np.tanh(x)]
        Xp = np.stack(cols, 1)
        dec = [j for j in range(int(k[i].reshape(-1)[0])) if j != t and j not in par]
        if not dec:
            continue
        dcols = []
        for j in dec:
            x = _shift(X[i, j], H) if desc[i, j] else X[i, j]
            dcols += [x, x * x, x ** 3]
        out.append(_fit_mae(np.concatenate([Xp, np.stack(dcols, 1)], 1), y) / _fit_mae(Xp, y))
    return np.array(out)
