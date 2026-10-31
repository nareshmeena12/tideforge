"""G9.1: build the ODE trajectory pool offline (scipy solve_ivp, LSODA). Run once on Perlmutter; commit only the manifest.

    python tools/build_ode_pool.py --out $TF_DATA/pools [--name ode_v1] [--per-system 4096] [--T 16384] [--seed 0]
                                   [--workers N]     # default: every CPU of the allocation
    python tools/build_ode_pool.py --out DIR --manifest-dir DIR --per-system 4 --T 512      # tiny pool (tests)

Writes <out>/<name>/traj.npy (float16, N x T x 3), meta.npy and manifest.json (commit that as tideforge/gen/ext/pools/<name>.json;
sha256 of both files, scipy / numpy versions, seed, arguments). Each trajectory has its own seed (numpy SeedSequence of (seed, system,
index)), so the result does not depend on the number of workers; a failed or non-finite solve redraws its parameters from
the next sub-seed. Channels are standardised per trajectory (mean / sd kept in meta) before the float16 cast.

Systems (index: name, channels; parameters uniform unless stated; step dt chosen so one cycle spans 16-400 steps):
  0 lotka_volterra  x' = a x - b x y, y' = d x y - g y; a, g in [0.5, 1.5], b in [0.2, 0.8], d in [0.1, 0.5];
                    x0, y0 log-uniform in [0.2, 5]; no burn-in (every state lies on its orbit)
  1 forced_sir      S' = mu - beta(t) S I - mu S - eps S, I' = beta(t) S I - (gamma + mu) I + eps S, beta(t) = R0 (gamma
                    + mu) (1 + delta cos(2 pi t / 365)); R0 in [1.5, 15], gamma in [1/14, 1/3] per day, delta in [0, 0.3],
                    mu = 1 / (50 x 365), immigration eps log-uniform in [1e-7, 1e-5] (bounds fade-outs); time in days;
                    dt 1 or 7 days; burn-in 60 years; channels S, log I
  2 van_der_pol     x' = y, y' = mu (1 - x^2) y - x + A cos(w t); mu in [0.1, 5]; forced (A in [0.5, 5], w in [0.5, 2]) in
                    50%; burn-in 50 cycles
  3 duffing         x' = y, y' = -delta y + x - x^3 + gamma cos(w t); delta in [0.1, 0.3], gamma in [0.2, 0.65],
                    w in [1, 1.4] (periodic and chaotic settings); burn-in 100 forcing periods
  4 lorenz          sigma 10, rho in [24, 35], beta 8/3; dt in [0.005, 0.05]; burn-in 50 time units
meta columns: system, channels used, dt, mean x3, sd x3, parameters x6 (unused 0).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SYSTEMS = ("lotka_volterra", "forced_sir", "van_der_pol", "duffing", "lorenz")
M = 15


def _draw(sid, rng):
    """(rhs, y0, dt, burn, params, channels, post) for one trajectory."""
    U = rng.uniform
    if sid == 0:
        a, b, d, g = U(0.5, 1.5), U(0.2, 0.8), U(0.1, 0.5), U(0.5, 1.5)
        per = 2 * math.pi / math.sqrt(a * g)
        y0 = np.exp(U(math.log(0.2), math.log(5.0), 2))
        return (lambda t, y: [a * y[0] - b * y[0] * y[1], d * y[0] * y[1] - g * y[1]], y0, per / U(16, 400), 0.0,
                [a, b, d, g], 2, None)
    if sid == 1:
        R0, gam, dl, mu = U(1.5, 15), U(1 / 14, 1 / 3), U(0, 0.3), 1 / (50 * 365)
        eps = math.exp(U(math.log(1e-7), math.log(1e-5)))           # immigration: keeps fade-outs bounded
        b0 = R0 * (gam + mu)

        def f(t, y):
            S, lI = y
            I = math.exp(lI)
            bt = b0 * (1 + dl * math.cos(2 * math.pi * t / 365))
            return [mu - bt * S * I - mu * S - eps * S, bt * S - (gam + mu) + eps * S / I]       # d log I / dt
        S0 = U(0.05, 1.0 / R0 + 0.3)
        return f, [min(S0, 0.99), math.log(U(1e-5, 1e-3))], 1.0 if rng.random() < 0.5 else 7.0, 60 * 365.0, \
            [R0, gam, dl, mu, eps], 2, None
    if sid == 2:
        mu = U(0.1, 5)
        A, w = (U(0.5, 5), U(0.5, 2)) if rng.random() < 0.5 else (0.0, 1.0)
        per = 2 * math.pi if mu < 1 else (3 - 2 * math.log(2)) * mu + 2 * math.pi / 3
        return (lambda t, y: [y[1], mu * (1 - y[0] ** 2) * y[1] - y[0] + A * math.cos(w * t)], U(-2, 2, 2),
                per / U(16, 400), 50 * per, [mu, A, w], 2, None)
    if sid == 3:
        dl, gm, w = U(0.1, 0.3), U(0.2, 0.65), U(1.0, 1.4)
        per = 2 * math.pi / w
        return (lambda t, y: [y[1], -dl * y[1] + y[0] - y[0] ** 3 + gm * math.cos(w * t)], U(-1.5, 1.5, 2),
                per / U(16, 200), 100 * per, [dl, gm, w], 2, None)
    rho = U(24, 35)
    return (lambda t, y: [10 * (y[1] - y[0]), y[0] * (rho - y[2]) - y[1], y[0] * y[1] - 8 / 3 * y[2]],
            U(-10, 10, 3) + np.array([0, 0, 25]), U(0.005, 0.05), 50.0, [10, rho, 8 / 3], 3, None)


def one(args):
    from scipy.integrate import solve_ivp
    seed, sid, i, T = args
    for attempt in range(20):
        rng = np.random.default_rng(np.random.SeedSequence([seed, sid, i, attempt]))
        f, y0, dt, burn, par, C, _ = _draw(sid, rng)
        t_eval = burn + dt * np.arange(T)
        try:
            r = solve_ivp(f, (0.0, float(t_eval[-1])), np.asarray(y0, float), method="LSODA", t_eval=t_eval,
                          rtol=1e-8, atol=1e-10)
        except Exception:                                       # noqa: BLE001 (any solver failure: redraw)
            continue
        if not r.success or r.y.shape[1] != T or not np.isfinite(r.y).all():
            continue
        y = r.y[:C].T
        sd = y.std(0)
        if (sd < 1e-8).any():
            continue
        mean = y.mean(0)
        out = np.zeros((T, 3), np.float32)
        out[:, :C] = (y - mean) / sd
        meta = np.zeros(M, np.float32)
        meta[0], meta[1], meta[2] = sid, C, dt
        meta[3:3 + C], meta[6:6 + C] = mean, sd
        meta[9:9 + len(par)] = par
        return out, meta
    raise RuntimeError(f"system {SYSTEMS[sid]} trajectory {i}: 20 failed draws")


def main(argv=None) -> int:
    import scipy
    from tideforge.gen.ext import _pool
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="ode_v1")
    ap.add_argument("--per-system", type=int, default=4096)
    ap.add_argument("--T", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=_pool.cpu_count())
    ap.add_argument("--manifest-dir", default=None, help="default: next to the pool")
    a = ap.parse_args(argv)
    jobs = [(a.seed, s, i, a.T) for s in range(len(SYSTEMS)) for i in range(a.per_system)]
    traj = np.zeros((len(jobs), a.T, 3), np.float16)
    meta = np.zeros((len(jobs), M), np.float32)
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            res = ex.map(one, jobs, chunksize=8)
            for k, (x, m) in enumerate(res):
                traj[k], meta[k] = x, m
    else:
        for k, j in enumerate(jobs):
            traj[k], meta[k] = one(j)
    sys.path[:0] = [str(REPO)]
    info = dict(builder=f"python tools/build_ode_pool.py --out $TF_DATA/pools --name {a.name} --per-system {a.per_system} "
                        f"--T {a.T} --seed {a.seed}", seed=a.seed, systems=list(SYSTEMS), per_system=a.per_system,
                versions=dict(scipy=scipy.__version__, numpy=np.__version__, python=sys.version.split()[0]),
                meta_columns="system, channels, dt, mean x3, sd x3, params x6")
    man = _pool.write(Path(a.out), a.name, traj, meta, info, a.manifest_dir)
    print(json.dumps(man, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
