"""G1 covariate scenarios: the pieces they share (time grid, smooth noise, event windows, resolutions, lane assembly).

A scenario is a group kind (H5). Its builder makes one target and C covariates at a fine step for a given resolution:
    build(g, n, Lf, dt, dev) -> (tgt (n, Lf), covs (n, C, Lf), roles (n, C), info)
roles hold ROLE_KNOWN / ROLE_PAST or DECOY (a known or past lane, drawn per group, that does not drive the target).
`scenario_kind` turns a builder into the H5 function: it draws each group's resolution, builds at the fine step, aggregates by
reshape (sum / mean / max per lane), puts the target in lane 0 and the first g - 1 covariates (priority order) in a random lane
order; lanes left over in the group get the block's own v0 series as decoys. Drivers that do not fit stay in the target's
equation unobserved. No timestamps exist: hour, day of week and year angle come from the step dt (hours) and a random start
offset per group. Every draw uses the switch's own generator; the only Python loops are over resolutions and graph nodes.
"""
from __future__ import annotations

import math

import torch

from tideforge.gen.ext._rand import bern, u
from tideforge.gen.groups import ROLE_KNOWN, ROLE_PAST, ROLE_TARGET

DECOY = -1
YEAR_H = 8766.0
AGG = {"mean": 0, "sum": 1, "max": 2}

# (dt hours, aggregation factor, probability): owner 2026-10-28, matching fev's task resolutions
RES_LOAD = ((0.25, 1, 0.20), (0.5, 1, 0.15), (1.0, 1, 0.45), (1.0, 24, 0.20))
RES_PV = ((0.25, 1, 0.25), (0.5, 1, 0.1875), (1.0, 1, 0.5625))              # no daily rows: 0.2 spread over the others
RES_PRICE = ((0.25, 1, 0.30), (1.0, 1, 0.70))
RES_DAILY = ((24.0, 1, 0.75), (24.0, 7, 0.25))                              # retail, events: daily / weekly (sums of 7)


def randn(g, shape, dev):
    return torch.randn(shape, generator=g, device=dev)


def log_u(g, shape, lo, hi, dev):
    return torch.exp(u(g, shape, math.log(lo), math.log(hi), dev))


def choice(g, n, probs, dev) -> torch.Tensor:
    """n draws of an index with the given probabilities (own generator, on dev)."""
    c = torch.cumsum(torch.tensor(probs, dtype=torch.float32, device=dev), 0)
    return torch.searchsorted(c / c[-1], torch.rand(n, generator=g, device=dev), right=True).clamp(max=len(probs) - 1)


def grid(g, n, L, dt, dev) -> dict:
    """Hours since a random start (aligned on the step), hour of day, day index, day of week, day of year, year angle."""
    t0 = torch.floor(u(g, (n, 1), 0.0, YEAR_H / dt, dev)) * dt
    th = t0 + dt * torch.arange(L, device=dev, dtype=torch.float32).view(1, L)
    day = torch.floor(th / 24.0)
    return dict(th=th, hour=torch.remainder(th, 24.0), day=day, dow=torch.remainder(day, 7.0),
                doy=torch.floor(torch.remainder(day, 365.25)), ang=2 * math.pi * th / YEAR_H, dt=dt)


def ar1(g, shape, rho, dev) -> torch.Tensor:
    """Stationary unit-variance AR(1) along the last axis, rho broadcastable to shape[:-1] + (1,): filtered in the frequency
    domain on twice the length (circular), the second half kept."""
    L = shape[-1]
    M = 2 * L
    e = randn(g, tuple(shape[:-1]) + (M,), dev)
    w = 2 * math.pi * torch.fft.rfftfreq(M, device=dev)
    rho = torch.as_tensor(rho, device=dev, dtype=torch.float32)
    H = 1.0 / (1.0 - rho.to(torch.complex64) * torch.exp(-1j * w))
    y = torch.fft.irfft(torch.fft.rfft(e) * H, n=M)[..., L:]
    return y * torch.sqrt((1.0 - rho * rho).clamp(min=1e-6))


def ema(x, a) -> torch.Tensor:
    """Causal exponential smoothing y_t = a y_{t-1} + (1 - a) x_t along the last axis, started at x_0 (zero-padded FFT)."""
    L = x.shape[-1]
    x0 = x[..., :1]
    a = torch.as_tensor(a, device=x.device, dtype=torch.float32)
    w = 2 * math.pi * torch.fft.rfftfreq(2 * L, device=x.device)
    H = (1.0 - a).to(torch.complex64) / (1.0 - a.to(torch.complex64) * torch.exp(-1j * w))
    return torch.fft.irfft(torch.fft.rfft(x - x0, n=2 * L) * H, n=2 * L)[..., :L] + x0


def decay(x, a) -> torch.Tensor:
    """Impulses x decaying by a per step: y_t = a y_{t-1} + x_t."""
    a = torch.as_tensor(a, device=x.device, dtype=torch.float32)
    return ema(x, a) / (1.0 - a).clamp(min=1e-6) if x.shape[-1] else x


def windows(g, n, L, rate, len_lo, len_hi, dev) -> dict:
    """Event windows: starts with probability `rate` (n, 1) per step, lengths in [len_lo, len_hi] steps. Returns masks and,
    per step, the index of the last start, steps since the last window ended (inf: none) and until the next start (inf)."""
    t = torch.arange(L, device=dev).view(1, L).expand(n, L)
    start = torch.rand((n, L), generator=g, device=dev) < rate
    ln = torch.randint(int(len_lo), int(len_hi) + 1, (n, L), generator=g, device=dev)
    ls = torch.cummax(torch.where(start, t, torch.full_like(t, -1)), 1).values
    has = ls >= 0
    end = ls + torch.gather(ln, 1, ls.clamp(min=0))
    ns = torch.flip(torch.cummin(torch.flip(torch.where(start, t, torch.full_like(t, 10 ** 9)), [1]), 1).values, [1])
    inf = torch.tensor(float("inf"), device=dev)
    return dict(start=start, last=ls.clamp(min=0), next=ns.clamp(max=L - 1), has=has, inside=has & (t < end),
                since_end=torch.where(has & (t >= end), (t - end).float(), inf),
                until=torch.where(ns < 10 ** 9, (ns - t).float(), inf))


def quantile(x, q):
    """Per-row quantile of x (n, L), q (n, 1)."""
    s = torch.sort(x, 1).values
    return torch.gather(s, 1, torch.floor(q * (x.shape[1] - 1)).long())


def at(v, idx):
    """v (n, L) read at per-step indices idx (n, L)."""
    return torch.gather(v, 1, idx)


def temperature(g, n, gr, dev) -> torch.Tensor:
    """Air temperature (deg C): yearly and daily cycles plus AR weather noise (rho per hour 0.97-0.995, sd 1-4)."""
    dt = gr["dt"]
    Tm, Ay, Ad = u(g, (n, 1), -5, 25, dev), u(g, (n, 1), 3, 15, dev), u(g, (n, 1), 2, 8, dev)
    hpk = u(g, (n, 1), 13, 17, dev)
    rho = u(g, (n, 1), 0.97, 0.995, dev) ** dt
    w = u(g, (n, 1), 1, 4, dev) * ar1(g, gr["th"].shape, rho, dev)
    return Tm - Ay * torch.cos(gr["ang"]) + Ad * torch.cos(2 * math.pi * (gr["hour"] - hpk) / 24) + w


def clear_sky(gr, lat) -> torch.Tensor:
    """Clear-sky irradiance (W/m2) from solar elevation: day of year and a latitude (radians, (n, 1)); 0 at night."""
    decl = math.radians(23.44) * torch.sin(2 * math.pi * (gr["th"] / 24.0 - 81.0) / 365.25)
    ha = torch.deg2rad(15.0 * (gr["hour"] - 12.0))
    se = torch.sin(lat) * torch.sin(decl) + torch.cos(lat) * torch.cos(decl) * torch.cos(ha)
    return 1000.0 * se.clamp(min=0.0) ** 1.15


def cloud(g, n, gr, dev) -> torch.Tensor:
    """Cloud factor in (0.1, 1]: 1 - kappa sigmoid(AR), kappa 0-0.9 (climate), rho per hour 0.8-0.98."""
    rho = u(g, (n, 1), 0.8, 0.98, dev) ** gr["dt"]
    return 1.0 - u(g, (n, 1), 0.0, 0.9, dev) * torch.sigmoid(2.0 * ar1(g, gr["th"].shape, rho, dev))


def holidays_fixed(g, n, gr, lo, hi, dev) -> torch.Tensor:
    """Default holiday source: lo-hi fixed day-of-year positions per group, repeating every year (a plain flag).
    G2's calendar engine replaces it through HOLIDAYS (same signature)."""
    K = torch.randint(lo, hi + 1, (n, 1), generator=g, device=dev)
    pos = torch.floor(u(g, (n, hi), 0.0, 365.0, dev))
    pos = torch.where(torch.arange(hi, device=dev).view(1, hi) < K, pos, torch.full_like(pos, -1.0))
    return (gr["doy"].unsqueeze(-1) == pos.unsqueeze(1)).any(-1)


HOLIDAYS = {"fixed": holidays_fixed}            # name -> fn(g, n, gr, lo, hi, dev) -> (n, L) bool; switch param `holidays`


def rescale(g, x, dev) -> torch.Tensor:
    """Per-lane unit: x times exp(U(log 0.1, log 10)) (sign and zeros kept)."""
    return x * log_u(g, x.shape[:-1] + (1,), 0.1, 10.0, dev)


def counts(g, lam, low) -> torch.Tensor:
    """Poisson counts for the groups flagged low (n, 1), the rate itself elsewhere."""
    return torch.where(low, torch.poisson(lam.clamp(min=0.0), generator=g), lam)


def _aggregate(x, k, mode):
    if k == 1:
        return x
    v = x.view(*x.shape[:-1], x.shape[-1] // k, k)
    return v.sum(-1) if mode == "sum" else v.amax(-1) if mode == "max" else v.mean(-1)


def by_resolution(g, n, L, res, build, dev):
    """Draw each group's resolution, build at the fine step (length L x k), aggregate by reshape.
    Returns tgt (n, L), covs (n, C, L), roles (n, C), pick (n,) = resolution index, infos per resolution."""
    pick = choice(g, n, [r[2] for r in res], dev)
    tgt = covs = roles = None
    infos = {}
    for i, (dt, k, _) in enumerate(res):
        idx = (pick == i).nonzero().squeeze(1)
        if idx.numel() == 0:
            continue
        t, c, r, info = build(g, idx.numel(), L * k, dt, dev)
        agg = info["agg"]
        t = _aggregate(t, k, agg[0])
        c = torch.stack([_aggregate(c[:, j], k, agg[1 + j]) for j in range(c.shape[1])], 1)
        if tgt is None:
            tgt = torch.zeros(n, L, device=dev)
            covs = torch.zeros(n, c.shape[1], L, device=dev)
            roles = torch.zeros(n, c.shape[1], dtype=torch.long, device=dev)
        tgt[idx], covs[idx], roles[idx] = t, c, r
        infos[i] = info
    return tgt, covs, roles, pick, infos


def assemble(g, z, act, mu, sd, tgt, covs, roles):
    """Target in lane 0; covariates in priority order, the first g - 1 of them in a random lane order; extra lanes get the
    block's v0 series (decoys). Decoys get a random role (known / past, 0.5 each)."""
    n, V, L = z.shape
    dev = z.device
    filler = (z[:, 1:] * sd[:, 1:] + mu[:, 1:])
    cand = torch.cat([covs, filler], 1)[:, :V - 1]
    cr = torch.cat([roles, torch.full((n, V - 1), DECOY, dtype=torch.long, device=dev)], 1)[:, :V - 1]
    rnd = torch.where(bern(g, (n, V - 1), 0.5, dev), ROLE_KNOWN, ROLE_PAST)
    cr = torch.where(cr == DECOY, rnd, cr)
    k = (act.sum(1, keepdim=True) - 1).clamp(min=0)
    j = torch.arange(V - 1, device=dev).view(1, -1)
    keys = torch.where(j < k, torch.rand((n, V - 1), generator=g, device=dev), 2.0 + j.float())
    order = torch.argsort(keys, 1)
    cl = torch.gather(cand, 1, order.unsqueeze(-1).expand(n, V - 1, L))
    rl = torch.gather(cr, 1, order)
    y = torch.cat([tgt.unsqueeze(1), cl], 1)
    role = torch.cat([torch.full((n, 1), ROLE_TARGET, dtype=torch.long, device=dev), rl], 1)
    return y, role


def scenario_kind(build, res, own: bool):
    """The H5 function for a scenario builder. own: add 0-0.3 sd of the block's own v0 series to the target (additive
    targets only: load, price; the others keep exact zeros and counts)."""
    def fn(z, act, mu, sd, ctx):
        g, dev = ctx.rng(), z.device
        n, V, L = z.shape
        tgt, covs, roles, _, _ = by_resolution(g, n, L, res, build, dev)
        if own:
            tgt = tgt + u(g, (n, 1), 0.0, 0.3, dev) * tgt.std(1, keepdim=True) * z[:, 0]
        return assemble(g, z, act, mu, sd, tgt, covs, roles)
    return fn
