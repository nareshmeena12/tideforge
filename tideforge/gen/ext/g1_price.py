"""G1.4 residual load -> price (switch sc_price, group kind through H5). Resolutions RES_PRICE (15 min / hourly).

  load   g1_load.load_core (mean ~1)
  wind   W = Cw sigmoid(a + b u), u AR (rho per hour 0.98-0.998), Cw in [0, 0.6]: persistent, >= 0
  solar  Sol = Cs clear-sky / 1000 x cloud factor, Cs in [0, 0.5]: diurnal, 0 at night
  r      (load - W - Sol) / mean(load)                                                       residual load
  fuel   F = exp(random walk), step sd in [0.002, 0.01] per hour
  price  F [a + b (softplus(k (r - r0)) / k / sd(r))^gamma] + spikes, convex in r: a in [0.2, 1], b in [0.5, 2],
         gamma in [1.5, 3], k = 8 / sd(r), r0 = the q0-quantile of r (q0 in [0.3, 0.6]);
         spikes where r is above its 90th percentile, probability in [0.02, 0.1] per hour, size lognormal x [2, 10] x mean
         price, decaying with tau in [1, 3] h; where r is below its q_low-quantile (q_low in [0.02, 0.1]) the price is
         F c_low a, c_low in [-0.5, 0.05] (negative or near zero); then x S, S in [10, 200], no offset; plus 0-0.3 sd of the
         block's own v0 series
Covariates (priority): load known, wind known, solar known, fuel past, neighbour-market price past (same fuel, residual
plus noise), temperature known (a decoy given load).
"""
from __future__ import annotations

import math

import torch

from tideforge.gen.ext import g1_common as C
from tideforge.gen.ext import g1_load
from tideforge.gen.ext._rand import u
from tideforge.gen.groups import ROLE_KNOWN, ROLE_PAST
from tideforge.gen.switches import Switch, register


def curve(r, a, b, gamma, r0, sdr):
    k = 8.0 / sdr
    return a + b * (torch.nn.functional.softplus(k * (r - r0)) / k / sdr) ** gamma


def build(g, n, Lf, dt, dev, clock="fixed"):
    gr = C.grid(g, n, Lf, dt, dev, clock)
    T = C.temperature(g, n, gr, dev)
    H = g1_load.humidity(g, n, gr, dev)
    cs = C.clear_sky(gr, torch.deg2rad(u(g, (n, 1), 0, 60, dev)))
    Sol = u(g, (n, 1), 0.0, 0.5, dev) * cs / 1000.0 * C.cloud(g, n, gr, dev)
    load, _ = g1_load.load_core(g, n, gr, T, H, cs, dev)
    load = load / load.mean(1, keepdim=True)
    uw = C.ar1(g, T.shape, u(g, (n, 1), 0.98, 0.998, dev) ** dt, dev)
    W = u(g, (n, 1), 0.0, 0.6, dev) * torch.sigmoid(u(g, (n, 1), -1.5, 0.5, dev) + u(g, (n, 1), 1.0, 2.5, dev) * uw)
    r = load - W - Sol
    sdr = r.std(1, keepdim=True) + 1e-6
    F = torch.exp(torch.cumsum(u(g, (n, 1), 0.002, 0.01, dev) * math.sqrt(dt) * C.randn(g, T.shape, dev), 1))
    a, b, gamma = u(g, (n, 1), 0.2, 1.0, dev), u(g, (n, 1), 0.5, 2.0, dev), u(g, (n, 1), 1.5, 3.0, dev)
    r0 = C.quantile(r, u(g, (n, 1), 0.3, 0.6, dev))
    base = curve(r, a, b, gamma, r0, sdr)
    hi = r > C.quantile(r, torch.full((n, 1), 0.9, device=dev))
    ps = u(g, (n, 1), 0.02, 0.1, dev) * dt
    s = hi & (torch.rand(T.shape, generator=g, device=dev) < ps)
    size = torch.exp(0.5 * C.randn(g, T.shape, dev)) * u(g, (n, 1), 2, 10, dev) * base.mean(1, keepdim=True)
    spikes = C.decay(s.float() * size, torch.exp(-dt / u(g, (n, 1), 1.0, 3.0, dev)))
    lo = r < C.quantile(r, u(g, (n, 1), 0.02, 0.1, dev))
    c_low = u(g, (n, 1), -0.5, 0.05, dev) * a
    price = F * torch.where(lo, c_low.expand_as(r), base) + F * spikes
    price = price + u(g, (n, 1), 0.01, 0.05, dev) * price.abs().mean(1, keepdim=True) * C.ar1(g, T.shape, 0.7 ** dt, dev)
    S = u(g, (n, 1), 10, 200, dev)
    r_nb = r + 0.1 * sdr * C.ar1(g, T.shape, 0.9 ** dt, dev)
    nb = F * curve(r_nb, a * u(g, (n, 1), 0.8, 1.2, dev), b, gamma, r0, sdr)
    covs = torch.stack([C.rescale(g, load, dev), C.rescale(g, W, dev), C.rescale(g, Sol, dev), F, S * nb, T], 1)
    roles = torch.tensor([ROLE_KNOWN, ROLE_KNOWN, ROLE_KNOWN, ROLE_PAST, ROLE_PAST, ROLE_KNOWN], device=dev).expand(n, -1)
    info = dict(agg=("mean",) * 7, gr=gr, r=r, F=F, base=base, spikes=spikes, lo=lo, hi=hi, S=S, a=a, c_low=c_low)
    return S * price, covs, roles, info


def _install(p, inst):
    def b(g, n, Lf, dt, dev):
        return build(g, n, Lf, dt, dev, clock=p["clock"])
    inst.add_group_kind("sc_price", C.scenario_kind(b, C.RES_PRICE, own=True), p["share"])


register(Switch("sc_price", "G1.4", "residual load -> price scenario (group kind)", _install, dict(share=0.15, clock="fixed")))
