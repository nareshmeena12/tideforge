"""G1.3 weather -> load (switch sc_load, group kind 5+ through H5).

  T     temperature (g1_common.temperature); Te = T smoothed (a in [0, 0.9] per hour: building thermal lag)
  R     m(h) [bh relu(Tc - w/2 - Te)^p + bc (1 + eta (H - 50) / 50) relu(Te - Tc - w/2)^p]       U-shaped in Te
        comfort point Tc in [14, 22], dead band w in [0, 6], p in [1, 1.5], bh, bc in [0, 0.04] / deg C (25% of groups:
        one of them 0, a heating-only or cooling-only climate), m(h) = 1 + kappa cos(2 pi (h - hm) / 24), kappa in [0, 0.6]
  B     daily profile 1 + sum_{k=1..3} a_k cos(2 pi k h / 24 + ph_k), a_k in [0, 0.15 / k]; weekend factor in [0.7, 0.95]
  load  S B (1 + R) (1 - zeta I / 1000) exp(sigma AR), sigma in [0.01, 0.05], zeta in [0, 0.1]; plus an offset in
        [-0.5, 0.5] S and 0-0.3 sd of the block's own v0 series (scenario_kind own=True)
Covariates (priority): T known, weekend flag known, humidity H known, irradiance I known, wind speed (decoy), neighbour-region
load past (its own parameters, temperature T + 2 AR). Resolutions RES_LOAD (daily rows: built hourly, mean of 24).
"""
from __future__ import annotations

import math

import torch

from tideforge.gen.ext import g1_common as C
from tideforge.gen.ext._rand import bern, u
from tideforge.gen.groups import ROLE_KNOWN, ROLE_PAST
from tideforge.gen.switches import Switch, register


def load_core(g, n, gr, T, H, I, dev) -> tuple[torch.Tensor, dict]:
    """Load shape (mean ~1) from temperature T, humidity H, irradiance I (n, L). Returns (load, parameters)."""
    dt = gr["dt"]
    Te = C.ema(T, u(g, (n, 1), 0.0, 0.9, dev) ** dt)
    Tc, w, p = u(g, (n, 1), 14, 22, dev), u(g, (n, 1), 0, 6, dev), u(g, (n, 1), 1.0, 1.5, dev)
    bh, bc = u(g, (n, 1), 0, 0.04, dev), u(g, (n, 1), 0, 0.04, dev)
    one = bern(g, (n, 1), 0.25, dev)
    side = bern(g, (n, 1), 0.5, dev)
    bh = torch.where(one & side, torch.zeros_like(bh), bh)
    bc = torch.where(one & ~side, torch.zeros_like(bc), bc)
    eta = u(g, (n, 1), 0.0, 0.5, dev)
    kap, hm = u(g, (n, 1), 0.0, 0.6, dev), u(g, (n, 1), 0.0, 24.0, dev)
    m = 1 + kap * torch.cos(2 * math.pi * (gr["hour"] - hm) / 24)
    heat = torch.relu(Tc - w / 2 - Te) ** p
    cool = torch.relu(Te - Tc - w / 2) ** p
    R = m * (bh * heat + bc * (1 + eta * (H - 50) / 50) * cool)
    B = torch.ones_like(T)
    for k in (1, 2, 3):
        B = B + u(g, (n, 1), 0.0, 0.15 / k, dev) * torch.cos(2 * math.pi * k * gr["hour"] / 24 + u(g, (n, 1), 0, 2 * math.pi, dev))
    wf = u(g, (n, 1), 0.7, 0.95, dev)
    B = B * torch.where(gr["dow"] >= 5, wf, torch.ones_like(wf))
    zeta = u(g, (n, 1), 0.0, 0.1, dev)
    noise = torch.exp(u(g, (n, 1), 0.01, 0.05, dev) * C.ar1(g, T.shape, 0.9 ** dt, dev))
    load = B * (1 + R) * (1 - zeta * I / 1000.0) * noise
    return load, dict(Te=Te, Tc=Tc, w=w, bh=bh, bc=bc, B=B, R=R)


def humidity(g, n, gr, dev):
    rho = u(g, (n, 1), 0.9, 0.99, dev) ** gr["dt"]
    return (u(g, (n, 1), 40, 75, dev) + 15 * C.ar1(g, gr["th"].shape, rho, dev)
            - 10 * torch.cos(2 * math.pi * (gr["hour"] - 15) / 24)).clamp(5, 100)


def wind_speed(g, n, gr, dev):
    rho = u(g, (n, 1), 0.95, 0.995, dev) ** gr["dt"]
    return torch.nn.functional.softplus(u(g, (n, 1), 1, 4, dev) + 2 * C.ar1(g, gr["th"].shape, rho, dev))


def build(g, n, Lf, dt, dev):
    gr = C.grid(g, n, Lf, dt, dev)
    T = C.temperature(g, n, gr, dev)
    H = humidity(g, n, gr, dev)
    I = C.clear_sky(gr, torch.deg2rad(u(g, (n, 1), 0, 60, dev))) * C.cloud(g, n, gr, dev)
    load, info = load_core(g, n, gr, T, H, I, dev)
    T2 = T + 2 * C.ar1(g, T.shape, 0.95 ** dt, dev)
    nb, _ = load_core(g, n, gr, T2, H, I, dev)
    S = C.log_u(g, (n, 1), 0.1, 1e4, dev)
    tgt = S * load + u(g, (n, 1), -0.5, 0.5, dev) * S
    covs = torch.stack([T, (gr["dow"] >= 5).float(), H, I, wind_speed(g, n, gr, dev), C.rescale(g, nb, dev)], 1)
    roles = torch.tensor([ROLE_KNOWN, ROLE_KNOWN, ROLE_KNOWN, ROLE_KNOWN, C.DECOY, ROLE_PAST], device=dev).expand(n, -1)
    info.update(agg=("mean",) * 7, gr=gr, load=load, S=S)
    return tgt, covs, roles, info


def _install(p, inst):
    inst.add_group_kind("sc_load", C.scenario_kind(build, C.RES_LOAD, own=True), p["share"])


register(Switch("sc_load", "G1.3", "weather -> load scenario (group kind)", _install, dict(share=0.15)))
