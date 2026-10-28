"""G9.2 tar family (switch tar): threshold / smooth-transition AR. Probe P9.

  regimes   2 (75%) or 3; AR(2) per regime: phi1 in [-0.9, 1.05], phi2 in [-0.3, 0.3], intercept in [-1, 1], noise sd in
            [0.5, 2]; the outer regimes are made stationary (|phi1| + |phi2| <= 0.98 by rescaling), the middle regime of a
            3-regime row may be locally explosive, so the process stays stationary
  switch    on y(t - d), d in 1..4; thresholds c = m0 + q s0 with q = Phi^-1(U(0.2, 0.8)) (sorted for 3 regimes), m0 / s0 the
            mean / sd of regime 0's linear AR (the pilot)
  forms     2-regime rows: TAR (hard, 40%), LSTAR G = sigmoid(gamma (y - c) / s0) (40%), ESTAR G = 1 - exp(-gamma ((y - c) /
            s0)^2) (20%), gamma in [1, 20]; y = (1 - G) AR_0 + G AR_1. 3-regime rows: hard switching.
  burn-in   256 steps; values clamped to +-1e4 (a safety net, not reached by the stationary draws)
The recursion is a Python loop over time steps, batched over rows (about 25 small ops per step). Param pool (default ""):
the name of an offline pool (tools/build_loop_pool.py --family tar) to crop instead, for when the loop costs too much on
GPU (owner 2026-10-28: switch when it exceeds 10% of B0's generator time at its share; gen_throughput --loops measures).
"""
from __future__ import annotations

import math

import torch

from tideforge.gen.ext import _family, _pool
from tideforge.gen.ext import g1_common as C
from tideforge.gen.ext._rand import bern, randint, u
from tideforge.gen.switches import Switch, register

BURN = 256


def _stationary(phi1, phi2):
    s = phi1.abs() + phi2.abs()
    f = torch.where(s > 0.98, 0.98 / s, torch.ones_like(s))
    return phi1 * f, phi2 * f


def tar_raw(g, n, T, dev):
    """(n, T) raw TAR / STAR paths (burn-in dropped) and the parameters."""
    R = torch.where(bern(g, (n,), 0.25, dev), 3, 2)
    phi1 = u(g, (n, 3), -0.9, 1.05, dev)
    phi2 = u(g, (n, 3), -0.3, 0.3, dev)
    inter = u(g, (n, 3), -1.0, 1.0, dev)
    sig = u(g, (n, 3), 0.5, 2.0, dev)
    outer = torch.zeros(n, 3, dtype=torch.bool, device=dev)
    outer[:, 0] = True
    outer[torch.arange(n, device=dev), R - 1] = True
    a, b = _stationary(phi1, phi2)
    phi1, phi2 = torch.where(outer, a, phi1), torch.where(outer, b, phi2)
    d = randint(g, 1, 4, (n,), dev)
    den = (1 - phi2[:, 0]) ** 2 - phi1[:, 0] ** 2
    var0 = (sig[:, 0] ** 2 * (1 - phi2[:, 0]) / ((1 + phi2[:, 0]) * den.clamp(min=1e-3))).clamp(max=1e4)
    m0 = inter[:, 0] / (1 - phi1[:, 0] - phi2[:, 0]).clamp(min=0.02)
    s0 = var0.sqrt().clamp(min=0.1)
    q = torch.sort(math.sqrt(2) * torch.erfinv(2 * u(g, (n, 2), 0.2, 0.8, dev) - 1), 1).values
    c = m0.view(n, 1) + q * s0.view(n, 1)
    form = C.choice(g, n, (0.4, 0.4, 0.2), dev)                       # 0 TAR, 1 LSTAR, 2 ESTAR (2-regime rows)
    form = torch.where(R == 3, 0, form)
    gam = u(g, (n,), 1.0, 20.0, dev)
    eps = torch.randn((n, T + BURN), generator=g, device=dev)
    y = torch.zeros(n, T + BURN + 4, device=dev)
    ar = torch.arange(n, device=dev)
    for t in range(4, T + BURN + 4):
        z = y[ar, t - d]
        hi1 = (z > c[:, 0]).long()
        reg = torch.where(R == 3, hi1 + (z > c[:, 1]).long(), hi1)
        zz = (z - c[:, 0]) / s0
        G = torch.where(form == 1, torch.sigmoid(gam * zz), 1 - torch.exp(-gam * zz * zz))
        G = torch.where(form == 0, torch.zeros_like(G), G)
        y1, y2, e = y[:, t - 1], y[:, t - 2], eps[:, t - 4]
        lin = phi1 * y1.view(n, 1) + phi2 * y2.view(n, 1) + inter + sig * e.view(n, 1)       # (n, 3): each regime's AR
        hard = lin.gather(1, reg.view(n, 1)).squeeze(1)
        y[:, t] = torch.where(form == 0, hard, (1 - G) * lin[:, 0] + G * lin[:, 1]).clamp(-1e4, 1e4)
    return y[:, BURN + 4:], dict(R=R, phi1=phi1, phi2=phi2, inter=inter, sig=sig, d=d, c=c, form=form, s0=s0)


def make_tar(p):
    def tar(n, L, device):
        g = _family.own_generator(device)
        dev = torch.device(device)
        if not p["pool"]:
            return tar_raw(g, n, L, dev)[0]
        x, _, _ = _pool.crop(g, _pool.load(p["pool"], p["pool_dir"], p["manifest_dir"]), n, L, dev)
        return x[..., 0] * C.log_u(g, (n, 1), 0.1, 10.0, dev)
    return tar


def _install(p, inst):
    if inst.apply and p["pool"]:
        _pool.load(p["pool"], p["pool_dir"], p["manifest_dir"])
    _family.install(inst, "tar", make_tar(p), p)


register(Switch("tar", "G9.2", "threshold / smooth-transition AR family", _install,
                dict(share=0.03, mode="add", pool="", pool_dir="", manifest_dir="")))
