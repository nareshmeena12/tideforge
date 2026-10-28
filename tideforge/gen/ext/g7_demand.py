"""G7.1 demand family (switch demand) and G7.2 life cycle (its parameter lifecycle_p; owner 2026-10-28: G7.2 is compared with
G7.1, not B0). Probe P7.

Counts on an intensity lam(t) = b season(t) promo(t) life(t):
  b         log-uniform in [0.05, 5]
  season    exp(sum_j a_j cos(2 pi t / P_j + ph_j)) over 1-2 bank periods with >= 2 cycles in the window, a_j in [0, 0.5]
  promo     windows at a rate 1 / U(30, 200) per step, 1-10 steps long, uplift x U(1.5, 5)
  counts    Croston rows (40%): demand occurs with probability p_t = min(p season promo, 0.95), p in [0.05, 0.5] (geometric
            inter-demand gaps), size 1 + NB(lam); other rows Poisson(lam) (50%) or NB(lam, k), k in [0.5, 5]. NB is drawn as
            Poisson-lognormal with the NB's mean and variance (log-sd^2 = log(1 + 1 / k)): torch's gamma takes no generator.
  life      (rows with probability lifecycle_p, default 0) launch at tau0: inside the window (50%, tau0 = U(0, 0.6) L) or
            before it (tau0 = -U(0, 2) L); time unit S = L U(0.02, 0.1) steps; Bass ramp F(v) = (1 - e^{-(p+q) v}) /
            (1 + (q/p) e^{-(p+q) v}) normalised to 1 at its plateau, p in [0.005, 0.05], q in [0.2, 0.6], or Gompertz
            exp(-b e^{-c v}) (30%), b in [2, 8], c in [0.1, 0.5]; death inside the window in 30% (tau_end = U(0.4, 1) L, after
            tau0 + 0.1 L), decline exp(-(t - tau_d) / kappa) from tau_d = tau_end - U(0.1, 0.4) L, kappa = (tau_end - tau_d) /
            U(1, 3); other rows decline from inside the window in 50% (no death). Exact zero before tau0 and after tau_end.
Flags: counts, non-negative, passthrough (no later layer: the support layer would redraw the counts). A row left all zero
gets one demand of 1 at a random step (finalize would add noise to a constant row).
"""
from __future__ import annotations

import math

import torch

from tideforge.gen import _state
from tideforge.gen.ext import _family
from tideforge.gen.ext import g1_common as C
from tideforge.gen.ext._rand import bern, randint, u
from tideforge.gen.switches import Switch, register


def _poisson_lognormal(g, lam, k):
    s2 = torch.log1p(1.0 / k)
    z = torch.randn(lam.shape, generator=g, device=lam.device)
    return torch.poisson(lam * torch.exp(torch.sqrt(s2) * z - 0.5 * s2), generator=g)


def season(g, n, L, dev):
    t = torch.arange(L, device=dev, dtype=torch.float32).view(1, L)
    bank = [p for p in _state._PERIOD_BANK_ACTIVE if 2 * p <= L] or [max(2, L // 2)]
    out = torch.zeros(n, L, device=dev)
    for j in range(2):
        P = torch.tensor(bank, dtype=torch.float32, device=dev)[randint(g, 0, len(bank) - 1, (n, 1), dev)]
        on = (bern(g, (n, 1), 0.5, dev) | (j == 0)).float()
        out = out + on * u(g, (n, 1), 0.0, 0.5, dev) * torch.cos(2 * math.pi * t / P + u(g, (n, 1), 0, 2 * math.pi, dev))
    return torch.exp(out)


def life(g, n, L, dev):
    """Envelope (n, L) >= 0 and the alive mask (zeros before launch and after death are exact)."""
    t = torch.arange(L, device=dev, dtype=torch.float32).view(1, L)
    inside = bern(g, (n, 1), 0.5, dev)
    tau0 = torch.where(inside, u(g, (n, 1), 0.0, 0.6, dev) * L, -u(g, (n, 1), 0.0, 2.0, dev) * L)
    S = L * u(g, (n, 1), 0.02, 0.1, dev)
    v = ((t - tau0) / S).clamp(min=0)
    p, q = u(g, (n, 1), 0.005, 0.05, dev), u(g, (n, 1), 0.2, 0.6, dev)
    e = torch.exp(-(p + q) * v)
    bass = (1 - e) / (1 + (q / p) * e)
    gb, gc = u(g, (n, 1), 2.0, 8.0, dev), u(g, (n, 1), 0.1, 0.5, dev)
    gomp = torch.exp(-gb * torch.exp(-gc * v))
    ramp = torch.where(bern(g, (n, 1), 0.3, dev), gomp, bass)
    dies = bern(g, (n, 1), 0.3, dev)
    tend = torch.maximum(u(g, (n, 1), 0.4, 1.0, dev) * L, tau0 + 0.1 * L)
    tend = torch.where(dies, tend, torch.full_like(tend, float("inf")))
    td_die = tend - u(g, (n, 1), 0.1, 0.4, dev) * L
    td_live = torch.where(bern(g, (n, 1), 0.5, dev), u(g, (n, 1), 0.3, 1.0, dev) * L, torch.full_like(tend, float("inf")))
    td = torch.where(dies, td_die, td_live)
    span = torch.where(dies, tend - td_die, 0.5 * L * torch.ones_like(tend))
    kappa = span / u(g, (n, 1), 1.0, 3.0, dev)
    dec = torch.exp(-((t - td).clamp(min=0)) / kappa)
    alive = (t >= tau0) & (t < tend)
    return torch.where(alive, ramp * dec, torch.zeros_like(ramp)), alive, dict(tau0=tau0, tend=tend, td=td)


def demand_parts(g, n, L, dev, lifecycle_p=0.0):
    b = C.log_u(g, (n, 1), 0.05, 5.0, dev)
    promo = C.windows(g, n, L, 1.0 / u(g, (n, 1), 30.0, 200.0, dev), 1, 10, dev)["inside"]
    up = 1 + (u(g, (n, 1), 1.5, 5.0, dev) - 1) * promo.float()
    sea = season(g, n, L, dev)
    env, alive, linfo = life(g, n, L, dev)
    lc = bern(g, (n, 1), lifecycle_p, dev)
    env = torch.where(lc, env, torch.ones_like(env))
    alive = torch.where(lc, alive, torch.ones_like(alive))
    lam = b * sea * up * env
    mode = C.choice(g, n, (0.4, 0.3, 0.3), dev).view(n, 1)            # 0 Croston, 1 Poisson, 2 NB
    k = u(g, (n, 1), 0.5, 5.0, dev)
    p = u(g, (n, 1), 0.05, 0.5, dev)
    occur = torch.rand((n, L), generator=g, device=dev) < (p * sea * up).clamp(max=0.95)
    nb = _poisson_lognormal(g, lam, k)
    pois = torch.poisson(lam, generator=g)
    cro = occur.float() * (1 + nb)
    y = torch.where(mode == 0, cro, torch.where(mode == 1, pois, nb))
    y = torch.where(alive, y, torch.zeros_like(y))
    empty = (y.sum(1, keepdim=True) == 0)
    if bool(empty.any()):
        pos = torch.argmax(alive.float() + torch.rand((n, L), generator=g, device=dev) * 0.5, 1, keepdim=True)
        y = torch.where(empty & (torch.arange(L, device=dev).view(1, L) == pos), torch.ones_like(y), y)
    return y, dict(mode=mode, lc=lc, alive=alive, lam=lam, k=k, p=p, occur=occur, promo=promo, **linfo)


def make_demand(lifecycle_p):
    def demand(n, L, device):
        g = _family.own_generator(device)
        return demand_parts(g, n, L, torch.device(device), lifecycle_p)[0]
    demand.__qualname__ = f"demand_lifecycle_{lifecycle_p:g}"
    return demand


def _install(p, inst):
    _family.install(inst, "demand", make_demand(float(p["lifecycle_p"])), p, is_count=True, is_nonneg=True,
                    allow_add=False, passthrough=True)


register(Switch("demand", "G7.1", "demand family: Poisson / NB / Croston counts, promotions; lifecycle_p = G7.2", _install,
                dict(share=0.05, mode="add", lifecycle_p=0.0)))
