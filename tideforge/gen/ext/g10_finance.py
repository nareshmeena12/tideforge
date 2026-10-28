"""G10.1 garch_jump family (switch garch_jump): GJR-GARCH leverage plus compound-Poisson jumps. Probe P10.

Returns in units of their unconditional sd: r_t = m + e_t + J_t, e_t = sigma_t eps_t,
  variance  sigma2_t = w + (alpha + gamma 1[e_{t-1} < 0]) e2_{t-1} + beta sigma2_{t-1}, alpha in [0.02, 0.1], gamma in
            [0.05, 0.2], beta in [0.85, 0.97], alpha + gamma / 2 + beta <= 0.995 (beta lowered), w = 1 - alpha - gamma/2 - beta
  eps       Student-t, nu in {3, ..., 10} (chi2 as a sum of nu squared normals: torch's gamma takes no generator), scaled to
            unit variance
  jumps     N_t ~ Poisson(lambda), lambda in [0.002, 0.02] per step; J_t = mu_J N_t + sigma_J sqrt(N_t) Z, mu_J in [-3, 1],
            sigma_J in [2, 6] (sd units: the proposal's [-0.05, 0.02] / [0.02, 0.1] would not be jumps)
  m         in [-0.05, 0.05]; burn-in 500 steps
Output: price rows (60%) P0 exp(cumsum(v r)), v log-uniform in [0.003, 0.03] per step, P0 log-uniform in [1, 1000]; return
rows (40%) v r. Flags: no additive layers (allow_add False).
The variance recursion is a Python loop over time steps (about 10 small ops per step). Param pool (default ""): an offline
pool of r (tools/build_loop_pool.py --family garch_jump) to crop instead; see g9_tar.py.
"""
from __future__ import annotations

import torch

from tideforge.gen.ext import _family, _pool
from tideforge.gen.ext import g1_common as C
from tideforge.gen.ext._rand import bern, randint, u
from tideforge.gen.switches import Switch, register

BURN = 500


def garch_raw(g, n, T, dev):
    """(n, T) returns r in sd units (burn-in dropped) and the parameters."""
    a, gm, b = u(g, (n, 1), 0.02, 0.1, dev), u(g, (n, 1), 0.05, 0.2, dev), u(g, (n, 1), 0.85, 0.97, dev)
    b = torch.minimum(b, 0.995 - a - gm / 2)
    w = 1 - a - gm / 2 - b
    nu = randint(g, 3, 10, (n, 1), dev)
    M = T + BURN
    k = torch.arange(10, device=dev).view(1, 1, 10)
    chi = ((torch.randn((n, M, 10), generator=g, device=dev) ** 2) * (k < nu.view(n, 1, 1))).sum(-1)
    nuf = nu.float()
    eps = torch.randn((n, M), generator=g, device=dev) / torch.sqrt(chi / nuf) / torch.sqrt(nuf / (nuf - 2))
    s2 = torch.ones(n, device=dev)
    e_prev = torch.zeros(n, device=dev)
    e = torch.empty(n, M, device=dev)
    a1, g1, b1, w1 = a.view(n), gm.view(n), b.view(n), w.view(n)
    for t in range(M):
        s2 = w1 + (a1 + g1 * (e_prev < 0).float()) * e_prev * e_prev + b1 * s2
        e_prev = s2.sqrt() * eps[:, t]
        e[:, t] = e_prev
    lam = u(g, (n, 1), 0.002, 0.02, dev)
    N = torch.poisson(lam.expand(n, M).contiguous(), generator=g)
    J = u(g, (n, 1), -3.0, 1.0, dev) * N + u(g, (n, 1), 2.0, 6.0, dev) * N.sqrt() * torch.randn((n, M), generator=g, device=dev)
    r = u(g, (n, 1), -0.05, 0.05, dev) + e + J
    return r[:, BURN:], dict(alpha=a, gamma=gm, beta=b, nu=nu, lam=lam, jumps=N[:, BURN:])


def to_output(g, r, dev):
    n = r.shape[0]
    v = C.log_u(g, (n, 1), 0.003, 0.03, dev)
    price = bern(g, (n, 1), 0.6, dev)
    p = C.log_u(g, (n, 1), 1.0, 1000.0, dev) * torch.exp(torch.cumsum(v * r, 1).clamp(-20, 20))
    return torch.where(price, p, v * r), price


def make_garch(p):
    def garch_jump(n, L, device):
        g = _family.own_generator(device)
        dev = torch.device(device)
        if p["pool"]:
            r = _pool.crop(g, _pool.load(p["pool"], p["pool_dir"], p["manifest_dir"]), n, L, dev)[0][..., 0]
        else:
            r = garch_raw(g, n, L, dev)[0]
        return to_output(g, r, dev)[0]
    return garch_jump


def _install(p, inst):
    if inst.apply and p["pool"]:
        _pool.load(p["pool"], p["pool_dir"], p["manifest_dir"])
    _family.install(inst, "garch_jump", make_garch(p), p, allow_add=False)


register(Switch("garch_jump", "G10.1", "GJR-GARCH with compound-Poisson jumps; prices or returns", _install,
                dict(share=0.04, mode="add", pool="", pool_dir="", manifest_dir="")))
