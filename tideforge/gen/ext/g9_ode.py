"""G9.1 ode family (switch ode): crops of the offline ODE pool (tools/build_ode_pool.py; _pool.py checks its sha against the
committed manifest and refuses to run otherwise). Probe P9.

Per row: a random trajectory (systems in equal shares), every s-th step, s in {1, 2, 4} with s L <= T, a random crop; then
  observed  one channel (70%) in its natural units (forced_sir: incidence I = exp(log I)), or a linear mix of the system's
            standardised channels (30%), weights N(0, 1)
  scale     x U(log-uniform 0.1, 100); an offset N(0, 1) sd for the mixes and the signed systems (van der Pol, Duffing,
            Lorenz); populations (Lotka-Volterra, SIR) keep their zero floor
  noise     sd 0.5-5%: of the row's sd (signed rows), multiplicative exp(N) for the populations (they stay > 0)
Flags: no additive layers (allow_add False); the noise / support / corruption / scale layers apply.
"""
from __future__ import annotations

import torch

from tideforge.gen.ext import _family, _pool
from tideforge.gen.ext import g1_common as C
from tideforge.gen.ext._rand import bern, u
from tideforge.gen.switches import Switch, register

POSITIVE = (0, 1)                           # lotka_volterra, forced_sir


def ode_rows(g, pool, n, L, dev):
    x, idx, s = _pool.crop(g, pool, n, L, dev, stride_choices=(1, 2, 4))
    meta = torch.as_tensor(pool["meta"], device=dev)[idx]
    sid, nch = meta[:, 0].long(), meta[:, 1].long()
    mean, sd = meta[:, 3:6], meta[:, 6:9]
    ch = torch.floor(torch.rand(n, generator=g, device=dev) * nch).long().clamp(max=2)
    ar = torch.arange(n, device=dev)
    single = x[ar, :, ch] * sd[ar, ch].view(n, 1) + mean[ar, ch].view(n, 1)
    single = torch.where(((sid == 1) & (ch == 1)).view(n, 1), torch.exp(single), single)
    used = (torch.arange(3, device=dev).view(1, 3) < nch.view(n, 1)).float()
    mix = (x * (torch.randn((n, 1, 3), generator=g, device=dev) * used.view(n, 1, 3))).sum(-1)
    is_mix = bern(g, (n, 1), 0.3, dev)
    y = torch.where(is_mix, mix, single)
    positive = torch.isin(sid, torch.tensor(POSITIVE, device=dev)).view(n, 1)
    signed = is_mix | ~positive
    ysd = y.std(1, keepdim=True) + 1e-6
    y = y + signed.float() * torch.randn((n, 1), generator=g, device=dev) * ysd
    y = y * C.log_u(g, (n, 1), 0.1, 100.0, dev)
    e = torch.randn((n, L), generator=g, device=dev) * u(g, (n, 1), 0.005, 0.05, dev)
    y = torch.where(signed, y + e * (y.std(1, keepdim=True) + 1e-6), y * torch.exp(e))
    return y, dict(sid=sid, ch=ch, is_mix=is_mix, stride=s, idx=idx)


def make_ode(p):
    def ode(n, L, device):
        g = _family.own_generator(device)
        pool = _pool.load(p["pool"], p["pool_dir"], p["manifest_dir"])
        return ode_rows(g, pool, n, L, torch.device(device))[0]
    return ode


def _install(p, inst):
    if inst.apply:
        _pool.load(p["pool"], p["pool_dir"], p["manifest_dir"])           # refuse at configure() if the sha differs
    _family.install(inst, "ode", make_ode(p), p, allow_add=False)


register(Switch("ode", "G9.1", "ODE pool crops (Lotka-Volterra, forced SIR, van der Pol, Duffing, Lorenz)", _install,
                dict(share=0.05, mode="add", pool="ode_v1", pool_dir="", manifest_dir="")))
