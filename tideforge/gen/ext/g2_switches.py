"""G2 calendar engine switches (g2_engine), G3.2 long_cycle_cal, G3.4 season_short.

  cal_engine      (G2.1-G2.5) H1 override of v0's _L_calendar: v0's layer still runs (its draws are made), then its effect is
                  REPLACED on every row (owner 2026-10-28): the exposed rows (allow_add rows with probability `exposure`,
                  0.3 = v0's rate) get x + the engine's calendar effect, all other rows get x unchanged. Changes values
                  before the noise layer, so on CPU later poisson draws can shift (switches.py).
  cal_cov         (G2.2) group kind (H5): the target is the block's lane-0 v0 series with the engine's effects; in a share
                  `show` (0.5) of the groups the known covariate lanes hold the holiday-type code, weekend, bridge and payday
                  flags, in the rest those lanes are the block's own v0 series (decoys). Flags can only be shown for group
                  targets: generate_batch shuffles rows after H4, so a row's calendar cannot follow it into a block.
  long_cycle_cal  (G3.2) family replacing long_cycle's whole weight: daily (80%) or weekly (20%) rows on real dates: a
                  yearly profile by day of year (1-4 harmonics of doy / year length: leap years), a level per real weekday
                  (so the yearly pattern drifts one weekday a year), the engine's holiday effects, a trend and AR noise.
                  Passthrough (no later layer), like long_cycle. A new family changes the global draws (switches.py).
  season_short    (G3.4) H1 override of _L_seasonality: in a share (0.2) of allow_add rows the seasonal part is replaced by
                  one whose period gives 1.5-3 cycles in the window (v0: at least 3, 2 in ets / tsi), with v0's amplitude
                  (or U(0.3, 1.5) sd where v0 added none).
"""
from __future__ import annotations

import math

import torch

from tideforge.gen import profiles as _profiles
from tideforge.gen.ext import g1_common as C1
from tideforge.gen.ext import g2_engine as E
from tideforge.gen.ext import g2_table
from tideforge.gen.ext._rand import bern, u
from tideforge.gen.groups import ROLE_KNOWN
from tideforge.gen.switches import Switch, register

ENGINE = dict(E.OPTS, exposure=0.3)


def _opts(p):
    return {k: (tuple(p[k]) if isinstance(p[k], (list, tuple)) else p[k]) for k in E.OPTS}


def _install_engine(p, inst):
    opts = _opts(p)

    def override(args, kwargs, out, ctx):
        x, allow_add = args[0], args[1]
        g, dev = ctx.rng(), x.device
        gate = allow_add & bern(g, (x.shape[0],), p["exposure"], dev)
        res = x.clone()
        idx = gate.nonzero().squeeze(1)
        if idx.numel():
            res[idx] = E.apply(g, x[idx], opts, dev)[0]
        return res
    inst.override_layer("calendar", override)


def _install_cov(p, inst):
    opts = _opts(p)

    def kind(z, act, mu, sd, ctx):
        g, dev = ctx.rng(), z.device
        n, V, L = z.shape
        base = z[:, 0] * sd[:, 0] + mu[:, 0]
        y, info = E.apply(g, base, opts, dev)
        flags = torch.stack([info["code"], info["weekend"], info["bridge"], info["payday"]], 1)
        show = bern(g, (n, 1), p["show"], dev)
        k = min(4, V - 1)
        filler = z[:, 1:1 + k] * sd[:, 1:1 + k] + mu[:, 1:1 + k]
        covs = torch.where(show.unsqueeze(-1), flags[:, :k], filler)
        roles = torch.where(show, torch.full((n, k), ROLE_KNOWN, device=dev), torch.full((n, k), C1.DECOY, device=dev))
        return C1.assemble(g, z, act, mu, sd, y, covs, roles.long())
    inst.add_group_kind("cal_cov", kind, p["share"])


def long_cycle_cal(n, L, device):
    """G3.2 family: draws one seed from the global stream (as families draw from it), then uses its own generator."""
    dev = torch.device(device)
    g = torch.Generator(device=dev)
    g.manual_seed(int(torch.randint(0, 2 ** 62, (1,))))
    tab = g2_table.table(dev)
    weekly = bern(g, (n,), 0.2, dev)
    out = torch.zeros(n, L, device=dev)
    for wk_rows, p in ((False, 1.0), (True, 1.0 / 7.0)):
        idx = (weekly == wk_rows).nonzero().squeeze(1)
        if idx.numel() == 0:
            continue
        m, nd = idx.numel(), E._days_needed(p, L)
        dd = E.daily(g, m, nd, tab, E.OPTS, dev)
        days = dd["d0"].view(m, 1) + torch.arange(nd, device=dev).view(1, nd)
        D = E.dates(days)
        ang = 2 * math.pi * D["doy"].float() / D["ylen"].float()
        yr = torch.zeros(m, nd, device=dev)
        for k in range(1, 5):
            amp = u(g, (m, 1), 0, 1, dev) / k ** u(g, (m, 1), 0.5, 1.5, dev)
            yr = yr + amp * torch.cos(k * ang + u(g, (m, 1), 0, 2 * math.pi, dev))
        yr = yr / (yr.std(1, keepdim=True) + 1e-6)
        wdl = torch.gather(torch.randn(m, 7, generator=g, device=dev) * u(g, (m, 1), 0, 0.5, dev), 1, D["wd"])
        trend = u(g, (m, 1), -1, 1, dev) * torch.linspace(0, 1, nd, device=dev).view(1, nd)
        noise = C1.ar1(g, (m, nd), u(g, (m, 1), 0.3, 0.95, dev), dev) * u(g, (m, 1), 0.05, 0.4, dev)
        out[idx] = E._to_steps(yr + wdl + 0.5 * dd["E"] + trend + noise, p, L)
    return out


def _install_long(p, inst):
    if inst.profile is None:               # plan only (config hash): record "all of long_cycle" as weight 1.0
        inst.add_family("long_cycle_cal", long_cycle_cal, weight=1.0, allow_add=False, passthrough=True,
                        mode="replace:long_cycle")
        return
    have = _profiles.GEN_PROFILES[inst.profile]["weights"].get("long_cycle", 0.0)
    if have <= 0:
        raise ValueError("long_cycle_cal: the profile has no long_cycle weight to replace")
    inst.add_family("long_cycle_cal", long_cycle_cal, weight=have, allow_add=False, passthrough=True,
                    mode="replace:long_cycle")


def _install_short(p, inst):
    def override(args, kwargs, out, ctx):
        x, allow_add = args[0], args[1]
        g, dev = ctx.rng(), x.device
        B, L = x.shape
        pick = allow_add & bern(g, (B,), p["share"], dev)
        P = L / u(g, (B, 1), p["cycles_lo"], p["cycles_hi"], dev)
        t = torch.arange(L, device=dev, dtype=torch.float32).view(1, L)
        comp = torch.zeros(B, L, device=dev)
        for h in (1, 2):
            comp = comp + u(g, (B, 1), 0, 1, dev) / h * torch.sin(2 * math.pi * h * t / P + u(g, (B, 1), 0, 2 * math.pi, dev))
        comp = comp / (comp.std(1, keepdim=True) + 1e-6)
        sx = x.std(1, keepdim=True) + 1e-6
        amp = (out - x).std(1, keepdim=True)
        amp = torch.where(amp > 1e-6 * sx, amp, u(g, (B, 1), 0.3, 1.5, dev) * sx)
        return torch.where(pick.view(B, 1), x + amp * comp, out)
    inst.override_layer("seasonality", override)


register(Switch("cal_engine", "G2.1", "calendar engine replaces v0's calendar layer", _install_engine, ENGINE))
register(Switch("cal_cov", "G2.2", "calendar-flag covariate group kind", _install_cov, dict(E.OPTS, share=0.15, show=0.5)))
register(Switch("long_cycle_cal", "G3.2", "long_cycle on the calendar engine (family)", _install_long, {}))
register(Switch("season_short", "G3.4", "seasonality with 1.5-3 cycles in a share of rows", _install_short,
                dict(share=0.2, cycles_lo=1.5, cycles_hi=3.0)))
