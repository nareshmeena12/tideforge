"""G1.6 PV output from irradiance (switch sc_pv, group kind through H5). Resolutions RES_PV (15 min / 30 min / hourly).

  G      clear-sky irradiance (g1_common.clear_sky): latitude in [0, 60] deg, declination from the day of year; 0 at night
  c      cloud factor (g1_common.cloud); I = G c
  Tcell  Tair + I / 800 x [20, 30]                                   (Tair: g1_common.temperature)
  P      I / 1000 (1 - gamma (Tcell - 25)) (1 + 0.02 AR), gamma in [0.003, 0.005], >= 0; clipped at an inverter limit
         clip = [0.6, 0.95] x the 99th percentile of P for 40% of the groups (none for the rest); exactly 0 where G = 0;
         then x S (S log-uniform in [0.1, 1e4], no offset: zeros matter)
Covariates (priority): irradiance I known, Tair known, clear-sky G known, cloud cover 1 - c known, neighbour plant past (its own
cloud factor, correlated with c), wind speed (decoy).
"""
from __future__ import annotations

import torch

from tideforge.gen.ext import g1_common as C
from tideforge.gen.ext import g1_load
from tideforge.gen.ext._rand import bern, u
from tideforge.gen.groups import ROLE_KNOWN, ROLE_PAST
from tideforge.gen.switches import Switch, register


def output(g, n, I, Tair, Gc, dev):
    Tcell = Tair + I / 800.0 * u(g, (n, 1), 20, 30, dev)
    P = I / 1000.0 * (1 - u(g, (n, 1), 0.003, 0.005, dev) * (Tcell - 25))
    return (P * (1 + 0.02 * C.ar1(g, I.shape, 0.8, dev))).clamp(min=0.0) * (Gc > 0)


def build(g, n, Lf, dt, dev, clock="fixed"):
    gr = C.grid(g, n, Lf, dt, dev, clock)
    Gc = C.clear_sky(gr, torch.deg2rad(u(g, (n, 1), 0, 60, dev)))
    c = C.cloud(g, n, gr, dev)
    I = Gc * c
    Tair = C.temperature(g, n, gr, dev)
    P = output(g, n, I, Tair, Gc, dev)
    clips = bern(g, (n, 1), 0.4, dev)
    clip = torch.where(clips, u(g, (n, 1), 0.6, 0.95, dev) * C.quantile(P, torch.full((n, 1), 0.99, device=dev)),
                       torch.full((n, 1), float("inf"), device=dev))
    P = torch.minimum(P, clip)
    c2 = (c + 0.1 * C.ar1(g, c.shape, 0.9 ** dt, dev)).clamp(0.05, 1.0)
    nb = output(g, n, Gc * c2, Tair, Gc, dev)
    S = C.log_u(g, (n, 1), 0.1, 1e4, dev)
    covs = torch.stack([I, Tair, Gc, 1 - c, C.rescale(g, nb, dev), g1_load.wind_speed(g, n, gr, dev)], 1)
    roles = torch.tensor([ROLE_KNOWN, ROLE_KNOWN, ROLE_KNOWN, ROLE_KNOWN, ROLE_PAST, C.DECOY], device=dev).expand(n, -1)
    info = dict(agg=("mean",) * 7, gr=gr, G=Gc, clip=clip, clips=clips, S=S)
    return S * P, covs, roles, info


def _install(p, inst):
    def b(g, n, Lf, dt, dev):
        return build(g, n, Lf, dt, dev, clock=p["clock"])
    inst.add_group_kind("sc_pv", C.scenario_kind(b, C.RES_PV, own=False), p["share"])


register(Switch("sc_pv", "G1.6", "PV output from irradiance scenario (group kind)", _install, dict(share=0.15, clock="fixed")))
