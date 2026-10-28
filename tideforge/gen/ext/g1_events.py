"""G1.7 events -> traffic / footfall (switch sc_events, group kind through H5). Daily rows (dt = 24 h) / weekly sums of 7
(RES_DAILY): event flag and day-type = any / mean, event size and rain = sums.

  day type  weekday 1, weekend x [0.5, 1.5], holiday (HOLIDAYS, 4-10 days a year) x [0.3, 1.5]
  events    windows, 1 per [7, 60] days, 1-3 days long; magnitude m lognormal (median 0.3, clipped to [0.1, 3]),
            20% of events negative (x -0.8: closures); build-up m_next exp(-(start - t) / tau_pre), tau_pre in [0, 2] days;
            decay m_last exp(-(t - end) / tau_post), tau_post in [0, 3] days
  rain      relu(AR - 0.5), effect x (1 - [0, 0.2] 1[rain > 0])
  target    S (day type) (1 + effect) (rain) exp(sigma AR), sigma in [0.03, 0.15], S log-uniform in [0.3, 1e4];
            Poisson when S < 20
Covariates (priority): event flag known, event size known (m x lognormal 0.2 noise), day-type code known (0 weekday,
1 weekend, 2 holiday), rain known, events in another city (decoy flag), neighbour sensor past.
"""
from __future__ import annotations

import torch

from tideforge.gen.ext import g1_common as C
from tideforge.gen.ext._rand import u
from tideforge.gen.groups import ROLE_KNOWN, ROLE_PAST
from tideforge.gen.switches import Switch, register


def effect(g, n, L, dev):
    ev = C.windows(g, n, L, 1.0 / u(g, (n, 1), 7, 60, dev), 1, 3, dev)
    m = torch.exp(-1.2 + 0.8 * C.randn(g, (n, L), dev)).clamp(0.1, 3.0)
    m = torch.where(torch.rand((n, L), generator=g, device=dev) < 0.2, -0.8 * m.clamp(max=1.0), m)
    ml, mn = C.at(m, ev["last"]), C.at(m, ev["next"])
    tp, tq = u(g, (n, 1), 0, 2, dev), u(g, (n, 1), 0, 3, dev)
    pre = torch.where(ev["until"] > 0, mn * torch.exp(-ev["until"] / tp.clamp(min=1e-3)), torch.zeros_like(m))
    post = torch.where(torch.isfinite(ev["since_end"]), ml * torch.exp(-(ev["since_end"] + 1) / tq.clamp(min=1e-3)),
                       torch.zeros_like(m))
    eff = torch.where(ev["inside"], ml, pre + post)
    return eff.clamp(min=-0.9), ev, ml


def build(g, n, Lf, dt, dev, holidays="fixed", clock="fixed"):
    gr = C.grid(g, n, Lf, dt, dev, clock)
    hol = C.HOLIDAYS[holidays](g, n, gr, 4, 10, dev)
    we = gr["weekend"]
    dtype = torch.where(we, u(g, (n, 1), 0.5, 1.5, dev), torch.ones(n, 1, device=dev))
    dtype = torch.where(hol, u(g, (n, 1), 0.3, 1.5, dev), dtype)
    eff, ev, ml = effect(g, n, Lf, dev)
    rain = torch.relu(C.ar1(g, (n, Lf), 0.6, dev) - 0.5)
    rf = 1 - u(g, (n, 1), 0, 0.2, dev) * (rain > 0)
    S = C.log_u(g, (n, 1), 0.3, 1e4, dev)
    base = S * dtype * rf
    lam = base * (1 + eff) * torch.exp(u(g, (n, 1), 0.03, 0.15, dev) * C.ar1(g, (n, Lf), 0.5, dev))
    tgt = C.counts(g, lam, S < 20)
    size = ev["inside"] * ml * torch.exp(0.2 * C.randn(g, (n, Lf), dev))
    code = torch.where(hol, 2.0, torch.where(we, 1.0, 0.0))
    other = C.windows(g, n, Lf, 1.0 / u(g, (n, 1), 7, 60, dev), 1, 3, dev)["inside"].float()
    nb = lam * u(g, (n, 1), 0.3, 3, dev) * torch.exp(0.1 * C.randn(g, (n, Lf), dev))
    covs = torch.stack([ev["inside"].float(), size, code, rain, other, nb], 1)
    roles = torch.tensor([ROLE_KNOWN, ROLE_KNOWN, ROLE_KNOWN, ROLE_KNOWN, C.DECOY, ROLE_PAST], device=dev).expand(n, -1)
    info = dict(agg=("sum", "max", "sum", "mean", "sum", "max", "sum"), gr=gr, base=base, eff=eff, ev=ev, S=S)
    return tgt, covs, roles, info


def _install(p, inst):
    def b(g, n, Lf, dt, dev):
        return build(g, n, Lf, dt, dev, holidays=p["holidays"], clock=p["clock"])
    inst.add_group_kind("sc_events", C.scenario_kind(b, C.RES_DAILY, own=False), p["share"])


register(Switch("sc_events", "G1.7", "events -> traffic / footfall scenario (group kind)", _install,
                dict(share=0.15, holidays="fixed", clock="fixed")))
