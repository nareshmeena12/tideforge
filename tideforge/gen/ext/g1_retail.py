"""G1.5 retail (switch sc_retail, group kind through H5). Daily rows built at dt = 24 h; weekly rows are sums of 7 days
(RES_DAILY), with promo = share of promo days, open = open days, price = mean, holiday = any.

  weekly   exp(sum_{k=1,2} a_k cos(2 pi k dow / 7 + ph_k)), a_k in [0.05, 0.4];  yearly exp(A cos(angle - phi)), A in [0, 0.3]
  holiday  plain flag from HOLIDAYS[holidays] (default 6-12 fixed days of the year; G2 plugs in real calendars); effect
           1 + e_h, e_h in [-0.5, 1.0]; the day before 1 + [0, 0.5]
  open     closed on one fixed weekday (50% of stores), on holidays with probability q in [0, 1], and in rare closures
           (1 per 400 days, 3-20 days); closed -> sales exactly 0
  promo    windows, 1 per [14, 60] days, 3-14 days long, discount d in [0.1, 0.5], display uplift 1 + [0, 0.5];
           post-promotion dip 1 - delta exp(-s / tau), delta in [0, 0.4], tau in [2, 10] days, s = days since the end
  price    p (1 - d promo), p with rare steps (1 per 180 days, x exp(0.05 N)); demand x (price / p)^eps, eps in [-3, -0.5]
  sales    lam = S weekly yearly (price effect) uplift dip holiday open exp(sigma AR), sigma in [0.05, 0.2],
           S log-uniform in [0.3, 500]; Poisson(lam) when S < 20 (low sellers), lam otherwise
Covariates (priority): promo known, open known, price known, holiday known, customers past (lam / basket x noise),
competitor promo (decoy).
"""
from __future__ import annotations

import math

import torch

from tideforge.gen.ext import g1_common as C
from tideforge.gen.ext._rand import bern, u
from tideforge.gen.groups import ROLE_KNOWN, ROLE_PAST
from tideforge.gen.switches import Switch, register

HOLIDAY_SOURCE = "fixed"


def weekly(g, n, gr, dev, lo=0.05, hi=0.4):
    w = torch.zeros_like(gr["dow"])
    for k in (1, 2):
        w = w + u(g, (n, 1), lo, hi, dev) * torch.cos(2 * math.pi * k * gr["dow"] / 7 + u(g, (n, 1), 0, 2 * math.pi, dev))
    return torch.exp(w)


def promos(g, n, L, dev):
    return C.windows(g, n, L, 1.0 / u(g, (n, 1), 14, 60, dev), 3, 14, dev)


def build(g, n, Lf, dt, dev, holidays=None):
    gr = C.grid(g, n, Lf, dt, dev)
    hol = C.HOLIDAYS[holidays or HOLIDAY_SOURCE](g, n, gr, 6, 12, dev)
    pre = torch.cat([hol[:, 1:], torch.zeros_like(hol[:, :1])], 1) & ~hol
    yearly = torch.exp(u(g, (n, 1), 0, 0.3, dev) * torch.cos(gr["ang"] - u(g, (n, 1), 0, 2 * math.pi, dev)))
    cd = torch.floor(u(g, (n, 1), 0, 7, dev))
    closed = bern(g, (n, 1), 0.5, dev) & (gr["dow"] == cd)
    closed = closed | (hol & (torch.rand(hol.shape, generator=g, device=dev) < u(g, (n, 1), 0, 1, dev)))
    closed = closed | C.windows(g, n, Lf, torch.full((n, 1), 1 / 400, device=dev), 3, 20, dev)["inside"]
    opn = (~closed).float()
    pw = promos(g, n, Lf, dev)
    promo = pw["inside"].float()
    d = u(g, (n, 1), 0.1, 0.5, dev)
    dip = 1 - u(g, (n, 1), 0, 0.4, dev) * torch.exp(-pw["since_end"] / u(g, (n, 1), 2, 10, dev)) * (1 - promo)
    steps = torch.rand((n, Lf), generator=g, device=dev) < 1 / 180
    pbar = torch.exp(torch.cumsum(steps.float() * 0.05 * C.randn(g, (n, Lf), dev), 1))
    rel = 1 - d * promo
    eps = u(g, (n, 1), -3, -0.5, dev)
    eff = rel ** eps * (1 + u(g, (n, 1), 0, 0.5, dev) * promo) * dip
    eff = eff * (1 + u(g, (n, 1), -0.5, 1.0, dev) * hol) * (1 + u(g, (n, 1), 0, 0.5, dev) * pre)
    S = C.log_u(g, (n, 1), 0.3, 500, dev)
    base = S * weekly(g, n, gr, dev) * yearly
    lam = base * eff * opn * torch.exp(u(g, (n, 1), 0.05, 0.2, dev) * C.ar1(g, (n, Lf), 0.5, dev))
    low = S < 20
    sales = C.counts(g, lam, low)
    cust = lam / u(g, (n, 1), 1.5, 4, dev) * torch.exp(0.05 * C.randn(g, (n, Lf), dev))
    comp = promos(g, n, Lf, dev)["inside"].float()
    covs = torch.stack([promo, opn, pbar * rel * u(g, (n, 1), 1, 50, dev), hol.float(), cust, comp], 1)
    roles = torch.tensor([ROLE_KNOWN, ROLE_KNOWN, ROLE_KNOWN, ROLE_KNOWN, ROLE_PAST, C.DECOY], device=dev).expand(n, -1)
    info = dict(agg=("sum", "mean", "sum", "mean", "max", "sum", "mean"), gr=gr, base=base, promo=pw, opn=opn, low=low,
                hol=hol, eff=eff, S=S)
    return sales, covs, roles, info


def _install(p, inst):
    def b(g, n, Lf, dt, dev):
        return build(g, n, Lf, dt, dev, holidays=p["holidays"])
    inst.add_group_kind("sc_retail", C.scenario_kind(b, C.RES_DAILY, own=False), p["share"])


register(Switch("sc_retail", "G1.5", "retail scenario (group kind)", _install, dict(share=0.15, holidays="fixed")))
