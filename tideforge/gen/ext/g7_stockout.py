"""G7.3 stockout censoring (switch stockout): H8 hook on count lanes. Probe P7.

A lane is a count lane when all its values are finite whole numbers >= 0 with mean >= 0.5 (rows are shuffled after H4, so the
family flags cannot reach the blocks; v0's count rows qualify, as do G7.1's). A share (0.3) of them gets stockouts:
  runs      windows at a rate 1 / U(50, 300) per step, 1-14 steps long: the value is 0 (no stock)
  rebound   the next R steps after each run, R in 1..5, are x (1 + U(0.2, 1.0)) rounded (pent-up demand); lanes stay integer
  unobserved  (param, default False) the stockout zeros are written as NaN instead (allowed at H8 only), for G6-style
            masking later; then the lane is no longer all-integer.
With G8.1 on as well, a stockout in a child breaks parent = sum of children (Phase 3 only; not combined here).
"""
from __future__ import annotations

import torch

from tideforge.gen.ext import g1_common as C
from tideforge.gen.ext._rand import bern, randint, u
from tideforge.gen.switches import Switch, register


def count_lanes(x):
    fin = torch.isfinite(x).all(-1)
    xs = torch.nan_to_num(x)
    whole = (xs == torch.round(xs)).all(-1) & (xs >= 0).all(-1)
    return fin & whole & (xs.mean(-1) >= 0.5)


def apply(g, x, p, dev):
    """x (n, L) count lanes -> (x with stockouts, stockout mask, rebound mask)."""
    n, L = x.shape
    w = C.windows(g, n, L, 1.0 / u(g, (n, 1), p["rate_lo"], p["rate_hi"], dev), 1, 14, dev)
    out = w["inside"]
    R = randint(g, 1, 5, (n, 1), dev).float()
    reb = (~out) & (w["since_end"] < R)
    y = torch.where(reb, torch.round(x * (1 + u(g, (n, 1), 0.2, 1.0, dev))), x)
    y = torch.where(out, torch.full_like(y, float("nan") if p["unobserved"] else 0.0), y)
    return y, out, reb


def _install(p, inst):
    def hook(x, role, lgid, ctx):
        g, dev = ctx.rng(), x.device
        N, V, L = x.shape
        pick = count_lanes(x) & bern(g, (N, V), p["share"], dev)
        idx = pick.view(-1).nonzero().squeeze(1)
        if idx.numel() == 0:
            return x, role, lgid
        flat = x.reshape(N * V, L).clone()
        flat[idx] = apply(g, flat[idx], p, dev)[0]
        return flat.view(N, V, L), role, lgid
    inst.add_hook("final_blocks", hook, "observation")


register(Switch("stockout", "G7.3", "stockout runs of zeros then a rebound, on count lanes (H8)", _install,
                dict(share=0.3, rate_lo=50.0, rate_hi=300.0, unobserved=False)))
