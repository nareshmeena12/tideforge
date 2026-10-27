"""G7.4: a larger share of integer-valued series.

  int_share  H4 (final, after scale): non-count rows are picked so that `share` of all rows are counts (rows that are counts
             already count towards it), and turned into counts that keep the series' shape (owner 2026-10-28): the intensity
             is a positive transform of the standardised series, lam = exp(a z + b) with a ~ U(0.3, 1.0), b ~ U(-1, 3)
             (the support layer's ranges), times a log-normal overdispersion factor for half the rows (negative-binomial
             like); counts ~ Poisson(lam) from the switch's own generator. is_count and is_nonneg are set on those rows.
"""
from __future__ import annotations

import torch

from tideforge.gen.ext._rand import bern, u
from tideforge.gen.switches import Switch, register

EXP_CLAMP = 20.0


def intensity(x: torch.Tensor, g: torch.Generator) -> torch.Tensor:
    B, _ = x.shape
    dev = x.device
    z = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
    a, b = u(g, (B, 1), 0.3, 1.0, dev), u(g, (B, 1), -1.0, 3.0, dev)
    od = torch.where(bern(g, (B, 1), 0.5, dev), torch.exp(torch.randn((B, 1), generator=g, device=dev) * 0.5), torch.ones(B, 1, device=dev))
    return torch.exp((a * z + b).clamp(-EXP_CLAMP, EXP_CLAMP)) * od


def to_counts(x: torch.Tensor, g: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    lam = intensity(x, g)
    return torch.poisson(lam, generator=g), lam


def _install(p, inst):
    def hook(x, ctx):
        g, dev = ctx.rng(), x.device
        B = x.shape[0]
        cnt = ctx.is_count.clone()
        have = float(cnt.float().mean())
        q = max(0.0, (p["share"] - have) / max(1e-9, 1.0 - have))
        pick = bern(g, (B,), q, dev) & ~cnt
        c, _ = to_counts(x, g)
        ctx.is_count = cnt | pick
        ctx.is_nonneg = ctx.is_nonneg | pick
        return torch.where(pick.unsqueeze(-1), c, x)

    inst.add_hook("final", hook, "observation")


register(Switch("int_share", "G7.4", "integer-valued share of the rows (counts that keep the series' shape)", _install,
                dict(share=0.10)))
