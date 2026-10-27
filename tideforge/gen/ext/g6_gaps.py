"""G6: true missing values, written as NaN at H8 (final_blocks, priority "gaps"), plus G6.4 (no forward-filled runs).

Target lanes (role 0) and covariate lanes (past / known) are hit separately: a lane is hit with probability target_lanes or
cov_lanes. Inside a hit lane the gaps never touch the first point or the last `tail` points (owner 2026-10-28: tail 128; with
ft3's causal decoder the batcher crops anywhere in the block, so the tail is not the forecast region), and at least 30% of
the lane stays observed. T3's batcher turns every NaN into value 0 / mask 0 and masks it out of the loss (CLAUDE.md, T3).

  gaps_scattered  (G6.1)  each point of the allowed region missing with probability p
  gaps_block      (G6.2)  missing share f ~ U(f_lo, f_hi) in 1..max_blocks blocks: the allowed region is cut into equal
                          segments, one block of length f x segment at a random offset in each (no overlap, so the share is f)
  gaps_periodic   (G6.3)  period P from `periods`, off-duty d ~ U(d_lo, d_hi): missing where (t + phase) mod P < d x P
                          (nights, weekends, market closures)
  no_ffill        (G6.4)  profile miss_p -> 0: v0's corruption layer no longer forward-fills missing runs (its draws are
                          still made, so the global stream is unchanged)
"""
from __future__ import annotations

import torch

from tideforge.gen.ext._rand import bern, randint, u
from tideforge.gen.groups import ROLE_TARGET
from tideforge.gen.switches import Switch, register

MAX_MISSING = 0.7
LANES = dict(target_lanes=0.3, cov_lanes=0.3, tail=128)


def hit_lanes(role: torch.Tensor, p: dict, g: torch.Generator) -> torch.Tensor:
    r = torch.rand(role.shape, generator=g, device=role.device)
    return torch.where(role == ROLE_TARGET, r < p["target_lanes"], r < p["cov_lanes"])


def allowed(L: int, tail: int, device) -> torch.Tensor:
    t = torch.arange(L, device=device)
    return (t >= 1) & (t < max(2, L - tail))


def scattered(shape, p: float, g, device) -> torch.Tensor:
    return torch.rand(shape, generator=g, device=device) < p


def blocks(shape, f_lo: float, f_hi: float, max_blocks: int, g, device, region: torch.Tensor):
    """(N, V, L) missing mask: per lane f ~ U, nb ~ {1..max_blocks}, one block per equal segment of the region."""
    N, V, L = shape
    idx = region.nonzero().squeeze(1)
    a0, A = int(idx[0]), int(idx.numel())
    f = u(g, (N, V, 1), f_lo, f_hi, device)
    nb = randint(g, 1, max_blocks, (N, V, 1), device)
    seg = A / nb
    k = torch.arange(L, device=device).view(1, 1, L) - a0                 # position inside the region
    j = torch.floor(k / seg).clamp(max=max_blocks - 1)                    # segment of each point
    blen = torch.floor(f * seg)
    off = torch.floor(torch.rand((N, V, max_blocks), generator=g, device=device) * (seg - blen + 1).clamp(min=1))
    o = off.gather(2, j.long().clamp(min=0).expand(N, V, L))
    pos = k - j * seg
    return (pos >= o) & (pos < o + blen) & region.view(1, 1, L)


def periodic(shape, periods, d_lo: float, d_hi: float, g, device) -> torch.Tensor:
    N, V, L = shape
    P = torch.as_tensor(periods, device=device, dtype=torch.float32)[randint(g, 0, len(periods) - 1, (N, V, 1), device)]
    d = u(g, (N, V, 1), d_lo, d_hi, device)
    phase = torch.floor(torch.rand((N, V, 1), generator=g, device=device) * P)
    t = torch.arange(L, device=device, dtype=torch.float32).view(1, 1, L)
    return torch.remainder(t + phase, P) < d * P


def _apply(x, role, miss, p, g):
    region = allowed(x.shape[-1], int(p["tail"]), x.device)
    miss = miss & region.view(1, 1, -1) & hit_lanes(role, p, g).unsqueeze(-1)
    frac = miss.float().sum(-1, keepdim=True) / region.sum().clamp(min=1)
    miss = miss & (frac <= MAX_MISSING)                                   # keep >= 30% of the lane observed
    return x.masked_fill(miss, float("nan"))


def _install_scattered(p, inst):
    def hook(x, role, lgid, ctx):
        g = ctx.rng()
        return _apply(x, role, scattered(x.shape, p["p"], g, x.device), p, g), role, lgid
    inst.add_hook("final_blocks", hook, "gaps")


def _install_block(p, inst):
    def hook(x, role, lgid, ctx):
        g = ctx.rng()
        region = allowed(x.shape[-1], int(p["tail"]), x.device)
        m = blocks(x.shape, p["f_lo"], p["f_hi"], int(p["max_blocks"]), g, x.device, region)
        return _apply(x, role, m, p, g), role, lgid
    inst.add_hook("final_blocks", hook, "gaps")


def _install_periodic(p, inst):
    def hook(x, role, lgid, ctx):
        g = ctx.rng()
        return _apply(x, role, periodic(x.shape, p["periods"], p["d_lo"], p["d_hi"], g, x.device), p, g), role, lgid
    inst.add_hook("final_blocks", hook, "gaps")


def _install_no_ffill(p, inst):
    inst.set_profile("miss_p", lambda _: 0.0)


register(Switch("gaps_scattered", "G6.1", "scattered NaN at H8", _install_scattered, dict(LANES, p=0.05)))
register(Switch("gaps_block", "G6.2", "block NaN at H8", _install_block, dict(LANES, f_lo=0.05, f_hi=0.5, max_blocks=4)))
register(Switch("gaps_periodic", "G6.3", "periodic NaN at H8", _install_periodic,
                dict(LANES, periods=(12, 24, 168), d_lo=0.15, d_hi=0.4)))
register(Switch("no_ffill", "G6.4", "no forward-filled missing runs (miss_p 0)", _install_no_ffill, {}))
