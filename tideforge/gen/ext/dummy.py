"""Test switches (experiment "TEST"): they exercise the registry and the hook points and back configs/exp/EXAMPLE.yaml. Never part
of a real experiment.

  test_post    shift a random share p of rows (blocks at final_blocks) by `shift`; draws only from ctx.rng()
  test_gaps    NaN at a random share p of the points of every lane, at final_blocks (priority "gaps")
  test_family  a sine family (period 24, random phase), mode "add" or "replace:<family>"
"""
from __future__ import annotations

import math

import torch

from tideforge.gen.switches import Switch, register


def _install_post(p, inst):
    def batch_hook(x, ctx):
        hit = torch.rand(x.shape[0], generator=ctx.rng(), device=x.device) < p["p"]
        return x + p["shift"] * hit.unsqueeze(-1).to(x.dtype)

    def block_hook(x, role, lgid, ctx):
        hit = torch.rand(x.shape[0], generator=ctx.rng(), device=x.device) < p["p"]
        return x + p["shift"] * hit.view(-1, 1, 1).to(x.dtype), role, lgid

    point = p["point"]
    inst.add_hook(point, block_hook if point in ("after_groups", "final_blocks") else batch_hook, p["priority"])


def _install_gaps(p, inst):
    def gaps(x, role, lgid, ctx):
        miss = torch.rand(x.shape, generator=ctx.rng(), device=x.device) < p["p"]
        return x.masked_fill(miss, float("nan")), role, lgid

    inst.add_hook("final_blocks", gaps, "gaps")


def _sine(n, L, device):
    phase = torch.rand(n, 1, device=device) * 2 * math.pi
    t = torch.arange(L, device=device, dtype=torch.float32)
    return torch.sin(2 * math.pi * t / 24 + phase)


def _install_family(p, inst):
    inst.add_family("test_sine", _sine, weight=p["weight"], mode=p["mode"])


register(Switch("test_post", "TEST", "shift a random share of rows / blocks (post-processing, own RNG)", _install_post,
                dict(point="final_blocks", p=0.25, shift=1.0, priority="observation")))
register(Switch("test_gaps", "TEST", "NaN at a random share of points, at final_blocks", _install_gaps, dict(p=0.05)))
register(Switch("test_family", "TEST", "a sine family", _install_family, dict(weight=0.05, mode="add")))
