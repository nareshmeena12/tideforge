"""E0.8 negative control (switch white_noise): a family of i.i.d. N(0, 1) rows added at a share of the rows (default 10%; add
mode, the other families renormalised, which takes the share from every family in proportion). Passthrough (no later
layer), so the rows stay pure white noise. Measures what dilution alone costs at 5M (EXPERIMENTS.md E0.8)."""
from __future__ import annotations

import torch

from tideforge.gen.ext import _family
from tideforge.gen.switches import Switch, register


def white_noise(n, L, device):
    g = _family.own_generator(device)
    return torch.randn((n, L), generator=g, device=torch.device(device))


def _install(p, inst):
    _family.install(inst, "white_noise", white_noise, p, allow_add=False, passthrough=True)


register(Switch("white_noise", "E0.8", "negative control: i.i.d. N(0, 1) rows at a share of the mix", _install,
                dict(share=0.10, mode="add")))
