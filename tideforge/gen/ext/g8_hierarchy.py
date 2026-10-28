"""G8.1 hierarchy (switch hierarchy): a group kind (H5) and its own group-size rule (H6, g in 3..V). Probe P8.

Lanes [0, g) of the group, every lane a target:
  children  leaves c_i = w_i exp(sigma z_i), z_i the lane's own standardised v0 series, sigma in [0.2, 0.8], w_i log-uniform
            in [0.1, 10] (one dominant child x 10 in 30% of groups); 30% of leaves become counts: Poisson(c_i m_i / mean c_i),
            m_i log-uniform in [0.1, 3] (zeros at low m_i)
  levels    one level: lane 0 = parent, lanes 1..g-1 = leaves. Two levels (30% of groups with g >= 7): lane 0 = parent,
            lanes 1, 2 = middle nodes, leaves 3..g-1 split between them (at least 2 each); each node = sum of its leaves
  parent    exact sums, added in float64 then cast (exact when the leaves are integers); smoother than its children by
            construction (independent parts average out)
"""
from __future__ import annotations

import torch

from tideforge.gen.ext import g1_common as C
from tideforge.gen.ext._rand import bern, randint, u
from tideforge.gen.groups import ROLE_TARGET
from tideforge.gen.switches import Switch, register


def build(g, z, gsz, dev, p):
    """z (n, V, L) standardised, gsz (n,) group sizes -> y (n, V, L) (lanes >= g: 0) and info."""
    n, V, L = z.shape
    lane = torch.arange(V, device=dev).view(1, V)
    two = (gsz.view(n, 1) >= 7) & bern(g, (n, 1), p["two_level_p"], dev)
    first = torch.where(two, 3, 1)                                         # first leaf lane
    leaf = (lane >= first) & (lane < gsz.view(n, 1))
    w = C.log_u(g, (n, V), 0.1, 10.0, dev)
    dom = randint(g, 0, V - 1, (n, 1), dev)
    w = torch.where(bern(g, (n, 1), 0.3, dev) & (lane == dom), 10 * w, w)
    c = w.unsqueeze(-1) * torch.exp(u(g, (n, 1, 1), 0.2, 0.8, dev) * z)
    cnt = leaf & bern(g, (n, V), p["count_p"], dev)
    m = C.log_u(g, (n, V, 1), 0.1, 3.0, dev)
    lam = c * m / (c.mean(-1, keepdim=True) + 1e-9)
    c = torch.where(cnt.unsqueeze(-1), torch.poisson(lam, generator=g), c)
    c = torch.where(leaf.unsqueeze(-1), c, torch.zeros_like(c)).double()
    nleaf = (gsz.view(n, 1) - first)
    split = first + 2 + torch.floor(u(g, (n, 1), 0, 1, dev) * (nleaf - 3).clamp(min=0).double()).long()
    to1 = leaf & (lane < split)
    to2 = leaf & (lane >= split)
    mid1 = (c * to1.unsqueeze(-1)).sum(1)
    mid2 = (c * to2.unsqueeze(-1)).sum(1)
    y = c.clone()
    y[:, 0] = c.sum(1)
    y[:, 1] = torch.where(two, mid1, y[:, 1])
    y[:, 2] = torch.where(two, mid2, y[:, 2])
    return y.float(), dict(two=two, leaf=leaf, cnt=cnt, to1=to1, to2=to2, split=split)


def _install(p, inst):
    def kind(z, act, mu, sd, ctx):
        g, dev = ctx.rng(), z.device
        gsz = act.sum(1)
        y, _ = build(g, z, gsz, dev, p)
        y = torch.where(act.unsqueeze(-1), y, z * sd + mu)
        return y, torch.full(act.shape, ROLE_TARGET, dtype=torch.long, device=dev)

    def size(gs, kind_, is_group, V, ctx):
        if V < 3:
            return gs
        return randint(ctx.rng(), 3, V, gs.shape, gs.device)
    inst.add_group_kind("hierarchy", kind, p["share"])
    inst.set_group_size(size, ("hierarchy",))


register(Switch("hierarchy", "G8.1", "hierarchy group kind: parent = exact sum of children, all lanes targets", _install,
                dict(share=0.15, two_level_p=0.3, count_p=0.3)))
