"""G1.9 causal-graph covariates (switch causal_graph, group kind through H5) and G1.11 covariate count (switch cov_count, H6).

causal_graph: K in {3..min(8, V)} nodes in a random order; edge i -> j (i < j) with probability p_e in [0.3, 0.7].
  roots    the block's own v0 series z_j (standardised)
  node j   standardise( sum_i w_ij f_ij(x_i(t - lag_ij)) + a_j z_j + sigma_j AR(0.5) ),  a_j in [0, 0.5], sigma_j in [0.1, 0.5]
  lag      0 with probability 0.4, else uniform in 1 .. floor(0.1 L)
  f        linear | tanh(c x), c in [0.5, 2] | threshold 1[x > c], c in [-0.5, 0.5] | multiplicative x sigmoid(z_j) |
           saturating x / (1 + |x|)
  w        +-[0.2, 1.5]; 30% of edges weak, +-[0.02, 0.1]
  target   a node with >= 1 parent, sinks preferred (if the graph has none, the edge 0 -> K-1 is added)
  roles    parents: known or past with probability 0.5 each, independent of the lag (owner 2026-10-28); descendants of the
           target: always past (their future would reveal the target's); siblings, other ancestors, unrelated nodes and the
           unused lanes (nodes >= K are the block's own v0 series): decoys, known or past at random
  lanes    target in lane 0, then parents, then the rest (random order within the group's g - 1 slots); in the lane's v0 scale
cov_count: in known_cov, past_cov and every G1 scenario / graph kind, the number of covariates k in 1..7 is drawn with
  P = (0.35, 0.25, 0.15, 0.10, 0.07, 0.05, 0.03), g = 1 + k (capped at V); other kinds keep v0's sizes. More than 7 needs
  more lanes (a model change; EXPERIMENTS.md G1.11).
"""
from __future__ import annotations

import torch

from tideforge.gen import _hooks
from tideforge.gen import groups as _groups
from tideforge.gen.ext import g1_common as C
from tideforge.gen.ext._rand import bern, randint, u
from tideforge.gen.groups import ROLE_KNOWN, ROLE_PAST
from tideforge.gen.switches import Switch, register

COUNT_P = (0.35, 0.25, 0.15, 0.10, 0.07, 0.05, 0.03)


def _std(x):
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-6)


def _shift(x, lag):
    """x (n, L) delayed by lag (n,) steps, the first value repeated."""
    L = x.shape[1]
    idx = (torch.arange(L, device=x.device).view(1, L) - lag.view(-1, 1)).clamp(min=0)
    return torch.gather(x, 1, idx)


def graph(g, z):
    """z (n, V, L) standardised v0 series -> nodes X (n, K, L) standardised, and the graph (dict of tensors)."""
    n, V, L = z.shape
    dev = z.device
    K = min(8, V)
    k = randint(g, 3, K, (n, 1), dev)
    i_, j_ = torch.arange(K, device=dev).view(1, K, 1), torch.arange(K, device=dev).view(1, 1, K)
    valid = (i_ < j_) & (j_ < k.view(n, 1, 1))
    A = (torch.rand((n, K, K), generator=g, device=dev) < u(g, (n, 1, 1), 0.3, 0.7, dev)) & valid
    none = ~A.any(1).any(1)
    force = torch.zeros_like(A)
    force[torch.arange(n, device=dev), 0, (k.view(n) - 1)] = True
    A = A | (force & none.view(n, 1, 1))
    lag = torch.where(bern(g, (n, K, K), 0.4, dev), torch.zeros(n, K, K, dtype=torch.long, device=dev),
                      randint(g, 1, max(1, int(0.1 * L)), (n, K, K), dev))
    form = randint(g, 0, 4, (n, K, K), dev)
    weak = bern(g, (n, K, K), 0.3, dev)
    sgn = torch.where(bern(g, (n, K, K), 0.5, dev), 1.0, -1.0)
    w = sgn * torch.where(weak, u(g, (n, K, K), 0.02, 0.1, dev), u(g, (n, K, K), 0.2, 1.5, dev))
    c_tanh, c_thr = u(g, (n, K, K), 0.5, 2.0, dev), u(g, (n, K, K), -0.5, 0.5, dev)
    a, sig = u(g, (n, K), 0.0, 0.5, dev), u(g, (n, K), 0.1, 0.5, dev)
    X = []
    for j in range(K):
        zj = _std(z[:, j])
        acc = a[:, j:j + 1] * zj + sig[:, j:j + 1] * C.ar1(g, (n, L), 0.5, dev)
        for i in range(j):
            xi = _shift(X[i], lag[:, i, j])
            f = form[:, i, j].view(n, 1)
            ct, cr = c_tanh[:, i, j].view(n, 1), c_thr[:, i, j].view(n, 1)
            fx = torch.where(f == 0, xi, torch.where(f == 1, torch.tanh(ct * xi), torch.where(
                f == 2, (xi > cr).float(), torch.where(f == 3, xi * torch.sigmoid(zj), xi / (1 + xi.abs())))))
            acc = acc + (A[:, i, j].view(n, 1) * w[:, i, j].view(n, 1)) * _std(fx)
        root = ~A[:, :j, j].any(1).view(n, 1) if j else torch.ones(n, 1, dtype=torch.bool, device=dev)
        X.append(torch.where(root, zj, _std(acc)))
    X = torch.stack(X, 1)
    has_par = A.any(1)
    sink = ~A.any(2)
    score = has_par.float() * (1 + sink.float()) + 0.5 * torch.rand((n, K), generator=g, device=dev)
    tgt = score.argmax(1)
    D = A.clone()
    for _ in range(K):
        D = D | (torch.bmm(D.float(), A.float()) > 0)
    ar = torch.arange(n, device=dev)
    parent = A[ar, :, tgt]
    desc = D[ar, tgt, :]
    return X, dict(A=A, lag=lag, form=form, weak=weak, w=w, k=k, target=tgt, parent=parent, desc=desc)


def kind(z, act, mu, sd, ctx):
    g, dev = ctx.rng(), z.device
    n, V, L = z.shape
    X, info = graph(g, z)
    K = X.shape[1]
    ar = torch.arange(n, device=dev)
    tgt = info["target"]
    prole = torch.where(bern(g, (n, K), 0.5, dev), ROLE_KNOWN, ROLE_PAST)
    role = torch.where(info["parent"], prole, torch.where(info["desc"], ROLE_PAST, C.DECOY))
    key = torch.where(info["parent"], 0.0, 1.0) + 0.5 * torch.rand((n, K), generator=g, device=dev)
    key[ar, tgt] = 9.0
    order = torch.argsort(key, 1)[:, :K - 1]                       # covariates, parents first
    covs = torch.gather(X, 1, order.unsqueeze(-1).expand(n, K - 1, L))
    roles = torch.gather(role, 1, order)
    lane_sd, lane_mu = sd[:, :K - 1].clone(), mu[:, :K - 1].clone()
    y, r = C.assemble(g, z, act, mu, sd, X[ar, tgt] * sd[:, 0] + mu[:, 0], covs * lane_sd + lane_mu, roles)
    return y, r


def _install_graph(p, inst):
    inst.add_group_kind("causal_graph", kind, p["share"])


def _install_count(p, inst):
    probs = tuple(p["probs"])

    def fn(g, kind_, is_group, V, ctx):
        gen, dev = ctx.rng(), g.device
        names = _groups.GROUP_KINDS
        cov = torch.tensor([nm in ("known_cov", "past_cov") or nm in _hooks.GROUP_KINDS_EXT for nm in names], device=dev)
        k = 1 + C.choice(gen, g.shape[0], probs, dev)
        return torch.where(cov[kind_], (1 + k).clamp(max=V), g)
    inst.set_group_size(fn)


register(Switch("causal_graph", "G1.9", "causal-graph covariates (group kind)", _install_graph, dict(share=0.15)))
register(Switch("cov_count", "G1.11", "covariate count 1-7, skewed to few (group size rule)", _install_count,
                dict(probs=COUNT_P)))
