"""causal_cov (experiment AD): covariates that act on targets through causal operators (iAmTime Appendix D.3, TimeH2O TCM style).

Runs at H7 on the blocks of generator series. A chosen block gets a group of g lanes: targets first, then c covariates (known or past),
c <= 40% of the block's lanes, so at least 60% of every block's lanes are targets (lanes outside the group stay lone targets). A
group of 2 is 1 target and 1 covariate: the 60% holds per block, not per group.

Each covariate lane draws one effect strength class: zero, weak, strong (shares in the config). A zero covariate has no edge at all, so
it has no path to any target, also through other targets. Edges run covariate -> target, and earlier target -> later target
(a lagged link, target_edge_p); never target -> covariate. Each covariate -> target edge draws one operator, in z units of the lanes:
    blend     y <- b y + a x + e
    nonlin    y <- y + a f(x), f in log, exp, power, tanh
    trend     y <- y + a cumsum(x) / sqrt(L)
    season    y <- y + a tanh(x) sin(2 pi t / p + phase)
    shock     sparse pulses with constant / linear / exponential decay, added to x and, times a, to y
    lag       y_t <- y_t + a x_(t-l)
    coint     y <- a x + (y - slow average of y): y stays tied to x, a filter stand-in for error correction
    granger   y_t <- y_t + sum_k a exp(-b k) x_(t-k)
Touched lanes are clipped to +-5 std; a lane nothing acted on keeps its values exactly. Draws depend on no value of x, so changing a
zero covariate changes no target. pop_effect() returns the true strength per lane of the blocks generated since the last call
(N, V): on a covariate its own strength, on a target the largest strength with a path to it; the batcher carries it as b["effect"].
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from tideforge.gen.ext._rand import bern, randint, u
from tideforge.gen.groups import ROLE_KNOWN, ROLE_PAST, ROLE_TARGET
from tideforge.gen.switches import Switch, register

OPS = ("blend", "nonlin", "trend", "season", "shock", "lag", "coint", "granger")
PERIODS = (7.0, 12.0, 24.0, 168.0)
KERNEL_LEN = 32
_EFFECT: list = []


def pop_effect() -> torch.Tensor | None:
    if not _EFFECT:
        return None
    e = torch.cat(_EFFECT, 0)
    _EFFECT.clear()
    return e


def causal_conv(x: torch.Tensor, kern: torch.Tensor) -> torch.Tensor:
    """y_t = sum_k kern[k] x_(t-k), the start padded with the first value."""
    p = F.pad(x.unsqueeze(1), (len(kern) - 1, 0), mode="replicate")
    return F.conv1d(p, kern.flip(0).view(1, 1, -1)).squeeze(1)


def lagged(x: torch.Tensor, lag: torch.Tensor) -> torch.Tensor:
    t = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
    return x.gather(1, (t - lag.unsqueeze(1)).clamp(min=0))


def apply_op(op: int, y, x, a, g, p):
    """Returns (y, x): the target and covariate rows after one edge. a (n, 1)."""
    n, L = y.shape
    dev = y.device
    name = OPS[op]
    if name == "blend":
        b = u(g, (n, 1), 0.7, 1.0, dev)
        return b * y + a * x + 0.1 * a * torch.randn(n, L, generator=g, device=dev), x
    if name == "nonlin":
        f = randint(g, 0, 3, (n, 1), dev)
        c = x.clamp(-3, 3)
        v = torch.stack([torch.log1p(c.abs()), c.exp() / math.e, c ** 2, c.tanh()])
        return y + a * v.gather(0, f.expand(n, L).unsqueeze(0)).squeeze(0), x
    if name == "trend":
        return y + a * x.cumsum(1) / L ** 0.5, x
    if name == "season":
        per = torch.tensor(PERIODS, device=dev)[randint(g, 0, len(PERIODS) - 1, (n,), dev)].unsqueeze(1)
        t = torch.arange(L, device=dev, dtype=y.dtype).unsqueeze(0)
        return y + a * x.tanh() * torch.sin(2 * math.pi * t / per + u(g, (n, 1), 0.0, 2 * math.pi, dev)), x
    if name == "shock":
        e = bern(g, (n, L), 0.005, dev).to(y.dtype) * u(g, (n, L), 1.0, 3.0, dev) * torch.sign(u(g, (n, L), -1.0, 1.0, dev))
        k = torch.arange(KERNEL_LEN, device=dev, dtype=y.dtype)
        kind = int(randint(g, 0, 2, (1,), dev))
        kern = [torch.ones_like(k), 1 - k / KERNEL_LEN, 0.8 ** k][kind]
        s = causal_conv(e, kern)
        return y + a * s, x + s
    if name == "lag":
        return y + a * lagged(x, randint(g, 1, p["lag_max"], (n,), dev)), x
    if name == "coint":
        kern = torch.full((KERNEL_LEN,), 1.0 / KERNEL_LEN, device=dev, dtype=y.dtype)
        return a * x + y - causal_conv(y, kern), x
    k = torch.arange(1, KERNEL_LEN + 1, device=dev, dtype=y.dtype)
    kern = torch.cat([torch.zeros(1, device=dev), (-float(u(g, (1,), 0.05, 0.5, dev)) * k).exp()])
    return y + a * causal_conv(x, kern), x


def _standard(x):
    m, s = x.mean(-1, keepdim=True), x.std(-1, keepdim=True).clamp(min=1e-6)
    return (x - m) / s, m, s


@torch.no_grad()
def causal_blocks(x, role, lgid, g, p):
    """x (N, V, L) -> (x, role, lgid, effect (N, V)). Blocks not chosen (share p["p"]) keep everything, with effect 0."""
    N, V, L = x.shape
    dev = x.device
    hit = bern(g, (N,), p["p"], dev)
    effect = torch.zeros(N, V, device=dev)
    max_c = max(1, min(int(p["max_cov"]), int(0.4 * V)))
    size = randint(g, 2, V, (N,), dev)
    nc = torch.minimum(randint(g, 1, max_c, (N,), dev), size - 1)
    first_cov = size - nc                                              # covariates at lanes first_cov .. size - 1
    lane = torch.arange(V, device=dev).unsqueeze(0)
    in_group = lane < size.unsqueeze(1)
    is_cov = in_group & (lane >= first_cov.unsqueeze(1))
    known = bern(g, (N, V), p["known_p"], dev)
    new_role = torch.where(is_cov, torch.where(known, ROLE_KNOWN, ROLE_PAST), torch.full_like(role, ROLE_TARGET))
    new_gid = torch.where(in_group, torch.zeros_like(lane), lane.expand(N, V))

    r = torch.rand(N, V, generator=g, device=dev)
    zero, weak = p["zero"], p["zero"] + p["weak"]
    st = torch.where(r < weak, u(g, (N, V), p["weak_lo"], p["weak_hi"], dev), u(g, (N, V), p["strong_lo"], p["strong_hi"], dev))
    st = torch.where((r < zero) | ~is_cov, torch.zeros_like(st), st)
    sign = torch.sign(u(g, (N, V), -1.0, 1.0, dev))

    z, mu, sd = _standard(x)
    z = z.clone()
    touched = torch.zeros(N, V, dtype=torch.bool, device=dev)
    reach = torch.zeros(N, V, device=dev)                              # the largest strength with a path to a target
    for t in range(V):
        is_t = hit & in_group[:, t] & ~is_cov[:, t]
        for off in range(max_c):
            ci = (first_cov + off).clamp(max=V - 1).unsqueeze(1)
            edge = is_t & (off < nc) & (st.gather(1, ci).squeeze(1) > 0)
            ops = randint(g, 0, len(OPS) - 1, (N,), dev)                   # drawn for every block, so draws do not depend on the strengths
            for op in range(len(OPS)):
                i = (edge & (ops == op)).nonzero().squeeze(1)
                if not len(i):
                    continue
                c = ci[i, 0]
                a = (st.gather(1, ci).squeeze(1) * sign.gather(1, ci).squeeze(1))[i].unsqueeze(1)
                z[i, t], z[i, c] = apply_op(op, z[i, t], z[i, c], a, g, p)
                touched[i, t] = True
                reach[i, t] = torch.maximum(reach[i, t], st[i, c])
        if t:                                                          # a lagged link from an earlier target; never to a covariate
            src = randint(g, 0, t - 1, (N,), dev)
            link = is_t & ~is_cov.gather(1, src.unsqueeze(1)).squeeze(1) & (src < size) & bern(g, (N,), p["target_edge_p"], dev)
            i = link.nonzero().squeeze(1)
            if len(i):
                z[i, t], _ = apply_op(OPS.index("lag"), z[i, t], z[i, src[i]], u(g, (len(i), 1), 0.2, 0.6, dev), g, p)
                touched[i, t] = True
                reach[i, t] = torch.maximum(reach[i, t], reach[i, src[i]])

    touched = touched | (hit.unsqueeze(1) & is_cov & (st > 0))        # a shock may have moved a covariate; a zero one is never touched
    y = z.clamp(-5, 5) * sd + mu
    out = torch.where(touched.unsqueeze(-1), y, x)
    h = hit.unsqueeze(1)
    effect = torch.where(h, torch.where(is_cov, st, reach), effect)
    return out, torch.where(h, new_role, role), torch.where(h, new_gid, lgid), effect


def _install(p, inst):
    shares = p["zero"] + p["weak"] + p["strong"]
    if abs(shares - 1) > 1e-9:
        raise ValueError(f"causal_cov: zero + weak + strong must be 1, not {shares}")

    def hook(x, role, lgid, ctx):
        x, role, lgid, eff = causal_blocks(x, role, lgid, ctx.rng(), p)
        _EFFECT.append(eff.cpu())
        return x, role, lgid

    inst.add_hook("after_groups", hook, "structure")


register(Switch("causal_cov", "AD", "groups whose covariates act on the targets through causal operators", _install,
                dict(p=1.0, zero=0.3, weak=0.3, strong=0.4, weak_lo=0.05, weak_hi=0.2, strong_lo=0.5, strong_hi=1.5,
                     max_cov=3, known_p=0.7, target_edge_p=0.3, lag_max=32)))
