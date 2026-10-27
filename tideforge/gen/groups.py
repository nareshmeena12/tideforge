"""Generator v0: Section 3d: group kinds, set_group_options, multivariatizers, generate_blocks.

Ported from third_party/tidemark_e21cbf0/tidemark/real/ft3_core.py L503-L503, L2399-L2677 (tidemark commit e21cbf0) by tools/port_gen_v0.py; do not edit by hand
(edit and rerun the tool). Edits vs the source: imports only; the hook points (`# [hook Hn]`, tideforge/gen/_hooks.py).
Must stay bit-exact with ft3_core (tests/test_gen_parity.py, tests/test_gen_source.py).
"""
# flake8: noqa
import math

import torch
import torch.nn.functional as F

from tideforge.gen import _hooks
from tideforge.gen.primitives import (  # noqa: F401
    _CLAMP,
    _MAX_SEG,
    _bern,
    _pick,
    _pick_period,
    _piecewise_constant,
    _segment_ids,
    _standardize,
    _u,
)
from tideforge.gen.batch import (  # noqa: F401
    generate_batch,
)


# >>> ft3_core.py L503-L503
ROLE_TARGET, ROLE_PAST, ROLE_KNOWN = 0, 1, 2     # variate roles inside a group
# <<< ft3_core.py

# >>> ft3_core.py L2399-L2677
# ================= 3d. GROUPS: multivariate / covariate data for group attention =================
# Chronos-2 trained its group attention on synthetic "multivariatizers" applied to univariate
# base series; this is the same idea. A BLOCK is `V` aligned series (lanes) of equal length.
# With probability group_prob the first g lanes (g in 2..V) are turned into a RELATED group of
# one of five kinds; all other lanes stay unrelated single series (each its own group of one).
#
#   mix        every variate = own series + random sparse mix of the others (shared factors:
#              stores of one chain, sensors of one plant). All targets.
#   leadlag    variate i = own series + lagged copy of a driver variate (lag 1..128 steps):
#              the driver's PAST predicts the others' future. All targets.
#   known_cov  1-2 targets driven by covariates whose FUTURE is known (promotions, prices,
#              calendar flags, weather forecasts): additive or multiplicative, linear or
#              nonlinear effects -- including V-SHAPED ones (load is high when it is hot AND
#              when it is cold) and, for a quarter of these groups, an ON/OFF GATE (a 'store
#              open' flag: the target is exactly zero while it is off). ~40% of the other
#              covariates are irrelevant DECOYS.
#   past_cov   same, but the covariates are only observed in the past and act with a lag.
#   unrelated  independent series declared as one group with random roles: nothing to learn
#              from each other -- the model must learn to ignore variates that do not help.

GROUP_KINDS = ("mix", "leadlag", "known_cov", "past_cov", "unrelated")
# v4 used (0.30, 0.15, 0.30, 0.10, 0.15). On fev-bench the gain from groups came almost entirely
# from KNOWN covariates on single-target tasks (Chronos-2: -16% error on those, v4: -4%), so
# known-covariate groups now make up half of all groups.
GROUP_KIND_PROBS = (0.15, 0.10, 0.50, 0.10, 0.15)
_GROUP_KIND_PROBS_V5 = GROUP_KIND_PROBS
_COV_SMALL_GROUP_P = 0.0      # share of covariate groups limited to 2-4 variates
_COV_DOMINANT_P = 0.0         # share of covariate groups whose target is almost entirely its covariates
_SECOND_GROUP_P = 0.0         # chance of a SECOND related group in the upper half of a block's lanes
_COV_R3 = False               # round-3 covariate data (see _mv_covariates); off for the holdout


def set_group_options(cfg=None):
    """Group-data settings of the run that is about to generate (None = the v5 defaults). Module
    globals, like the active period bank: generation is single-threaded."""
    global GROUP_KIND_PROBS, _COV_SMALL_GROUP_P, _COV_DOMINANT_P, _SECOND_GROUP_P, _COV_R3
    cfg = cfg or {}
    GROUP_KIND_PROBS = tuple(cfg.get("group_kind_probs") or _GROUP_KIND_PROBS_V5)
    assert len(GROUP_KIND_PROBS) == len(GROUP_KINDS)
    _COV_SMALL_GROUP_P = float(cfg.get("cov_small_group_p") or 0.0)
    _COV_DOMINANT_P = float(cfg.get("cov_dominant_p") or 0.0)
    _SECOND_GROUP_P = float(cfg.get("second_group_p") or 0.0)
    _COV_R3 = bool(cfg.get("cov_r3") or False)


def _covariate_signal(base, device):
    """base (n, L) standardized series -> covariate values (n, L), its effect input (n, L) and
    is_binary (n, 1). Types: continuous (the base series itself: weather-like), binary events
    (iid days or on/off runs: promotions, holidays), piecewise-constant level (price), and a
    periodic calendar flag (e.g. 'weekend')."""
    n, L = base.shape
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    kind = torch.randint(0, 4, (n, 1), device=device)
    iid = torch.rand(n, L, device=device) < _u(n, 1, device, 0.01, 0.2)
    seg = _segment_ids(n, L, device, 2.0, 20.0)
    runs = torch.gather(torch.rand(n, _MAX_SEG, device=device) < _u(n, 1, device, 0.1, 0.5), 1, seg)
    binary = torch.where(_bern(n, device, 0.5), iid, runs).float()
    price = _piecewise_constant(n, L, device, 1.0, 10.0)
    P = _pick_period(n, L, device, min_cycles=2.0)
    cal = (torch.remainder(t + torch.floor(_u(n, 1, device, 0.0, 1.0) * P), P)
           < _u(n, 1, device, 0.1, 0.5) * P).float()
    raw = torch.where(kind == 0, base, torch.where(kind == 1, binary, torch.where(kind == 2, price, cal)))
    is_bin = (kind == 1) | (kind == 3)
    return raw, torch.where(is_bin, raw, _standardize(raw)), is_bin


def _mv_mix(z, act, mu, sd, device):
    n, V, L = z.shape
    W = torch.randn(n, V, V, device=device) * (torch.rand(n, V, V, device=device) < 0.5).float() \
        * _u(n, 1, device, 0.3, 1.5).unsqueeze(-1)
    W = W * act.unsqueeze(2).float() * act.unsqueeze(1).float() * (1.0 - torch.eye(V, device=device))
    y = z * _u(n, V, device, 0.3, 1.0).unsqueeze(-1) + torch.bmm(W, z)
    return y * sd + mu, torch.zeros(n, V, dtype=torch.long, device=device)


def _mv_leadlag(z, act, mu, sd, device):
    n, V, L = z.shape
    d = torch.exp(_u(n, V, device, 0.0, math.log(128.0))).long()
    idx = (torch.arange(L, device=device).view(1, 1, L) - d.unsqueeze(-1)).clamp_(min=0)
    drv = z[:, :1, :].expand(n, V, L).gather(2, idx)
    sgn = torch.where(torch.rand(n, V, device=device) < 0.5, 1.0, -1.0)
    y = z * _u(n, V, device, 0.2, 1.0).unsqueeze(-1) + (sgn * _u(n, V, device, 0.5, 2.0)).unsqueeze(-1) * drv
    lane0 = (torch.arange(V, device=device) == 0).view(1, V, 1)
    y = torch.where(lane0, z, y)
    return y * sd + mu, torch.zeros(n, V, dtype=torch.long, device=device)


def _mv_covariates(z, act, mu, sd, device, known):
    """Targets (lanes 0..n_tgt-1) driven by covariates (the remaining active lanes).

    ROUND 3 (_COV_R3, off for the holdout; with it off this function draws exactly the random
    numbers it always did):
      look-alikes   35% of groups: the target and 60% of the other lanes get a periodic component
                    with the SAME period and different shapes. A covariate that follows the target's
                    cycle is then no evidence that it drives the target.
      weak drivers  35% of the relevant covariates have an effect of 0.03-0.3 instead of 0.3-2.0.
      joint effect  30% of groups: the nonlinearity is applied to the weighted SUM of the covariates.
      rare flags    see the gate below."""
    n, V, L = z.shape
    lane = torch.arange(V, device=device).unsqueeze(0)
    g = act.sum(1, keepdim=True)
    n_tgt = torch.where((torch.rand(n, 1, device=device) < 0.25) & (g >= 3), 2, 1)
    is_cov = act & (lane >= n_tgt)
    if _COV_R3:
        Psh = _pick_period(n, L, device).unsqueeze(-1)                             # (n, 1, 1)
        tL = torch.arange(L, device=device, dtype=torch.float32).view(1, 1, L)
        wave = torch.zeros(n, V, L, device=device)
        for k in (1, 2, 3):
            ang = 2 * math.pi * k * tL / Psh
            wave = wave + (torch.randn(n, V, 1, device=device) * torch.cos(ang)
                           + torch.randn(n, V, 1, device=device) * torch.sin(ang)) / k
        wave = wave / (wave.std(2, keepdim=True) + 1e-6)
        shared = act & ((lane < n_tgt) | (torch.rand(n, V, device=device) < 0.6)) & _bern(n, device, 0.35)
        z = z + (_u(n, V, device, 0.5, 2.0) * shared.float()).unsqueeze(-1) * wave
    raw, e, is_bin = _covariate_signal(z.reshape(n * V, L), device)
    raw, e, is_bin = raw.view(n, V, L), e.view(n, V, L), is_bin.view(n, V)
    fn = torch.randint(0, 5, (n, V, 1), device=device)       # effect shape per covariate
    vshape = (e - _u(n, V, device, -1.0, 1.0).unsqueeze(-1)).abs()        # high at both extremes
    ge = torch.where(fn == 0, e, torch.where(fn == 1, torch.relu(e), torch.where(
        fn == 2, torch.tanh(2.0 * e), torch.where(fn == 3, (e > 0.5).float(), vshape))))
    # lag of the effect: known covariates act (almost always) at once; past-only ones act later
    if known:
        d = torch.where(torch.rand(n, V, device=device) < 0.8, 0, torch.randint(1, 9, (n, V), device=device))
    else:
        d = torch.exp(_u(n, V, device, 0.0, math.log(96.0))).long()
    idx = (torch.arange(L, device=device).view(1, 1, L) - d.unsqueeze(-1)).clamp_(min=0)
    ge = ge.gather(2, idx)
    relevant = (is_cov & (torch.rand(n, V, device=device) < 0.6)).float()          # the rest are decoys
    sgn = torch.where(torch.rand(n, 2, V, device=device) < 0.5, 1.0, -1.0)
    beta = sgn * _u(n * 2, V, device, 0.3, 2.0).view(n, 2, V) * relevant.unsqueeze(1)
    if _COV_R3:
        weak = torch.rand(n, 1, V, device=device) < 0.35
        beta = torch.where(weak, sgn * torch.exp(_u(n * 2, V, device, math.log(0.03), math.log(0.3))).view(n, 2, V)
                           * relevant.unsqueeze(1), beta)
    eff = torch.bmm(beta, ge)                                                      # (n, 2, L)
    if _COV_R3:
        lin = torch.bmm(beta, e.gather(2, idx))                                    # weighted sum of the (lagged) covariates
        ls = lin.std(2, keepdim=True)
        s_ = lin / (ls + 1e-6)
        gk = torch.randint(0, 5, (n, 1, 1), device=device)
        G = torch.where(gk == 0, torch.relu(s_), torch.where(gk == 1, torch.tanh(1.5 * s_), torch.where(
            gk == 2, s_.abs(), torch.where(gk == 3, (s_ > 0.7).float(), torch.expm1(s_.clamp(max=3.0))))))
        G = (G - G.mean(2, keepdim=True)) / (G.std(2, keepdim=True) + 1e-6) * ls
        eff = torch.where(_bern(n, device, 0.3).unsqueeze(-1), G, eff)
    eff = F.pad(eff, (0, 0, 0, V - 2)) if V > 2 else eff[:, :V]
    is_tgt = (act & (lane < n_tgt)).unsqueeze(-1)
    # COVARIATE-DOMINANT targets: the target's own dynamics shrink to 5-30%, so it is almost a
    # pure function of its covariates (a price set by forecast load) -- history alone cannot
    # forecast it, the covariates' future values can.
    dom = _bern(n, device, _COV_DOMINANT_P) if _COV_DOMINANT_P > 0 else torch.zeros(n, 1, dtype=torch.bool, device=device)
    own = torch.where(dom, _u(n, 1, device, 0.05, 0.3), torch.ones(n, 1, device=device)).unsqueeze(-1)
    add = (z * own + eff) * sd + mu
    mult = (z * sd + mu) * (1.0 + 0.5 * torch.tanh(eff))
    tgt = torch.where((_bern(n, device, 0.7) | dom).unsqueeze(-1), add, mult)
    cov = torch.where(is_bin.unsqueeze(-1), raw, raw * sd + mu)
    if known:
        # ON/OFF GATE: the first covariate lane becomes a known 0/1 'open' flag and the targets
        # are exactly zero while it is off -- closed days at a fixed place in a cycle (a weekly
        # rest day) or irregular closed runs. No multiplicative effect above can produce this.
        t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
        Pg = _pick(n, [7.0, 7.0, 24.0, 168.0], device)
        off_len = 1.0 + torch.floor(_u(n, 1, device, 0.0, 0.3) * Pg)
        periodic = torch.remainder(t + torch.floor(_u(n, 1, device, 0.0, 1.0) * Pg), Pg) < off_len
        runs = torch.gather(torch.rand(n, _MAX_SEG, device=device) < _u(n, 1, device, 0.05, 0.25), 1,
                            _segment_ids(n, L, device, 2.0, 12.0))
        if _COV_R3:
            # RARE SHUTDOWN FLAGS (round 3). Closed days are now one of: the weekly rest day, irregular
            # runs, RARE days only (0.3-4% of the steps, each lasting 1-3 steps: holidays), or the rest
            # day PLUS rare days. While closed the target is zero (75%) or a fraction of itself. The
            # flag is shown as 'open' (1 = normal) or as 'holiday' (1 = closed; for 30% of those a
            # code 1-3, like a holiday type), in ANY covariate lane, for 40% of the groups.
            rs = torch.rand(n, L, device=device) < torch.exp(_u(n, 1, device, math.log(0.003), math.log(0.04)))
            rl = torch.randint(1, 4, (n, 1), device=device)
            rare = rs.clone()
            rare[:, 1:] |= rs[:, :-1] & (rl >= 2)
            rare[:, 2:] |= rs[:, :-2] & (rl >= 3)
            gk2 = torch.randint(0, 4, (n, 1), device=device)
            closed = torch.where(gk2 == 0, periodic, torch.where(gk2 == 1, runs, torch.where(gk2 == 2, rare, periodic | rare)))
            keep = torch.where(_bern(n, device, 0.75), torch.zeros(n, 1, device=device), _u(n, 1, device, 0.05, 0.5))
            factor = torch.where(closed, keep, torch.ones(n, 1, device=device)).unsqueeze(1)          # (n, 1, L)
            code = (1 + torch.remainder(_segment_ids(n, L, device, 3.0, 20.0), 3)).float()
            shown = torch.where(_bern(n, device, 0.5), (~closed).float(),
                                closed.float() * torch.where(_bern(n, device, 0.3), code, torch.ones_like(code)))
            gated = _bern(n, device, 0.4) & (g > n_tgt)                                               # (n, 1)
            gl = n_tgt + torch.floor(torch.rand(n, 1, device=device) * (g - n_tgt).clamp(min=1).float()).long()
            gate_lane = (is_cov & (lane == gl) & gated).unsqueeze(-1)
            tgt = torch.where(gated.unsqueeze(-1), tgt * factor, tgt)
            cov = torch.where(gate_lane, shown.unsqueeze(1).expand(n, V, L), cov)
        else:
            open_f = (~torch.where(_bern(n, device, 0.6), periodic, runs)).float().unsqueeze(1)   # (n, 1, L)
            gated = _bern(n, device, 0.25) & (g > n_tgt)                                          # (n, 1)
            gate_lane = (is_cov & (lane == n_tgt) & gated).unsqueeze(-1)
            tgt = torch.where(gated.unsqueeze(-1), tgt * open_f, tgt)
            cov = torch.where(gate_lane, open_f.expand(n, V, L), cov)
    y = torch.where(is_tgt, tgt, cov)
    role = torch.where(is_cov, ROLE_KNOWN if known else ROLE_PAST, ROLE_TARGET)
    return y, role


def _mv_unrelated(z, act, mu, sd, device):
    n, V, L = z.shape
    role = torch.multinomial(torch.tensor([0.4, 0.3, 0.3]), n * V, replacement=True).view(n, V).to(device)
    role[:, 0] = ROLE_TARGET
    return z * sd + mu, role


def _group_pass(x, is_group, g, kind, device):
    """Turn lanes [0, g) of the blocks flagged is_group into ONE related group of the given kind.
    x (N, V, L) -> (out (N, V, L), role (N, V), act (N, V) = lanes that belong to the group)."""
    N, V, L = x.shape
    lane = torch.arange(V, device=device).unsqueeze(0).expand(N, V)
    role = torch.zeros(N, V, dtype=torch.long, device=device)
    act = is_group.unsqueeze(1) & (lane < g.unsqueeze(1))
    mu = x.mean(2, keepdim=True)
    sd = x.std(2, keepdim=True) + 1e-6
    out = x.clone()
    for k, name in enumerate(GROUP_KINDS):
        idx = (is_group & (kind == k)).nonzero().squeeze(1)
        if idx.numel() == 0:
            continue
        zk, ak, mk, sk = (x[idx] - mu[idx]) / sd[idx], act[idx], mu[idx], sd[idx]
        if name == "mix":
            y, r = _mv_mix(zk, ak, mk, sk, device)
        elif name == "leadlag":
            y, r = _mv_leadlag(zk, ak, mk, sk, device)
        elif name == "unrelated":
            y, r = _mv_unrelated(zk, ak, mk, sk, device)
        elif name in _hooks.GROUP_KINDS_EXT:  # [hook H5]
            y, r = _hooks.group_kind(name, zk, ak, mk, sk, device)  # [hook H5]
        else:
            y, r = _mv_covariates(zk, ak, mk, sk, device, known=(name == "known_cov"))
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).clamp(-_CLAMP, _CLAMP)
        out[idx] = torch.where(ak.unsqueeze(-1), y, x[idx])
        role[idx] = torch.where(ak, r, torch.zeros_like(r))
    return out, role, act


def _multivariatize(x, group_prob, device):
    """x (N, V, L) independent series -> (x, role (N, V), lgid (N, V)): the lanes of a related
    group share one lgid (0 for the group in lanes [0, g); V//2 for an optional second group in
    the upper half); every other lane keeps lgid = its lane index (a group of one)."""
    N, V, L = x.shape
    role = torch.zeros(N, V, dtype=torch.long, device=device)
    lane = torch.arange(V, device=device).unsqueeze(0).expand(N, V)
    if V < 2 or group_prob <= 0:
        return x, role, lane.clone()
    probs = torch.tensor(GROUP_KIND_PROBS)
    is_group = torch.rand(N, device=device) < group_prob
    g = torch.randint(2, V + 1, (N,), device=device)
    kind = torch.multinomial(probs, N, replacement=True).to(device)
    if _COV_SMALL_GROUP_P > 0:      # real covariate tasks have 1-4 covariates, not up to 7
        small = (torch.rand(N, device=device) < _COV_SMALL_GROUP_P) & ((kind == 2) | (kind == 3))
        g = torch.where(small, torch.randint(2, min(V, 4) + 1, (N,), device=device), g)
    g = _hooks.group_size(g, kind, is_group, V, device)  # [hook H6]
    out, role, act = _group_pass(x, is_group, g, kind, device)
    lgid = torch.where(act, torch.zeros_like(lane), lane)
    h = V // 2
    if _SECOND_GROUP_P > 0 and h >= 2:
        # a block has V lanes but a covariate group has one or two targets, so one group per block
        # caps covariate-driven targets at ~6% of all series. When lanes h..V-1 are free (no first
        # group, or one of size <= h) they can hold a second, independent group: the same routine
        # is run on the block rolled by h lanes.
        free = (~is_group) | (g <= h)
        is2 = free & (torch.rand(N, device=device) < _SECOND_GROUP_P)
        g2 = torch.randint(2, V - h + 1, (N,), device=device)
        kind2 = torch.multinomial(probs, N, replacement=True).to(device)
        o2, r2, a2 = _group_pass(out.roll(-h, dims=1), is2, g2, kind2, device)
        out, r2, a2 = o2.roll(h, dims=1), r2.roll(h, dims=1), a2.roll(h, dims=1)
        role = torch.where(a2, r2, role)
        lgid = torch.where(a2, torch.full_like(lane, h), lgid)
    out, role, lgid = _hooks.run_groups("after_groups", out, role, lgid, device)  # [hook H7]
    return out, role, lgid


@torch.no_grad()
def generate_blocks(N, V, L, device, weights=None, profile="v1", group_prob=0.0):
    """N blocks of V aligned series of length L: x (N, V, L), role (N, V), lgid (N, V)."""
    # per-family rounding in _allocate can return a few rows more than asked for small batches
    _hooks.tick("blocks")  # [hook tick]
    x = generate_batch(N * V, L, device, weights, profile=profile)
    while x.shape[0] < N * V:
        x = torch.cat([x, generate_batch(N * V, L, device, weights, profile=profile)], 0)
    return _hooks.final_blocks(_multivariatize(x[:N * V].view(N, V, L), group_prob, device), device)  # [hook H8]

# <<< ft3_core.py
