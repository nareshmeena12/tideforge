"""Switch-only experiments (EXPERIMENTS.md section 8, step 2): every switch draws nothing when off, the post-processing ones
leave torch's global stream as in v0, and each one's statistics match its parameters. Plus G1.2's lead_noise and the
experiment configs."""
import math

import numpy as np
import pytest
import torch

import tideforge.gen as G
from tideforge.expcfg import EXP_DIR, check_train, load_exp
from tideforge.gen import _hooks, switches
from tideforge.gen.ext import g1_lead_noise as LN
from tideforge.gen.ext import g3_share, g4_breaks, g6_gaps, g7_integer
from tideforge.gen.groups import ROLE_KNOWN, ROLE_PAST, ROLE_TARGET

B0 = G.b0()
NEW = ["family_share", "breaks_rate", "breaks_late", "collapse", "gaps_scattered", "gaps_block", "gaps_periodic", "no_ffill",
       "int_share"]
# switches whose on-run must leave torch's global RNG exactly as the off-run (H4, H8, a profile value whose draws are still
# made); the others reweight families, add layers or change values before the noise layer (breaks_late: its H1 override
# changes values before noise, so on CPU the later poisson draws can shift, as switches.py says for H2 / H3)
STREAM_SAFE = {"gaps_scattered": {}, "gaps_block": {},
               "gaps_periodic": {}, "no_ffill": {}, "int_share": {}}


@pytest.fixture(autouse=True)
def strict():
    _hooks.STRICT = True
    yield
    _hooks.STRICT = False
    G.configure(B0)


def blocks(gc, N=64, V=8, L=512, seed=0):
    prof = G.configure(gc)
    torch.manual_seed(seed)
    out = G.generate_blocks(N, V, L, "cpu", profile=prof, group_prob=gc.group_prob)
    return out, torch.get_rng_state()


def g(seed=0):
    return torch.Generator().manual_seed(seed)


# ---------------------------------------------------------------------------------------------------- off / stream
@pytest.mark.parametrize("name", NEW)
def test_off_draws_nothing(name):
    assert name in switches.REGISTRY
    (x0, r0, l0), st0 = blocks(B0)
    assert not _hooks.installed() and not _hooks._gens
    (x1, r1, l1), st1 = blocks(B0)
    assert torch.equal(x0, x1) and torch.equal(st0, st1)


@pytest.mark.parametrize("name", sorted(STREAM_SAFE))
def test_on_keeps_the_global_stream(name):
    base = B0.with_changes(**STREAM_SAFE[name])
    on = base.with_changes(switches={name: True})
    (x0, _, _), st0 = blocks(base)
    (x1, _, _), st1 = blocks(on)
    assert torch.equal(st0, st1)
    assert not torch.equal(torch.nan_to_num(x0, nan=1e9), torch.nan_to_num(x1, nan=1e9))


def test_set_profile_unknown_key_fails():
    def bad(p, inst):
        inst.set_profile("no_such_setting", lambda v: v)
    switches.register(switches.Switch("t_badprof", "TEST", "", bad))
    try:
        with pytest.raises(KeyError, match="no setting"):
            G.configure(B0.with_changes(switches={"t_badprof": True}))
    finally:
        switches.unregister("t_badprof")


# ---------------------------------------------------------------------------------------------------- G3.1
@pytest.mark.parametrize("share", [0.15, 0.25])
def test_family_share(share):
    prof = G.configure(B0.with_changes(switches={"family_share": {"share": share}}))
    w = G.GEN_PROFILES[prof]["weights"]
    assert w["long_cycle"] / sum(w.values()) == pytest.approx(share)
    with pytest.raises(ValueError):
        g3_share.target_weight(w, "long_cycle", 1.0)


# ---------------------------------------------------------------------------------------------------- G4
def test_breaks_rate_sets_the_profile():
    for p in (0.1, 0.3):
        prof = G.configure(B0.with_changes(switches={"breaks_rate": {"p": p}}))
        assert G.GEN_PROFILES[prof]["breaks_p"] == p
    assert G.GEN_PROFILES[G.configure(B0)]["breaks_p"] == 0.0


def test_breaks_late_position_and_rate():
    x = torch.randn(4000, 1024).cumsum(1)
    y, tb = g4_breaks.breaks(x, 0.2, g(), 0.7, 0.9)
    hit = tb[:, 0] >= 0
    assert abs(float(hit.float().mean()) - 0.2) < 3 * math.sqrt(0.2 * 0.8 / 4000)
    t = tb[tb >= 0]
    assert float(t.min()) >= 0.7 * 1024 - 1 and float(t.max()) <= 0.9 * 1024
    assert torch.equal(y[~hit], x[~hit])


def test_breaks_late_simulated_crop_runs():
    cfg = {k: v for k, v in G.config.load_ft3_cfg().items()}
    s = g4_breaks.late_share_in_context(cfg, torch.rand(10000) * 0.2 + 0.7, n_layouts=2)
    assert s["n"] > 0 and s["late"] + s["outside"] + s["inside_other"] == pytest.approx(1.0)


def test_collapse_shape_and_rate():
    n, L = 4000, 512
    x = torch.randn(n, L) * 0.1 + 5.0
    y, tb = g4_breaks.collapse(x, 0.05, g())
    hit = tb >= 0
    assert abs(float(hit.float().mean()) - 0.05) < 3 * math.sqrt(0.05 * 0.95 / n)
    assert torch.equal(y[~hit], x[~hit])
    ratio = []
    for i in hit.nonzero().squeeze(1).tolist():
        t0 = int(tb[i])
        assert 0.2 * L - 1 <= t0 <= 0.9 * L
        assert torch.equal(y[i, :t0], x[i, :t0])
        assert float((x[i, t0:t0 + 20] - y[i, t0:t0 + 20]).mean()) > 1.5 * float(x[i].std())   # the drop (>= 2 sd)
        d = (x[i] - y[i]).double()                                           # the collapse still in place
        ratio.append(float(d[-20:].mean()) / float(d[t0:t0 + 20].mean()))
    ratio = np.array(ratio)                                                   # partial recovery: part of the collapse stays
    assert (ratio > 0).all() and np.median(ratio) < 0.9 and (ratio < 1).mean() > 0.8


# ---------------------------------------------------------------------------------------------------- G6
def roles(N=6000, V=8, seed=1):
    r = torch.randint(0, 3, (N, V), generator=g(seed))
    return r


def _hook(name, params):
    p = dict(switches.REGISTRY[name].params, **params)
    hooks = []

    class I:
        def add_hook(self, point, fn, prio):
            hooks.append(fn)
    switches.REGISTRY[name].install(p, I())
    ctx = _hooks.Ctx("blocks", torch.device("cpu"), 0, switch=name)
    ctx.rng = lambda name=None, _g=g(3): _g
    return lambda x, role: hooks[0](x, role, None, ctx)[0]


@pytest.mark.parametrize("name,params,frac", [("gaps_scattered", {"p": 0.05}, 0.05),
                                              ("gaps_block", {"f_lo": 0.05, "f_hi": 0.5}, 0.275),
                                              ("gaps_periodic", {"d_lo": 0.15, "d_hi": 0.4}, "periodic")])
def test_gaps_statistics(name, params, frac):
    role = roles()
    x = torch.randn(*role.shape, 512)
    y = _hook(name, dict(params, target_lanes=0.3, cov_lanes=0.6))(x, role)
    nan = torch.isnan(y)
    lane = nan.any(-1)
    tgt = role == ROLE_TARGET
    assert abs(float(lane[tgt].float().mean()) - 0.3) < 0.03
    assert abs(float(lane[~tgt].float().mean()) - 0.6) < 0.03
    assert not nan[..., 0].any() and not nan[..., -128:].any()               # first point and the tail stay observed
    region = 512 - 128 - 1
    f = nan[lane].float().sum(-1) / region
    if frac == "periodic":       # whole points off-duty: ceil(d P) / P on average, P from the bank, d ~ U(0.15, 0.4)
        d = np.random.default_rng(0).uniform(0.15, 0.4, 200000)
        frac = float(np.mean([np.ceil(d * P) / P for P in (12, 24, 168)]))
    assert abs(float(f.mean()) - frac) < 0.02 and float(f.max()) <= g6_gaps.MAX_MISSING + 1e-6
    assert torch.equal(y[~nan], x[~nan])


def test_no_ffill_profile():
    prof = G.configure(B0.with_changes(switches={"no_ffill": True}))
    assert G.GEN_PROFILES[prof]["miss_p"] == 0.0
    assert G.GEN_PROFILES[G.configure(B0)]["miss_p"] > 0


# ---------------------------------------------------------------------------------------------------- G7.4
@pytest.mark.parametrize("share", [0.10, 0.20])
def test_int_share_total(share):
    n, L = 6000, 256
    x = torch.randn(n, L).cumsum(1)
    is_count = torch.rand(n, generator=g(5)) < 0.03
    hooks = []

    class I:
        def add_hook(self, point, fn, prio):
            hooks.append(fn)
    switches.REGISTRY["int_share"].install({"share": share}, I())
    ctx = _hooks.Ctx("batch", torch.device("cpu"), L, None, is_count.clone(), torch.zeros(n, dtype=torch.bool),
                     torch.ones(n, dtype=torch.bool), None, switch="int_share")
    ctx.rng = lambda name=None, _g=g(6): _g
    y = hooks[0](x, ctx)
    assert abs(float(ctx.is_count.float().mean()) - share) < 3 * math.sqrt(share * (1 - share) / n)
    new = ctx.is_count & ~is_count
    assert torch.equal(y[new], torch.round(y[new])) and bool((y[new] >= 0).all())
    assert torch.equal(y[~new], x[~new])


def test_int_share_keeps_the_shape():
    t = torch.arange(512, dtype=torch.float32)
    x = (torch.sin(2 * math.pi * t / 24) * 3 + t / 100).expand(200, 512).clone() + torch.randn(200, 512) * 0.1
    c, lam = g7_integer.to_counts(x, g(7))
    z = (x - x.mean(1, keepdim=True)) / x.std(1, keepdim=True)
    r_lam = [np.corrcoef(np.log(lam[i]), z[i])[0, 1] for i in range(200)]
    assert min(r_lam) > 0.999                                                 # log intensity is linear in the series
    big = lam.mean(1) > 5
    r_c = [np.corrcoef(c[i], x[i])[0, 1] for i in big.nonzero().squeeze(1).tolist()]
    assert big.sum() > 20 and np.median(r_c) > 0.7                            # the counts keep seasonality and trend


# ---------------------------------------------------------------------------------------------------- G1.2
def test_lead_noise_zero_before_origin_and_linear_variance():
    n, T, step = 20000, 256, 32
    x = torch.randn(n, T)
    origin = torch.full((n,), 128)
    origin[:100] = -1
    s0 = torch.full((n,), 0.3)
    y = LN.lead_noise(x, origin, s0, g(8), step=step)
    assert torch.equal(y[:, :128], x[:, :128]) and torch.equal(y[:100], x[:100])
    eps = (y - x)[100:, 128:]
    sd = x[100:, :128].std(1, unbiased=False, keepdim=True)
    v = ((eps / sd) ** 2).mean(0)
    h = torch.arange(1, T - 128 + 1, dtype=torch.float32)
    assert torch.allclose(v, 0.3 ** 2 * h / step, rtol=0.06)


def test_draw_sigma0():
    s = LN.draw_sigma0(20000, g(9))
    on = s > 0
    assert abs(float(on.float().mean()) - 0.5) < 0.02
    assert float(s[on].min()) >= 0.05 and float(s[on].max()) <= 0.5


# ---------------------------------------------------------------------------------------------------- configs
def test_every_experiment_config_loads():
    ids = sorted(p.stem for p in EXP_DIR.glob("G*.yaml"))
    assert {"G1.1", "G1.2", "G3.1a", "G3.1b", "G3.3", "G4.1", "G4.2a", "G4.2b", "G4.3", "G4.4", "G5.1", "G6.1", "G6.2",
            "G6.3", "G6.4", "G7.4a", "G7.4b"} <= set(ids)
    for i in ids:
        e = load_exp(i)
        assert " " not in e.run_id() and "(" not in e.run_id()
    assert load_exp("G3.3").train == {"long_row_frac": 0.5}


def test_train_section_is_checked():
    with pytest.raises(KeyError):
        check_train({"lr": 1e-3})
    with pytest.raises(ValueError):
        check_train({"long_row_frac": 1.5})
    with pytest.raises(KeyError):
        check_train({"lead_noise": {"sigma": 1}})
