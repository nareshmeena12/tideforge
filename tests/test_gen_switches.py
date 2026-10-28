"""Switch registry and hook points: v0 is restored by configure(), hooks that draw nothing keep v0 bit-exact, a post-processing
switch changes only what it touches and leaves the global RNG as in v0, NaN only at final_blocks, priorities, families
(add / replace), hash and run id, worker processes."""
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import tidemark.real.ft3_core as C
import tideforge.gen as G
from tideforge.gen import _hooks, switches
from tideforge.gen.config import GenConfig
from tideforge.gen.switches import Switch, register, unregister

STRESS = dict(gen_additions={k: True for k in C.GEN_ADDITIONS_ALL}, period_ext_share=0.3, cov_r3=True, cov_dominant_p=0.3,
              second_group_p=0.5, cov_small_group_p=0.4, group_kind_probs=[0.1, 0.1, 0.3, 0.3, 0.2])
BASE = GenConfig()
STRESS_GC = GenConfig.from_ft3_cfg(dict(C.CONFIG, **STRESS))


@pytest.fixture(autouse=True)
def clean_state():
    _hooks.STRICT = True
    yield
    _hooks.STRICT = False
    for n in [n for n in switches.REGISTRY if n.startswith("t_")]:
        unregister(n)
    G.configure(BASE)
    C.set_group_options(None)
    C.GEN_PROFILES["clean2"] = C.make_clean2_profile()


def run(fn, seed=0):
    torch.manual_seed(seed)
    out = fn()
    return out, torch.get_rng_state()


def ref_blocks(cfg, N=16, V=8, L=256, gp=0.7):
    C.set_group_options(cfg)
    name = C.resolve_profile(cfg)
    return lambda: C.generate_blocks(N, V, L, "cpu", profile=name, group_prob=gp)


def port_blocks(gc, N=16, V=8, L=256, gp=0.7, seed=0, stream=0):
    name = G.configure(gc, seed=seed, stream=stream)
    return lambda: G.generate_blocks(N, V, L, "cpu", profile=name, group_prob=gp)


def assert_same(a, b):
    a = a if isinstance(a, tuple) else (a,)
    b = b if isinstance(b, tuple) else (b,)
    assert len(a) == len(b) and all(x.dtype == y.dtype and torch.equal(x, y) for x, y in zip(a, b))


# ---------------------------------------------------------------------------------------------------- v0 restored
def test_configure_restores_v0_tables():
    G.configure(BASE.with_changes(switches={"test_family": True, "test_post": True, "test_gaps": True}))
    assert "test_sine" in G.FAMILIES and _hooks.installed()
    G.configure(BASE)
    assert not _hooks.installed()
    assert list(G.FAMILIES) == list(C.FAMILIES) and G.PASSTHROUGH_FAMILIES == C.PASSTHROUGH_FAMILIES
    assert G.groups.GROUP_KINDS == C.GROUP_KINDS
    for name in ("v1", "clean", "clean2"):
        C.resolve_profile(dict(C.CONFIG))
        assert G.GEN_PROFILES[name] == C.GEN_PROFILES[name], name


def test_switch_then_v0_is_bit_exact():
    G.configure(STRESS_GC.with_changes(switches={"test_family": {"mode": "replace:ets", "weight": 0.01}, "test_post": True}))
    G.generate_blocks(4, 8, 64, "cpu", profile="clean2", group_prob=0.7)
    cfg = dict(C.CONFIG, **STRESS)
    a, sa = run(ref_blocks(cfg), 3)
    b, sb = run(port_blocks(STRESS_GC), 3)
    assert_same(a, b)
    assert torch.equal(sa, sb)


# ---------------------------------------------------------------------------------------------------- hooks that draw nothing
def _identity_switch(seen):
    def mark(tag, ret):
        def fn(*a):
            seen.add(tag)
            return ret(*a)
        return fn

    def install(p, inst):
        inst.add_hook("after_families", mark("H2", lambda x, ctx: x), "structure")
        inst.add_hook("after_structure", mark("H3", lambda x, ctx: x))
        inst.add_hook("final", mark("H4", lambda x, ctx: x), "gaps")
        inst.add_hook("after_groups", mark("H7", lambda x, r, l, ctx: (x, r, l)))
        inst.add_hook("final_blocks", mark("H8", lambda x, r, l, ctx: (x, r, l)), "gaps")
        for layer in _hooks.LAYERS:
            inst.override_layer(layer, mark(layer, lambda args, kwargs, out, ctx: out))
        inst.set_group_size(mark("H6", lambda g, kind, is_group, V, ctx: g), ("mix",))
    register(Switch("t_identity", "TEST", "identity at every hook point", install))


@pytest.mark.parametrize("L", [64, 512, 2048])
def test_identity_hooks_keep_v0_bit_exact(L):
    seen = set()
    _identity_switch(seen)
    gc = STRESS_GC.with_changes(switches={"t_identity": True}, gen_additions={"breaks": True})
    cfg = dict(C.CONFIG, **dict(STRESS, gen_additions=dict(STRESS["gen_additions"], breaks=True)))
    for seed in (0, 1):
        a, sa = run(ref_blocks(cfg, L=L), seed)
        b, sb = run(port_blocks(gc, L=L), seed)
        assert _hooks.installed()
        assert_same(a, b)
        assert torch.equal(sa, sb)
        assert seen == {"H2", "H3", "H4", "H6", "H7", "H8", *_hooks.LAYERS}, seen
        C.resolve_profile(cfg)
        a, sa = run(lambda: C.generate_batch(256, L, "cpu", profile="clean2"), seed)
        b, sb = run(lambda: G.generate_batch(256, L, "cpu", profile="clean2"), seed)
        assert_same(a, b)
        assert torch.equal(sa, sb)


# ---------------------------------------------------------------------------------------------------- post-processing switch
@pytest.mark.parametrize("point", ["final_blocks", "after_groups"])
def test_post_switch_changes_only_what_it_touches(point):
    off, s_off = run(port_blocks(STRESS_GC), 5)
    on, s_on = run(port_blocks(STRESS_GC.with_changes(switches={"test_post": {"point": point, "p": 0.5}})), 5)
    assert torch.equal(s_off, s_on), "the switch changed torch's global RNG stream"
    d = (on[0] - off[0])
    hit = (d != 0).flatten(1).any(1)
    assert 0 < int(hit.sum()) < hit.numel()
    assert torch.equal(on[0][~hit], off[0][~hit])
    # exact: the hook computes x + shift * hit in the series' dtype, so the same expression on the off output must match bit for
    # bit. (on - off) itself is NOT exactly 1 in float32 for |x| >~ 100: (x + 1) - x is off by up to one ulp of x, which is
    # what failed on Perlmutter with torch.allclose (2026-10-27).
    expect = off[0] + 1.0 * hit.view(-1, 1, 1).to(off[0].dtype)
    if not torch.equal(on[0], expect):
        k = int((on[0] - expect).abs().flatten().argmax())
        raise AssertionError(f"worst element {k}: on {on[0].flatten()[k].item()!r}, off {off[0].flatten()[k].item()!r}, "
                             f"expected {expect.flatten()[k].item()!r}")
    assert torch.equal(on[1], off[1]) and torch.equal(on[2], off[2])


def test_post_switch_in_generate_batch():
    G.configure(BASE)
    off, s_off = run(lambda: G.generate_batch(256, 256, "cpu", profile="clean2"), 2)
    G.configure(BASE.with_changes(switches={"test_post": {"point": "final"}}))
    on, s_on = run(lambda: G.generate_batch(256, 256, "cpu", profile="clean2"), 2)
    assert torch.equal(s_off, s_on)
    # H4 runs before the row shuffle: same multiset of rows, a quarter of them shifted by 1
    hit = ~(on.unsqueeze(1) == off.unsqueeze(0)).all(-1).any(1)
    assert 0 < int(hit.sum()) < 256


def test_switch_rng_depends_on_seed_and_stream_and_call():
    gc = BASE.with_changes(switches={"test_post": {"p": 0.5}})

    def hits(seed, stream, calls=1):
        G.configure(gc, seed=seed, stream=stream)
        torch.manual_seed(0)
        out = [G.generate_blocks(64, 1, 32, "cpu", profile="clean2")[0] for _ in range(calls)]
        torch.manual_seed(0)
        G.configure(BASE)
        ref = [G.generate_blocks(64, 1, 32, "cpu", profile="clean2")[0] for _ in range(calls)]
        return [tuple((o != r).flatten(1).any(1).tolist()) for o, r in zip(out, ref)]

    assert hits(0, 0) == hits(0, 0)
    assert hits(0, 0) != hits(1, 0) and hits(0, 0) != hits(0, 1)
    two = hits(0, 0, calls=2)
    assert two[0] == hits(0, 0)[0] and two[0] != two[1]           # the call counter moves the stream on


# ---------------------------------------------------------------------------------------------------- RNG and NaN rules
def test_hook_drawing_from_global_rng_fails():
    register(Switch("t_global", "TEST", "", lambda p, inst: inst.add_hook("final", lambda x, ctx: x + torch.rand_like(x))))
    G.configure(BASE.with_changes(switches={"t_global": True}))
    with pytest.raises(RuntimeError, match="global RNG"):
        G.generate_batch(16, 32, "cpu", profile="clean2")


@pytest.mark.parametrize("point", ["after_families", "after_structure", "final", "after_groups"])
def test_nan_before_final_blocks_fails(point):
    blk = point == "after_groups"
    fn = (lambda x, r, l, ctx: (x * float("nan"), r, l)) if blk else (lambda x, ctx: x * float("nan"))
    register(Switch("t_nan", "TEST", "", lambda p, inst: inst.add_hook(point, fn)))
    with pytest.raises(RuntimeError, match="NaN"):
        port_blocks(STRESS_GC.with_changes(switches={"t_nan": True}), N=4, L=64)()


def test_nan_at_final_blocks_is_allowed():
    x, role, lgid = port_blocks(STRESS_GC.with_changes(switches={"test_gaps": {"p": 0.1}}), N=8, L=128)()
    share = torch.isnan(x).float().mean().item()
    assert 0.05 < share < 0.15


def test_priorities_order_and_hash():
    order = []

    def mk(tag, prio):
        def install(p, inst):
            inst.add_hook("final", lambda x, ctx: (order.append(tag), x)[1], prio)
        register(Switch(f"t_{tag}", "TEST", "", install))
    mk("a", "gaps")
    mk("b", "structure")
    mk("c", "structure")
    G.configure(BASE.with_changes(switches={"t_a": True, "t_b": True, "t_c": True}))
    G.generate_batch(8, 32, "cpu", profile="clean2")
    assert order == ["b", "c", "a"]
    plan = switches.plan({"t_a": True, "t_b": True})
    assert [(e[1], e[2]) for e in plan if e[0] == "hook"] == [("final", 300), ("final", 100)]


# ---------------------------------------------------------------------------------------------------- families
def test_add_family():
    G.configure(BASE.with_changes(switches={"test_family": {"weight": 0.1}}))
    w = G.GEN_PROFILES["clean2"]["weights"]
    C.resolve_profile(dict(C.CONFIG))
    assert w == dict(C.GEN_PROFILES["clean2"]["weights"], test_sine=0.1)
    torch.manual_seed(0)
    x = G.generate_batch(512, 64, "cpu", profile="clean2")
    assert torch.isfinite(x).all()


def test_replace_family_keeps_the_other_counts():
    C.resolve_profile(dict(C.CONFIG))
    w0 = C.GEN_PROFILES["clean2"]["weights"]
    G.configure(BASE.with_changes(switches={"test_family": {"weight": w0["ets"] / 2, "mode": "replace:ets"}}))
    w = G.GEN_PROFILES["clean2"]["weights"]
    assert list(w) == [k for f in w0 for k in ((f, "test_sine") if f == "ets" else (f,))]
    assert abs(w["ets"] + w["test_sine"] - w0["ets"]) < 1e-12 and sum(w.values()) == pytest.approx(sum(w0.values()))
    a, b = dict(C._allocate(4096, w0)), dict(G._allocate(4096, w))
    assert all(a[k] == b[k] for k in a if k != "ets") and a["ets"] == b["ets"] + b["test_sine"]


def test_replace_family_needs_enough_weight():
    with pytest.raises(ValueError, match="replace:ets"):
        G.configure(BASE.with_changes(switches={"test_family": {"weight": 0.99, "mode": "replace:ets"}}))
    with pytest.raises(ValueError, match="mode"):
        G.configure(BASE.with_changes(switches={"test_family": {"mode": "swap"}}))


def test_group_kind_and_size_hooks():
    def install(p, inst):
        inst.add_group_kind("t_copy", lambda z, act, mu, sd, ctx: (z * sd + mu, torch.zeros(z.shape[:2], dtype=torch.long)), 0.5)
        inst.set_group_size(lambda g, kind, is_group, V, ctx: torch.full_like(g, 2), ("t_copy",))
    register(Switch("t_kind", "TEST", "", install))
    x, role, lgid = port_blocks(STRESS_GC.with_changes(switches={"t_kind": True}), N=32)()
    assert G.groups.GROUP_KINDS[-1] == "t_copy" and len(G.groups.GROUP_KIND_PROBS) == 6
    assert torch.isfinite(x).all()
    G.configure(BASE)
    assert G.groups.GROUP_KINDS == C.GROUP_KINDS and len(G.groups.GROUP_KIND_PROBS) == 5


def test_bad_group_size_fails():
    register(Switch("t_size", "TEST", "", lambda p, inst: inst.set_group_size(lambda g, k, i, V, ctx: g * 0 + V + 1, ("mix",))))
    with pytest.raises(RuntimeError, match="group sizes"):
        port_blocks(STRESS_GC.with_changes(switches={"t_size": True}), N=4, L=64)()


def test_one_override_per_layer():
    register(Switch("t_o1", "TEST", "", lambda p, inst: inst.override_layer("noise", lambda a, k, o, c: o)))
    register(Switch("t_o2", "TEST", "", lambda p, inst: inst.override_layer("noise", lambda a, k, o, c: o)))
    with pytest.raises(ValueError, match="already overridden"):
        G.configure(BASE.with_changes(switches={"t_o1": True, "t_o2": True}))


# ---------------------------------------------------------------------------------------------------- config, hash, run id
def test_switch_params_hash_and_run_id():
    a = BASE.with_changes(switches={"test_post": True})
    b = BASE.with_changes(switches={"test_post": {"p": 0.3}})
    assert dict(dict(a.switches)["test_post"])["p"] == 0.25          # defaults are filled in
    assert a.config_hash() != b.config_hash() != BASE.config_hash()
    assert a.run_id_fragment(BASE).startswith("v0-test_post-") and BASE.run_id_fragment(a).startswith("v0-notest_post-")
    assert BASE.with_changes(switches={"test_post": False}) == BASE
    c = BASE.with_changes(switches={"test_post": {"priority": "gaps"}})
    assert c.config_hash() != a.config_hash()                       # priorities are in the plan
    with pytest.raises(KeyError, match="no parameter"):
        BASE.with_changes(switches={"test_post": {"nope": 1}})


# ---------------------------------------------------------------------------------------------------- workers
CHILD = """
import sys, torch
sys.path[:0] = [{repo!r}, {tp!r}]
import tideforge.gen as G
from tideforge.gen.config import GenConfig
import pickle
gc = pickle.loads(bytes.fromhex({gc!r}))
G.WorkerInit(gc, seed=7, rank=0, workers_per_rank=2)(1)
torch.manual_seed(3)
torch.save(G.generate_blocks(8, 8, 64, "cpu", profile="clean2", group_prob=0.7), {out!r})
"""


def test_worker_process_reproduces_the_parent(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    gc = STRESS_GC.with_changes(switches={"test_post": True, "test_family": True})
    w = pickle.loads(pickle.dumps(G.WorkerInit(gc, seed=7, rank=0, workers_per_rank=2)))
    w(1)
    torch.manual_seed(3)
    mine = G.generate_blocks(8, 8, 64, "cpu", profile="clean2", group_prob=0.7)
    out = tmp_path / "child.pt"
    code = CHILD.format(repo=str(repo), tp=str(repo / "third_party" / "tidemark_e21cbf0"), gc=pickle.dumps(gc).hex(),
                        out=str(out))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert_same(mine, torch.load(out))


# ---------------------------------------------------------------------------------------------------- GPU (Perlmutter)
@pytest.mark.gpu
@pytest.mark.parametrize("point", ["after_families", "after_structure"])
def test_gpu_h2_h3_keep_the_other_draws(point):
    """On CUDA, does a value-changing hook before the noise / support layers (torch.poisson) leave the other global draws
    unchanged? If so, the rows the switch did not touch are bit-identical and the CUDA RNG state matches v0's. The result
    is recorded in tideforge/gen/switches.py (on CPU it does not hold: poisson's draw count depends on the values)."""
    dev = torch.device("cuda")
    gc = STRESS_GC.with_changes(gen_additions={"breaks": True})

    def run_on(g):
        name = G.configure(g)
        torch.manual_seed(11)
        x = G.generate_batch(512, 1024, dev, profile=name)
        return x, torch.cuda.get_rng_state(dev), torch.get_rng_state()

    off, c_off, s_off = run_on(gc)
    on, c_on, s_on = run_on(gc.with_changes(switches={"test_post": {"point": point, "p": 0.5}}))
    assert torch.equal(c_off, c_on) and torch.equal(s_off, s_on), "the global RNG streams differ after the call"
    same = (on.unsqueeze(1) == off.unsqueeze(0)).all(-1).any(1)            # rows of `on` found unchanged in `off`
    # about half the rows are untouched (p = 0.5) and must be found unchanged; touched rows can also come out unchanged,
    # because _L_support standardises each row and removes a constant shift
    assert same.float().mean().item() >= 0.4, f"only {same.float().mean().item():.2f} of the rows unchanged"
