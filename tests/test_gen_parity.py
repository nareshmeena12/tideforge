"""E0.1: generator v0 (tideforge.gen) reproduces ft3_core's generator bit for bit.

Both sides start from the same torch.manual_seed. Every case checks, for every output: torch.equal, dtype and device; and that the
global RNG state after the call is identical (an extra or missing random draw fails even when the outputs happen to agree);
and that v0 never outputs NaN / inf (missing values may only come from a switch at final_blocks, H8).
The ft3_core side is configured from the raw ft3 cfg dict (resolve_profile / set_group_options, as ft3 training does); the
tideforge side from GenConfig.from_ft3_cfg(cfg) through tideforge.gen.configure, so the mapping to GenConfig is tested too.
"""
import json

import pytest
import torch

import tidemark.real.ft3_core as C
import tideforge.gen as G
from tideforge.gen import _state
from tideforge.gen.config import FT3_MAIN_CFG, GenConfig

BUCKETS = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]     # ft3_core CONFIG len_bucket_edges
SEEDS = [0, 1, 2]
# (L, seed) grid of the per-call cases; seeds 1 and 2 at L >= 4096 are marked slow (pytest --runslow runs them)
GRID = [pytest.param(L, s, marks=pytest.mark.slow) if (L >= 4096 and s > 0) else (L, s) for L in BUCKETS for s in SEEDS]
B = 256          # rows per generate_batch call: every family of every mix gets rows at this size
V = 8            # lanes per block (CONFIG group_lanes)

STRESS = dict(gen_additions={k: True for k in C.GEN_ADDITIONS_ALL}, period_ext_share=0.3, cov_r3=True, cov_dominant_p=0.3,
              second_group_p=0.5, cov_small_group_p=0.4, group_kind_probs=[0.1, 0.1, 0.3, 0.3, 0.2])
CFGS = {"config_defaults": dict(C.CONFIG), "stress": dict(C.CONFIG, **STRESS)}

MISSING_FT3 = f"{FT3_MAIN_CFG} is missing (the owner provides it); E0.1 is not passed until this case runs"


def ft3_main_cfg():
    if not FT3_MAIN_CFG.exists():
        pytest.skip(MISSING_FT3)
    return json.loads(FT3_MAIN_CFG.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def restore_module_state():
    """Both generators keep their settings in module state; put both back after every test."""
    ref = (C._PERIOD_BANK_ACTIVE, C._PERIOD_EXT_SHARE, C._TV_SEASON, C._ONE_OFF_P, C.GEN_PROFILES["clean2"])
    port = (_state._PERIOD_BANK_ACTIVE, _state._PERIOD_EXT_SHARE, _state._TV_SEASON, _state._ONE_OFF_P,
            G.GEN_PROFILES["clean2"])
    yield
    C._PERIOD_BANK_ACTIVE, C._PERIOD_EXT_SHARE, C._TV_SEASON, C._ONE_OFF_P, C.GEN_PROFILES["clean2"] = ref
    (_state._PERIOD_BANK_ACTIVE, _state._PERIOD_EXT_SHARE, _state._TV_SEASON, _state._ONE_OFF_P,
     G.GEN_PROFILES["clean2"]) = port
    C.set_group_options(None)
    G.set_group_options(None)


# ---------------------------------------------------------------------------------------------------------------- helpers
def rng_state(device):
    st = [torch.get_rng_state()]
    if torch.device(device).type == "cuda":
        st.append(torch.cuda.get_rng_state())
    return st


def same(ref_fn, port_fn, seed, device="cpu"):
    """Run both from the same seed; assert identical outputs (value, dtype, device) and identical RNG state afterwards."""
    torch.manual_seed(seed)
    a = ref_fn()
    sa = rng_state(device)
    torch.manual_seed(seed)
    b = port_fn()
    sb = rng_state(device)
    a = a if isinstance(a, tuple) else (a,)
    b = b if isinstance(b, tuple) else (b,)
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.dtype == y.dtype, f"output {i}: dtype {x.dtype} vs {y.dtype}"
        assert x.device == y.device, f"output {i}: device {x.device} vs {y.device}"
        assert x.shape == y.shape, f"output {i}: shape {tuple(x.shape)} vs {tuple(y.shape)}"
        assert torch.equal(x, y), f"output {i}: values differ (max abs diff {(x.double() - y.double()).abs().max().item()})"
        if y.is_floating_point():
            assert bool(torch.isfinite(y).all()), f"output {i}: v0 produced NaN / inf"
    for i, (x, y) in enumerate(zip(sa, sb)):
        assert torch.equal(x, y), f"RNG state {i} differs after the call: different random draws"


def configure_both(cfg, holdout=False):
    """ft3_core from the raw cfg; tideforge through GenConfig. Returns (profile, group_prob) and checks the profiles agree."""
    gc = GenConfig.from_ft3_cfg(cfg)
    if holdout:
        C.set_group_options(None)
        name = C.resolve_profile(dict(cfg, corpus=cfg.get("val_corpus") or cfg.get("corpus", "v1")))
        gp = gc.val_group_prob
        assert G.configure(gc, holdout=True) == name
    else:
        C.set_group_options(cfg)
        name = C.resolve_profile(cfg)
        gp = gc.group_prob
        assert G.configure(gc) == name
    assert G.GEN_PROFILES[name] == C.GEN_PROFILES[name], "profile dicts differ"
    assert (G.groups.GROUP_KIND_PROBS, G.groups._COV_SMALL_GROUP_P, G.groups._COV_DOMINANT_P, G.groups._SECOND_GROUP_P,
            G.groups._COV_R3) == (C.GROUP_KIND_PROBS, C._COV_SMALL_GROUP_P, C._COV_DOMINANT_P, C._SECOND_GROUP_P, C._COV_R3)
    return name, gp


def set_profile_state(profile):
    """What generate_batch sets before calling the families, on both sides."""
    pr = C.GEN_PROFILES[profile]
    C._PERIOD_BANK_ACTIVE = _state._PERIOD_BANK_ACTIVE = pr.get("period_bank", C.PERIOD_BANK)
    C._PERIOD_EXT_SHARE = _state._PERIOD_EXT_SHARE = pr.get("period_ext_share", None)
    C._TV_SEASON = _state._TV_SEASON = bool(pr.get("tv_season", False))
    C._ONE_OFF_P = _state._ONE_OFF_P = float(pr.get("one_off_p", 0.0))


# ---------------------------------------------------------------------------------------------------------------- tables
def test_tables_match():
    assert list(G.FAMILIES) == list(C.FAMILIES)
    for k, v in C.FAMILIES.items():
        assert G.FAMILIES[k][1:] == v[1:], k
        assert G.FAMILIES[k][0].__name__ == v[0].__name__, k
    assert G.DEFAULT_WEIGHTS == C.DEFAULT_WEIGHTS and G.CLEAN_WEIGHTS == C.CLEAN_WEIGHTS
    assert G.PASSTHROUGH_FAMILIES == C.PASSTHROUGH_FAMILIES and G.GEN_ADDITIONS_ALL == C.GEN_ADDITIONS_ALL
    assert G.GROUP_KINDS == C.GROUP_KINDS
    assert (G.ROLE_TARGET, G.ROLE_PAST, G.ROLE_KNOWN) == (C.ROLE_TARGET, C.ROLE_PAST, C.ROLE_KNOWN)
    for name in ("v1", "clean"):
        assert G.GEN_PROFILES[name] == C.GEN_PROFILES[name]


# ---------------------------------------------------------------------------------------------------------------- 1. families
@pytest.mark.parametrize("cfg_name", list(CFGS))
@pytest.mark.parametrize("L", BUCKETS)
def test_every_family(cfg_name, L):
    name, _ = configure_both(CFGS[cfg_name])
    set_profile_state(name)
    n = 8
    for fam in C.FAMILIES:
        for seed in SEEDS:
            try:
                same(lambda: C.FAMILIES[fam][0](n, L, "cpu"), lambda: G.FAMILIES[fam][0](n, L, "cpu"), seed)
            except AssertionError as e:
                raise AssertionError(f"family {fam}, L={L}, seed {seed}: {e}") from None


# ---------------------------------------------------------------------------------------------------------------- 2. generate_batch
@pytest.mark.parametrize("profile", ["v1", "clean", "clean2:config_defaults", "clean2:stress"])
@pytest.mark.parametrize("L, seed", GRID)
def test_generate_batch(profile, L, seed):
    name = profile.split(":")[0]
    if ":" in profile:
        assert configure_both(CFGS[profile.split(":")[1]])[0] == name
    same(lambda: C.generate_batch(B, L, "cpu", profile=name), lambda: G.generate_batch(B, L, "cpu", profile=name), seed)


def test_generate_batch_raw_families_and_explicit_weights():
    configure_both(CFGS["config_defaults"])
    same(lambda: C.generate_batch(B, 512, "cpu", apply_layers=False, profile="clean2"),
         lambda: G.generate_batch(B, 512, "cpu", apply_layers=False, profile="clean2"), 0)
    w = {"ets": 1.0, "repeat": 1.0, "long_cycle": 1.0}
    same(lambda: C.generate_batch(64, 512, "cpu", weights=w, profile="clean2"),
         lambda: G.generate_batch(64, 512, "cpu", weights=w, profile="clean2"), 0)
    assert G._allocate(37, C.CLEAN_WEIGHTS) == C._allocate(37, C.CLEAN_WEIGHTS)


# ---------------------------------------------------------------------------------------------------------------- 3. generate_blocks
BLOCK_CASES = ["holdout", "config_defaults", "stress"]


def _blocks(case, cfg, L, seed, group_prob=None, device="cpu"):
    name, gp = configure_both(cfg, holdout=(case == "holdout"))
    gp = gp if group_prob is None else group_prob
    N = B // V
    same(lambda: C.generate_blocks(N, V, L, device, profile=name, group_prob=gp),
         lambda: G.generate_blocks(N, V, L, device, profile=name, group_prob=gp), seed, device)


@pytest.mark.parametrize("case", BLOCK_CASES)
@pytest.mark.parametrize("group_prob", [0.0, 0.7])
@pytest.mark.parametrize("L, seed", GRID)
def test_generate_blocks(case, group_prob, L, seed):
    cfg = CFGS["stress" if case == "stress" else "config_defaults"]
    _blocks(case, cfg, L, seed, group_prob)


# ---------------------------------------------------------------------------------------------------------------- 4. ft3_main's cfg
@pytest.mark.parametrize("L, seed", GRID)
def test_ft3_main_cfg(L, seed):
    cfg = ft3_main_cfg()
    name, _ = configure_both(cfg)
    same(lambda: C.generate_batch(B, L, "cpu", profile=name), lambda: G.generate_batch(B, L, "cpu", profile=name), seed)
    _blocks("train", cfg, L, seed)
    _blocks("holdout", cfg, L, seed)


# ---------------------------------------------------------------------------------------------------------------- 5. GPU (Perlmutter)
@pytest.mark.gpu
@pytest.mark.parametrize("L", BUCKETS)
def test_gpu_parity(L):
    cfg = json.loads(FT3_MAIN_CFG.read_text(encoding="utf-8")) if FT3_MAIN_CFG.exists() else dict(C.CONFIG)
    dev = torch.device("cuda")
    for seed in SEEDS:
        name, _ = configure_both(cfg)
        same(lambda: C.generate_batch(B, L, dev, profile=name), lambda: G.generate_batch(B, L, dev, profile=name), seed, "cuda")
        _blocks("train", cfg, L, seed, device=dev)
        _blocks("holdout", cfg, L, seed, device=dev)
