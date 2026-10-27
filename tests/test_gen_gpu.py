"""GPU diagnostics for E0.1 (Perlmutter, 2026-10-27: test_gpu_parity failed with 1-2 float32 ulp differences).

1. test_gpu_ft3_self_repro: is ft3_core's own generate_batch reproducible on CUDA (same seed, run twice)? Run in a fresh
   process with and without torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8. Prints one line per L.
2. test_gpu_layer_diff: ft3_core vs the port, layer by layer, each layer fed the SAME input and started from the SAME RNG
   state on both sides; reports the first family / layer whose output differs (needed if 1 shows ft3_core IS reproducible).
3. test_gpu_parity_ulp4: the tolerant GPU rule (used if 1 shows ft3_core is NOT reproducible against itself): discrete outputs
   exact (NaN positions, integer-valued rows, roles, group ids), RNG state after the call exact, float values within 4 ulp.
Run: python -m pytest -q -s -m gpu tests/test_gen_gpu.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import tidemark.real.ft3_core as C
import tideforge.gen as G
from tideforge.gen import batch as GB
from tideforge.gen import layers as GL
from tideforge.gen.config import GenConfig

pytestmark = pytest.mark.gpu
REPO = Path(__file__).resolve().parents[1]
BUCKETS = [16, 64, 256, 1024, 4096, 8192]
B = 256


def ulp_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distance in float32 units in the last place (NaN-NaN = 0; NaN vs number = huge)."""
    def ordered(x):
        i = x.float().contiguous().view(torch.int32).long()
        return torch.where(i < 0, -(i & 0x7FFFFFFF), i)
    d = (ordered(a) - ordered(b)).abs()
    both_nan = torch.isnan(a) & torch.isnan(b)
    return torch.where(both_nan, torch.zeros_like(d), d)


SELF = r"""
import json, os, sys, torch
sys.path[:0] = [{repo!r}, {tp!r}]
det = {det}
if det:
    torch.use_deterministic_algorithms(True)
import tidemark.real.ft3_core as C
C.set_group_options(None)
name = C.resolve_profile(dict(C.CONFIG))
dev = torch.device("cuda")
out = []
for L in {buckets}:
    r = []
    for _ in range(2):
        torch.manual_seed(0)
        try:
            r.append(C.generate_batch({B}, L, dev, profile=name).cpu())
        except RuntimeError as e:
            out.append(dict(L=L, error=str(e)[:200])); break
    if len(r) == 2:
        a, b = r
        i = lambda x: x.contiguous().view(torch.int32).long()
        o = lambda x: torch.where(i(x) < 0, -(i(x) & 0x7FFFFFFF), i(x))
        u = (o(a) - o(b)).abs()
        out.append(dict(L=L, equal=bool(torch.equal(a, b)), max_abs=float((a - b).abs().nan_to_num().max()),
                        max_ulp=int(u.max()), n_diff=int((a != b).sum())))
print("RESULT " + json.dumps(out))
"""


@pytest.mark.parametrize("det", [False, True])
def test_gpu_ft3_self_repro(det):
    env = dict(os.environ)
    if det:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    code = SELF.format(repo=str(REPO), tp=str(REPO / "third_party" / "tidemark_e21cbf0"), det=det, buckets=BUCKETS, B=B)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(r.stdout.split("RESULT ", 1)[1])
    for row in res:
        print(f"ft3_core self-repro deterministic={det}: {row}")
    bad = [row for row in res if not row.get("equal", False)]
    assert not bad, f"ft3_core is not reproducible against itself on CUDA (deterministic={det}): {bad}"


def test_gpu_layer_diff():
    """First family / layer where the port differs from ft3_core on CUDA, with the same input and RNG state."""
    dev = torch.device("cuda")
    C.set_group_options(None)
    name = C.resolve_profile(dict(C.CONFIG))
    G.configure(GenConfig())
    pr = C.GEN_PROFILES[name]
    C._PERIOD_BANK_ACTIVE = pr.get("period_bank", C.PERIOD_BANK)
    C._PERIOD_EXT_SHARE = pr.get("period_ext_share", None)
    C._TV_SEASON, C._ONE_OFF_P = bool(pr.get("tv_season", False)), float(pr.get("one_off_p", 0.0))
    from tideforge.gen import _state
    _state._PERIOD_BANK_ACTIVE, _state._PERIOD_EXT_SHARE = C._PERIOD_BANK_ACTIVE, C._PERIOD_EXT_SHARE
    _state._TV_SEASON, _state._ONE_OFF_P = C._TV_SEASON, C._ONE_OFF_P
    report = []

    def both(label, fc, fg):
        st = (torch.get_rng_state(), torch.cuda.get_rng_state())
        a = fc()
        torch.set_rng_state(st[0]); torch.cuda.set_rng_state(st[1])
        b = fg()
        a, b = (a if isinstance(a, tuple) else (a,)), (b if isinstance(b, tuple) else (b,))
        worst = max(int(ulp_dist(x, y).max()) if x.is_floating_point() else int((x != y).sum()) for x, y in zip(a, b))
        report.append((label, worst))
        return a

    for L in (64, 1024):
        torch.manual_seed(0)
        for fam in C.FAMILIES:
            both(f"family {fam} L={L}", lambda: C.FAMILIES[fam][0](32, L, dev), lambda: G.FAMILIES[fam][0](32, L, dev))
        torch.manual_seed(1)
        x = C.generate_batch(B, L, dev, apply_layers=False, profile=name)
        n = x.shape[0]
        flags = lambda p: torch.rand(n, device=dev) < p
        allow, cnt, nn = flags(0.8), flags(0.1), flags(0.3)
        steps = [("seasonality", lambda m, x: m._L_seasonality(x, allow, dev)), ("trend", lambda m, x: m._L_trend(x, allow, dev)),
                 ("calendar", lambda m, x: m._L_calendar(x, allow, dev)), ("noise", lambda m, x: m._L_noise(x, cnt, dev)),
                 ("support", lambda m, x: m._L_support(x, cnt, nn, dev, pr["support_probs"])[0]),
                 ("corruption", lambda m, x: m._L_corruption(x, cnt, nn, dev, pr["out_p"], pr["miss_p"], pr["quant_p"],
                                                             pr["quant_step"], pr["clean_p"])),
                 ("scale", lambda m, x: m._L_scale(x, cnt, nn, dev)), ("finalize", lambda m, x: m._finalize(x, pr["fix_degenerate"]))]
        for label, f in steps:
            x = both(f"layer {label} L={L}", lambda: f(C, x), lambda: f(GL, x))[0]
    del GB
    for label, worst in report:
        print(f"{label:40s} max diff (ulp or count) {worst}")
    first = next(((lab, w) for lab, w in report if w), None)
    assert first is None, f"first difference: {first}"


@pytest.mark.parametrize("L", BUCKETS)
def test_gpu_parity_ulp4(L):
    """The tolerant rule (documented in experiments/E0.1.md if ft3_core is not reproducible against itself)."""
    dev = torch.device("cuda")
    cfg_path = REPO / "configs" / "ft3_main_cfg.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else dict(C.CONFIG)
    gc = GenConfig.from_ft3_cfg(cfg)
    for seed in (0, 1, 2):
        C.set_group_options(cfg)
        name = C.resolve_profile(cfg)
        torch.manual_seed(seed)
        a = C.generate_blocks(B // 8, 8, L, dev, profile=name, group_prob=gc.group_prob)
        sa = (torch.get_rng_state(), torch.cuda.get_rng_state())
        assert G.configure(gc) == name
        torch.manual_seed(seed)
        b = G.generate_blocks(B // 8, 8, L, dev, profile=name, group_prob=gc.group_prob)
        sb = (torch.get_rng_state(), torch.cuda.get_rng_state())
        assert all(torch.equal(x, y) for x, y in zip(sa, sb)), "RNG state differs after the call"
        assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]), "roles / group ids differ"
        xa, xb = a[0], b[0]
        assert torch.equal(torch.isnan(xa), torch.isnan(xb)), "NaN positions differ"
        integral = (xa == xa.round()).flatten(2).all(2)
        assert torch.equal(xa[integral], xb[integral]), "integer-valued series differ"
        u = int(ulp_dist(xa, xb).max())
        print(f"L={L} seed={seed}: max ulp {u}, differing values {int((xa != xb).sum())} of {xa.numel()}")
        assert u <= 4, f"max ulp {u} > 4"
