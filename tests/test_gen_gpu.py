"""GPU diagnostics for E0.1. Result of 2026-10-27 (Perlmutter A100, torch 2.13.0+cu130): ft3_core is NOT reproducible
against itself on CUDA in torch's default mode (1-2749 values differ by 1-10 ulp per length) and IS bit-reproducible with
torch.use_deterministic_algorithms(True) + CUBLAS_WORKSPACE_CONFIG=:4096:8. The E0.1 GPU rule (owner) is therefore bit-exact
parity of ft3_core and the port, both in deterministic mode (tests/conftest.py puts every gpu test in that mode).

1. test_gpu_ft3_self_repro: ft3_core's generate_batch twice, same seed, in a fresh process, default vs deterministic mode
   (default mode is an expected failure, xfail strict: it documents ft3_core's own noise).
2. test_gpu_nondet_sources: in default mode, every family, layer, group building and the full batch / blocks of ft3_core run
   twice from the same input and RNG state; prints which ones differ. Reports only. 2026-10-27: only the aggregated family
   differs (12,005 values at 256 x 1024; its scatter_add_, ft3_core L1394); every other family, every layer and group
   building reproduce exactly.
3. test_gpu_layer_diff: ft3_core vs the port, layer by layer, same input and RNG state, deterministic mode: names the first
   family / layer that differs if test_gpu_parity fails.
Run: python -m pytest -q -s -rA -m gpu tests/test_gen_gpu.py tests/test_gen_parity.py tests/test_gen_switches.py
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


@pytest.mark.parametrize("det", [
    pytest.param(False, marks=pytest.mark.xfail(strict=True, reason="documents ft3_core's own run-to-run noise on CUDA in "
                                                                     "torch's default mode (the aggregated family's scatter_add_)")),
    True])
def test_gpu_ft3_self_repro(det):
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
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
    for label, worst in report:
        print(f"{label:40s} max diff (ulp or count) {worst}")
    first = next(((lab, w) for lab, w in report if w), None)
    assert first is None, f"first difference: {first}"


NONDET = r"""
import json, sys, torch
sys.path[:0] = [{repo!r}, {tp!r}]
import tidemark.real.ft3_core as C
torch.use_deterministic_algorithms(False)
C.set_group_options(None)
name = C.resolve_profile(dict(C.CONFIG, gen_additions=dict(C.CONFIG["gen_additions"], sarima=True, bursty=True)))
pr = C.GEN_PROFILES[name]
C._PERIOD_BANK_ACTIVE = pr.get("period_bank", C.PERIOD_BANK)
C._PERIOD_EXT_SHARE = pr.get("period_ext_share", None)
dev = torch.device("cuda")
out = {{}}

def twice(label, f):
    st = (torch.get_rng_state(), torch.cuda.get_rng_state())
    a = f()
    torch.set_rng_state(st[0]); torch.cuda.set_rng_state(st[1])
    b = f()
    a = a[0] if isinstance(a, tuple) else a
    b = b[0] if isinstance(b, tuple) else b
    out[label] = int((a != b).sum()) if a.shape == b.shape else -1

L = 1024
torch.manual_seed(0)
for fam in C.FAMILIES:
    twice("family " + fam, lambda: C.FAMILIES[fam][0](256, L, dev))
x = C.generate_batch(256, L, dev, apply_layers=False, profile=name)
n = x.shape[0]
allow, cnt, nn = (torch.rand(n, device=dev) < p for p in (0.8, 0.1, 0.3))
for label, f in [("seasonality", lambda: C._L_seasonality(x, allow, dev)), ("trend", lambda: C._L_trend(x, allow, dev)),
                 ("calendar", lambda: C._L_calendar(x, allow, dev)), ("breaks", lambda: C._L_breaks(x, dev, 0.2)),
                 ("noise", lambda: C._L_noise(x, cnt, dev)), ("support", lambda: C._L_support(x, cnt, nn, dev, pr["support_probs"])),
                 ("corruption", lambda: C._L_corruption(x, cnt, nn, dev, pr["out_p"], pr["miss_p"], pr["quant_p"],
                                                        pr["quant_step"], pr["clean_p"])),
                 ("scale", lambda: C._L_scale(x, cnt, nn, dev)), ("finalize", lambda: C._finalize(x, pr["fix_degenerate"]))]:
    twice("layer " + label, f)
C.set_group_options(dict(C.CONFIG, cov_r3=True, second_group_p=0.5))
twice("groups _multivariatize", lambda: C._multivariatize(x[:256].view(32, 8, L).clone(), 0.7, dev))
for Lf in (256, 1024):
    twice(f"full generate_batch L={{Lf}}", lambda: C.generate_batch(256, Lf, dev, profile=name))
    twice(f"full generate_blocks L={{Lf}}", lambda: C.generate_blocks(32, 8, Lf, dev, profile=name, group_prob=0.7))
print("RESULT " + json.dumps(out))
"""


def test_gpu_nondet_sources():
    """Which families / layers of ft3_core differ between two identical runs in torch's DEFAULT mode (reports only)."""
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    code = NONDET.format(repo=str(REPO), tp=str(REPO / "third_party" / "tidemark_e21cbf0"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(r.stdout.split("RESULT ", 1)[1])
    for k, v in res.items():
        print(f"default-mode run-to-run differences: {k:40s} {v}")
