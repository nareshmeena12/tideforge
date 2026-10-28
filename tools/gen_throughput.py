"""E0.9: generator throughput on GPU, torch's default mode vs deterministic mode (each in a fresh process).

    python tools/gen_throughput.py                      # B0, blocks of 8 lanes, L = 1024 / 4096 / 8192 (+ pred 128)

Prints points per second of tideforge.gen.generate_blocks (B0 = configs/ft3_main_cfg.json) per length and mode, and the
ratio deterministic / default. The owner decides from these numbers whether training uses deterministic mode.

    python tools/gen_throughput.py --device cpu --exp G4.1 G6.2 ...     # per experiment config vs B0 (torch's default mode)
    python tools/gen_throughput.py --device cuda --exp G4.1 G6.2 ...    # the same on GPU (Perlmutter)
    python tools/gen_throughput.py --device cpu --all                   # every configs/exp/G*.yaml
    python tools/gen_throughput.py --device cuda --loops                # tar / garch_jump loop cost vs B0 (owner 2026-10-28)
    python tools/gen_throughput.py --device cuda --loops --pool-dir $TF_DATA/pools     # also their offline-pool path
--loops: per length, the time of the family function alone on its add-mode share of a pool's rows (tar 3%, garch_jump 4%),
as a fraction of B0's generate_blocks time for the whole pool. Above 0.10 the family switches to its offline pool
(tools/build_loop_pool.py; switch param pool). The pool rows need the pools built and their manifests committed.
Each config runs in its own process (fresh module state), same lengths as above; CPU uses a smaller pool (estimate:
~1/16 of a GPU pool) so it finishes in minutes. Ratios are what to compare; CPU and GPU absolute numbers are not comparable.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHILD = r"""
import json, sys, time, torch
sys.path[:0] = [{repo!r}, {tp!r}]
if {det}:
    torch.use_deterministic_algorithms(True)
import tideforge.gen as G
from tideforge.expcfg import load_exp
gc = load_exp({exp!r}).gen if {exp!r} else G.b0()
name = G.configure(gc)
dev = torch.device({device!r})
sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
out = []
for L in (1024, 4096, 8192):
    Lb = L + 128
    N = max(8, int({pool} // (Lb * 8)))            # about one generation pool (gen_pool_points 16M on GPU)
    for _ in range({warm}):
        G.generate_blocks(N, 8, Lb, dev, profile=name, group_prob=gc.group_prob)
    sync()
    t0, reps = time.time(), {reps}
    for _ in range(reps):
        G.generate_blocks(N, 8, Lb, dev, profile=name, group_prob=gc.group_prob)
    sync()
    dt = (time.time() - t0) / reps
    out.append(dict(L=L, blocks=N, points_per_s=N * 8 * Lb / dt, seconds_per_pool=dt))
print("RESULT " + json.dumps(out))
"""


LOOP_CHILD = r"""
import json, sys, time, torch
sys.path[:0] = [{repo!r}, {tp!r}]
import tideforge.gen as G
from tideforge.gen import switches
from tideforge.gen.ext import g9_tar, g10_finance
gc = G.b0()
name = G.configure(gc)
dev = torch.device({device!r})
sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
fams = dict(tar=g9_tar.make_tar, garch_jump=g10_finance.make_garch)


def timed(fn, reps):
    for _ in range({warm}):
        fn()
    sync()
    t0 = time.time()
    for _ in range(reps):
        fn()
    sync()
    return (time.time() - t0) / reps


out = []
for L in (1024, 4096, 8192):
    Lb = L + 128
    N = max(8, int({pool} // (Lb * 8)))
    G.configure(gc)
    tb = timed(lambda: G.generate_blocks(N, 8, Lb, dev, profile=name, group_prob=gc.group_prob), {reps})
    row = dict(L=L, b0_s=tb)
    for fam, make in fams.items():
        p = dict(switches.REGISTRY[fam].params)
        n = max(1, round(p["share"] * N * 8))
        g = torch.Generator(device=dev).manual_seed(0)
        row[fam] = timed(lambda: make(p)(n, Lb, dev), {reps}) / tb
        if {pool_dir!r}:
            pp = dict(p, pool=fam + "_v1", pool_dir={pool_dir!r}, manifest_dir={manifest_dir!r})
            try:
                row[fam + "_pool"] = timed(lambda: make(pp)(n, Lb, dev), {reps}) / tb
            except Exception as e:
                row[fam + "_pool"] = repr(e)[:80]
    out.append(row)
print("RESULT " + json.dumps(out))
"""


def loops(device: str, pool_dir: str = "", warm: int | None = None, manifest_dir: str = "") -> list[dict]:
    cpu = device == "cpu"
    code = LOOP_CHILD.format(repo=str(REPO), tp=str(REPO / "third_party" / "tidemark_e21cbf0"), device=device,
                             pool=1e6 if cpu else 16e6, warm=warm if warm is not None else (1 if cpu else 2),
                             reps=3 if cpu else 5, pool_dir=pool_dir, manifest_dir=manifest_dir)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr[-3000:])
    return json.loads(r.stdout.split("RESULT ", 1)[1])


def run(det: bool, exp: str = "", device: str = "cuda", warm: int | None = None) -> list[dict]:
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    if det:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    cpu = device == "cpu"
    code = CHILD.format(repo=str(REPO), tp=str(REPO / "third_party" / "tidemark_e21cbf0"), det=det, exp=exp, device=device,
                        pool=1e6 if cpu else 16e6, warm=warm if warm is not None else (1 if cpu else 2), reps=6 if cpu else 5)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(r.stderr[-3000:])
    return json.loads(r.stdout.split("RESULT ", 1)[1])


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--exp", nargs="*", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--warm", type=int, default=None, help="warm-up calls per length before timing (default 1 CPU / 2 GPU)")
    ap.add_argument("--loops", action="store_true")
    ap.add_argument("--pool-dir", default="")
    ap.add_argument("--manifest-dir", default="", help="default: the committed manifests (tests: a tiny pool's own)")
    o = ap.parse_args(argv)
    if o.loops:
        res = loops(o.device, o.pool_dir, o.warm, o.manifest_dir)
        print(f"{o.device}: family time on its add share / B0 generate_blocks time (switch to the pool above 0.10)")
        keys = [k for k in res[0] if k not in ("L", "b0_s")]
        print(f"{'L':>6s} {'B0 s/pool':>10s} " + " ".join(f"{k:>16s}" for k in keys))
        for r in res:
            print(f"{r['L']:6d} {r['b0_s']:10.3f} " + " ".join(
                f"{r[k]:16.3f}" if isinstance(r[k], float) else f"{r[k][:16]:>16s}" for k in keys))
        return 0
    exps = list(o.exp)
    if o.all:
        from tideforge.expcfg import EXP_DIR, load_exp
        sys.path[:0] = [str(REPO), str(REPO / "third_party" / "tidemark_e21cbf0")]
        exps = [p.stem for p in sorted(EXP_DIR.glob("G*.yaml")) if load_exp(p).gen != load_exp(p).base]
    if exps:
        ref = run(False, "", o.device, o.warm)
        print(f"{o.device}, torch default mode; points/s and ratio vs the B0 run made right after each config "
              f"(not the first B0 row)")
        print(f"{'config':8s} " + " ".join(f"{'L' + str(r['L']):>20s}" for r in ref))
        print(f"{'B0':8s} " + " ".join(f"{r['points_per_s']:12.3e} (1.000)" for r in ref))
        for e in exps:
            try:
                res = run(False, e, o.device, o.warm)
            except SystemExit as err:                      # e.g. G9.1 without its pool: say so, go on
                print(f"{e:8s} failed: {str(err).strip().splitlines()[-1][:150]}", flush=True)
                continue
            ref = run(False, "", o.device, o.warm)         # B0 again right after: the ratio uses the adjacent B0 (machine load drifts)
            print(f"{e:8s} " + " ".join(f"{x['points_per_s']:12.3e} ({x['points_per_s'] / r['points_per_s']:.3f})"
                                       for x, r in zip(res, ref)), flush=True)
        return 0
    a, b = run(False), run(True)
    print(f"{'L':>6s} {'default pts/s':>15s} {'determ. pts/s':>15s} {'determ./default':>16s}")
    for x, y in zip(a, b):
        print(f"{x['L']:6d} {x['points_per_s']:15.3e} {y['points_per_s']:15.3e} {y['points_per_s'] / x['points_per_s']:16.3f}")
    return 0


if __name__ == "__main__":
    sys.path[:0] = [str(REPO), str(REPO / "third_party" / "tidemark_e21cbf0")]
    sys.exit(main())
