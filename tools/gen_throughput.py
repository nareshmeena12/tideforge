"""E0.9: generator throughput on GPU, torch's default mode vs deterministic mode (each in a fresh process).

    python tools/gen_throughput.py                      # B0, blocks of 8 lanes, L = 1024 / 4096 / 8192 (+ pred 128)

Prints points per second of tideforge.gen.generate_blocks (B0 = configs/ft3_main_cfg.json) per length and mode, and the
ratio deterministic / default. The owner decides from these numbers whether training uses deterministic mode.

    python tools/gen_throughput.py --device cpu --exp G4.1 G6.2 ...     # per experiment config vs B0 (torch's default mode)
    python tools/gen_throughput.py --device cuda --exp G4.1 G6.2 ...    # the same on GPU (Perlmutter)
    python tools/gen_throughput.py --device cpu --all                   # every configs/exp/G*.yaml
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
    o = ap.parse_args(argv)
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
            res = run(False, e, o.device, o.warm)
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
