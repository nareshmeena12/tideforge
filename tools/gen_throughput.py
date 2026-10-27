"""E0.9: generator throughput on GPU, torch's default mode vs deterministic mode (each in a fresh process).

    python tools/gen_throughput.py                      # B0, blocks of 8 lanes, L = 1024 / 4096 / 8192 (+ pred 128)

Prints points per second of tideforge.gen.generate_blocks (B0 = configs/ft3_main_cfg.json) per length and mode, and the
ratio deterministic / default. The owner decides from these numbers whether training uses deterministic mode.
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
gc = G.b0()
name = G.configure(gc)
dev = torch.device("cuda")
out = []
for L in (1024, 4096, 8192):
    Lb = L + 128
    N = max(8, int(16e6 // (Lb * 8)))              # about one generation pool (gen_pool_points 16M)
    for _ in range(2):
        G.generate_blocks(N, 8, Lb, dev, profile=name, group_prob=gc.group_prob)
    torch.cuda.synchronize()
    t0, reps = time.time(), 5
    for _ in range(reps):
        G.generate_blocks(N, 8, Lb, dev, profile=name, group_prob=gc.group_prob)
    torch.cuda.synchronize()
    dt = (time.time() - t0) / reps
    out.append(dict(L=L, blocks=N, points_per_s=N * 8 * Lb / dt, seconds_per_pool=dt))
print("RESULT " + json.dumps(out))
"""


def run(det: bool) -> list[dict]:
    env = {k: v for k, v in os.environ.items() if k != "CUBLAS_WORKSPACE_CONFIG"}
    if det:
        env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    code = CHILD.format(repo=str(REPO), tp=str(REPO / "third_party" / "tidemark_e21cbf0"), det=det)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
    if r.returncode:
        raise SystemExit(r.stderr[-3000:])
    return json.loads(r.stdout.split("RESULT ", 1)[1])


def main() -> int:
    a, b = run(False), run(True)
    print(f"{'L':>6s} {'default pts/s':>15s} {'determ. pts/s':>15s} {'determ./default':>16s}")
    for x, y in zip(a, b):
        print(f"{x['L']:6d} {x['points_per_s']:15.3e} {y['points_per_s']:15.3e} {y['points_per_s'] / x['points_per_s']:16.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
