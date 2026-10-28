"""G9.2 / G10.1: offline pools for the two families with a time loop (tar, garch_jump). Off by default: the families run their
loop online unless their switch param `pool` names a pool. Build on Perlmutter (GPU) only if gen_throughput --loops shows
the loop above 10% of B0's generator time at the family's share (owner 2026-10-28).

    python tools/build_loop_pool.py --family tar        --out $TF_DATA/pools [--name tar_v1]        [--n 65536] [--T 16384]
    python tools/build_loop_pool.py --family garch_jump --out $TF_DATA/pools [--name garch_jump_v1] [--n 65536] [--T 16384]
        [--device cuda] [--seed 0] [--chunk 8192]

Writes <out>/<name>/traj.npy (float16, n x T x 1) and meta.npy, and the manifest tideforge/gen/ext/pools/<name>.json. tar
rows are standardised per row (generation rescales them); garch_jump rows are returns in sd units (generation turns them into
prices or returns). Chunk k is drawn from its own torch generator seeded (seed, k), so the pool depends on the device and
torch version (recorded), not on anything else. Size: n T 2 bytes (65536 x 16384: 2.1 GB).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "third_party" / "tidemark_e21cbf0")]

import torch  # noqa: E402

from tideforge.gen.ext import _pool  # noqa: E402
from tideforge.gen.ext import g9_tar, g10_finance  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["tar", "garch_jump"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--name")
    ap.add_argument("--n", type=int, default=65536)
    ap.add_argument("--T", type=int, default=16384)
    ap.add_argument("--chunk", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--manifest-dir", default=str(REPO / "tideforge" / "gen" / "ext" / "pools"))
    a = ap.parse_args(argv)
    name = a.name or f"{a.family}_v1"
    dev = torch.device(a.device)
    traj = np.zeros((a.n, a.T, 1), np.float16)
    meta = np.zeros((a.n, 4), np.float32)
    for k, i0 in enumerate(range(0, a.n, a.chunk)):
        m = min(a.chunk, a.n - i0)
        g = torch.Generator(device=dev).manual_seed(a.seed * 1_000_003 + k)
        if a.family == "tar":
            x, info = g9_tar.tar_raw(g, m, a.T, dev)
            x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)
            meta[i0:i0 + m, 0] = info["R"].cpu().numpy()
            meta[i0:i0 + m, 1] = info["form"].cpu().numpy()
        else:
            x, info = g10_finance.garch_raw(g, m, a.T, dev)
            meta[i0:i0 + m, 0] = info["nu"].view(-1).cpu().numpy()
            meta[i0:i0 + m, 1] = info["lam"].view(-1).cpu().numpy()
        traj[i0:i0 + m, :, 0] = x.clamp(-6e4, 6e4).cpu().numpy().astype(np.float16)
        print(f"chunk {k}: rows {i0}..{i0 + m - 1}", flush=True)
    info = dict(builder=f"python tools/build_loop_pool.py --family {a.family} --out $TF_DATA/pools --name {name} --n {a.n} "
                        f"--T {a.T} --chunk {a.chunk} --seed {a.seed} --device {dev.type}",
                seed=a.seed, family=a.family, device=str(dev),
                versions=dict(torch=torch.__version__, numpy=np.__version__, python=sys.version.split()[0]),
                meta_columns="tar: regimes, form; garch_jump: nu, lambda")
    print(json.dumps(_pool.write(Path(a.out), name, traj, meta, info, Path(a.manifest_dir)), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
