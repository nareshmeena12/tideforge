"""CPU training-step throughput, known_attn off vs mask (M1, G1.13), on the tiny test model and the 5M recipe model.

    python tools/m1_speed.py [--reps 10] [--threads 8]

One step = forward (packed, loss) + backward + AdamW on a synthetic packed batch with the training geometry: blocks of 8 lanes
sharing one layout, groups packed one after another along the row (lengths drawn per group), roles per lane (1-2 targets, the
rest known / past), a CPM span blanked on the targets. off and mask run interleaved on the same batch and the same weights;
the median step time of each is reported. CPU only: a relative cost, not the GPU number (E0.9 measures that on Perlmutter).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "third_party" / "tidemark_e21cbf0")]

from tideforge import model as TM                                   # noqa: E402
from tideforge.model.ft3_model import ROLE_KNOWN, ROLE_PAST, ROLE_TARGET   # noqa: E402
from tideforge.train import recipe as RC                            # noqa: E402


def packed_batch(blocks: int, lanes: int, P: int, pl: int, seed: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    R, W = blocks * lanes, P * pl
    x = torch.randn(R, W, generator=g).cumsum(1) * 0.1
    seg = torch.empty(R, P, dtype=torch.long)
    gid = torch.empty(R, P, dtype=torch.long)
    role = torch.empty(R, P, dtype=torch.long)
    blank = torch.zeros(R, W, dtype=torch.bool)
    s = gi = 0
    for b in range(blocks):
        p = 0
        while p < P:
            n = min(P - p, int(torch.randint(8, 65, (1,), generator=g)))
            nt = int(torch.randint(1, 3, (1,), generator=g))
            for v in range(lanes):
                r = b * lanes + v
                seg[r, p:p + n], gid[r, p:p + n] = s + v, gi
                role[r, p:p + n] = ROLE_TARGET if v < nt else (ROLE_KNOWN if torch.rand(1, generator=g) < 0.5 else ROLE_PAST)
                if v < nt and n > 4:
                    blank[r, (p + n - min(16, n // 2)) * pl:(p + n) * pl] = True            # a suffix span
            s, gi, p = s + lanes, gi + 1, p + n
    return dict(x=x, mask=torch.ones(R, W, dtype=torch.bool), seg_patch=seg, slot_valid=torch.ones(s, dtype=torch.bool),
                blank=blank, lanes=lanes, gid=gid, role=role)


def step_time(model, opt, b) -> float:
    t = time.perf_counter()
    out = model(b["x"], mask=b["mask"], seg_patch=b["seg_patch"], slot_valid=b["slot_valid"], blank=b["blank"],
                lanes=b["lanes"], gid=b["gid"], role=b["role"])
    out["loss"].backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return time.perf_counter() - t


def measure(cfg: dict, b: dict, reps: int) -> dict:
    torch.manual_seed(0)
    models = {k: TM.build(cfg, {"known_attn": k}).train() for k in ("off", "mask")}
    models["mask"].load_state_dict(models["off"].state_dict())
    opts = {k: torch.optim.AdamW(m.parameters(), lr=1e-4) for k, m in models.items()}
    times = {k: [] for k in models}
    for i in range(reps + 2):                                        # 2 warm-up steps each, interleaved
        for k in ("off", "mask") if i % 2 == 0 else ("mask", "off"):
            dt = step_time(models[k], opts[k], b)
            if i >= 2:
                times[k].append(dt)
    med = {k: statistics.median(v) for k, v in times.items()}
    return dict(params={k: TM.n_params(m) for k, m in models.items()}, median_s=med,
                steps_per_s={k: 1 / v for k, v in med.items()}, delta=med["mask"] / med["off"] - 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--blocks-5m", type=int, default=2, help="row blocks of 8 lanes for the 5M batch (recipe: 1024 rows)")
    a = ap.parse_args(argv)
    torch.set_num_threads(a.threads)
    rc = RC.load("E0.7")
    cfg5 = dict(rc.ft3_cfg(), dropout=0.0)
    tiny = dict(cfg5, d_model=32, n_layers=1, n_heads=2, d_kv=16, d_ff=64)       # tests/train/test_trainer.py's tiny model
    pl, lanes = cfg5["patch_len"], cfg5["group_lanes"]
    runs = (("tiny (test_trainer)", tiny, packed_batch(2, lanes, 256 // pl, pl), a.reps * 5),
            (f"5M recipe ({a.blocks_5m * lanes} rows x {cfg5['max_ctx']})", cfg5,
             packed_batch(a.blocks_5m, lanes, cfg5["max_ctx"] // pl, pl), a.reps))
    print(f"torch {torch.__version__}, {torch.get_num_threads()} threads, CPU; median of the timed steps (fwd + bwd + AdamW)")
    for name, cfg, b, reps in runs:
        r = measure(cfg, b, reps)
        print(f"{name:34s} params off {r['params']['off']:,} mask {r['params']['mask']:,} | steps/s off "
              f"{r['steps_per_s']['off']:.3f} mask {r['steps_per_s']['mask']:.3f} | step time {100 * r['delta']:+.1f}% ({reps} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
