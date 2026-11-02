"""GPU check before the adapter jobs (one interactive GPU, a few minutes). Nothing is written.

    python tools/ad_check.py [--exp AD.2] [--steps 30]

Prints: backbone found, parameter counts, adapted equals backbone at init on the real checkpoint (bf16), points/s with covariate-only
batches, the generator's share of the time, and a projection for the budget and the 1B pilot. Fails on any mismatch.
"""
from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path
import os

import torch

from tideforge import model as TM
from tideforge.train import adapter_train as AT
from tideforge.train import batcher as TB
from tideforge.train import recipe as RC
from tideforge.train.adjob import BACKBONE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="AD.2")
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    root = Path(os.environ["TF_ROOT"])
    cks = sorted((root / "runs").glob(f"{BACKBONE[a.exp]}-*/final.pt"), key=lambda p: p.stat().st_mtime)
    if not cks:
        raise SystemExit(f"no final.pt for {BACKBONE[a.exp]}")
    rc = RC.load(a.exp)
    rc.sections["optim"]["peak_lr"] = 3e-4
    run = AT.AdapterRun(rc, 0, Path(tempfile.mkdtemp()), "cuda", "check", cks[-1])
    run.build()
    print(f"backbone {cks[-1]}\nadapter {run.n_adapter:,} parameters, model {run.n_params:,}")
    base, _, _ = TM.load(str(cks[-1]), run.device)
    b = run.batcher.next_batch()
    with torch.no_grad(), run._amp():
        qa = TB.packed_forward(run.model.eval(), b)["_q"]
        qb = TB.packed_forward(base, b)["_q"]
    if not torch.equal(qa, qb):
        raise SystemExit(f"adapted differs from the backbone at init, max {(qa - qb).abs().max():.3g}")
    print("init equals the backbone on a real batch: yes")
    del base
    run.model.train()
    run.n_bad = 0
    torch.cuda.reset_peak_memory_stats()
    g0, p0, t0 = run.batcher.gen_seconds, run.points, time.time()
    for _ in range(a.steps):
        run.one_step()
    torch.cuda.synchronize()
    el = time.time() - t0
    rate = (run.points - p0) / el
    print(f"{rate / 1e6:.2f}M pts/s, generator {(run.batcher.gen_seconds - g0) / el:.1%}, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB")
    print(f"projection (alone on the GPU): 1B pilot {1e9 / rate / 3600:.2f} h, 5B run {5e9 / rate / 3600:.2f} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
