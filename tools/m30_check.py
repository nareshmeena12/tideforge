"""R1 GPU check (one interactive GPU, under 10 minutes), before the job is written.

    python tools/m30_check.py [--ckpt $FT3_MAIN_CKPT] [--steps 100] [--exp R1.3]

Builds m30, prints its parameter count, diffs the architecture against ft3_main's checkpoint config and loads its weights
strictly (any difference other than dropout fails, exit 1). Then runs the steps on B0 data and prints peak memory, points/s and
the projected hours for 100B points (training + generation); starts at 1024 rows, halves micro_rows on OOM or above 90% of GPU
memory, and prints the --micro-rows to give the job writer.
"""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO)]

import torch  # noqa: E402

from tideforge import model as TM  # noqa: E402
from tideforge.train import recipe as RC  # noqa: E402
from tideforge.train import trainer as TR  # noqa: E402

ARCH = ("patch_len", "d_model", "n_layers", "n_heads", "d_kv", "d_ff", "group_attn", "group_lanes", "arcsinh", "mask_channel",
        "skip_group_attn", "decoder", "max_pass_patches", "horizon_loss_share", "max_ctx", "min_ctx", "pred_len", "pack_rows",
        "max_segs_per_row", "long_row_frac", "long_min_ctx", "len_bucket_edges", "cov_r3", "cpm_prob", "cpm_min_patches",
        "cpm_max_patches", "cpm_suffix_frac", "cpm_long_p", "cpm_long_max", "dropout")
ALLOWED = {"dropout"}


def arch_diff(cfg: dict, ref: dict) -> list[str]:
    """Differences in the architecture fields, ALLOWED left out. A field in one cfg only is a difference."""
    return [f"{k}: ours {cfg.get(k, '<absent>')!r} vs ft3_main {ref.get(k, '<absent>')!r}" for k in ARCH
            if k not in ALLOWED and cfg.get(k, "<absent>") != ref.get(k, "<absent>")]


def try_rows(exp: str, micro: int | None, steps: int) -> dict:
    rc = RC.load(exp)
    rc.sections["run"]["micro_rows"] = micro
    with tempfile.TemporaryDirectory() as d:
        r = TR.Run(rc, 0, Path(d), "cuda", "m30check", log=print).build()
        r.n_bad = 0
        r.model.train()
        torch.cuda.reset_peak_memory_stats()
        skip = min(10, steps // 2)
        for _ in range(skip):
            r.one_step()
        torch.cuda.synchronize()
        t0, p0, g0 = time.time(), r.points, r.batcher.gen_seconds
        for _ in range(steps - skip):
            r.one_step()
        torch.cuda.synchronize()
        el = time.time() - t0
        v = r.validate()
        return dict(pts_s=(r.points - p0) / el, gen_frac=(r.batcher.gen_seconds - g0) / el, peak=torch.cuda.max_memory_allocated(),
                    total=torch.cuda.get_device_properties(0).total_memory, bad=r.n_bad, holdout=v["holdout"],
                    target=rc.sections["optim"]["target_points"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", default=os.environ.get("FT3_MAIN_CKPT"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--exp", default="R1.3")
    a = ap.parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if not a.ckpt or not Path(a.ckpt).exists():
        print(f"FAIL: no ft3_main checkpoint at {a.ckpt!r} (set FT3_MAIN_CKPT)")
        return 1
    cfg = RC.load(a.exp).ft3_cfg()
    model = TM.build(cfg)
    print(f"m30 parameters: {TM.n_params(model):,}")
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    diff = arch_diff(cfg, ck["cfg"])
    res = model.load_state_dict(ck["model"], strict=False)
    if res.missing_keys or res.unexpected_keys:
        diff.append(f"state dict: missing {res.missing_keys[:5]} unexpected {res.unexpected_keys[:5]}")
    print(f"dropout: ours {cfg['dropout']} vs ft3_main {ck['cfg'].get('dropout')} (allowed)")
    if diff:
        print("FAIL: architecture differs from ft3_main:", *diff, sep="\n  ")
        return 1
    print("architecture: identical to ft3_main except dropout; weights load strictly")
    del model, ck
    micro = None
    while True:
        rows = micro or cfg["pack_rows"]
        try:
            m = try_rows(a.exp, micro, a.steps)
        except torch.cuda.OutOfMemoryError:
            print(f"rows {rows}: out of memory")
            m = None
        torch.cuda.empty_cache()
        if m and m["peak"] <= 0.9 * m["total"]:
            break
        if m:
            print(f"rows {rows}: peak {m['peak'] / 2**30:.1f} GB is above 90% of {m['total'] / 2**30:.1f} GB")
        if rows <= cfg["group_lanes"] * 2:
            print("FAIL: no micro-batch fits")
            return 1
        micro = rows // 2
    h = m["target"] / m["pts_s"] / 3600
    print(f"rows per forward {rows}: peak {m['peak'] / 2**30:.1f} GB of {m['total'] / 2**30:.1f} GB, {m['pts_s'] / 1e6:.2f}M pts/s, "
          f"generation {m['gen_frac']:.1%}, bad steps {m['bad']}, holdout {m['holdout']:.4f}")
    print(f"projected: {h:.1f} h for {m['target'] / 1e9:.0f}B points, training + generation, one run (ESTIMATE; validation, "
          f"checkpoints, finish and fev not included)")
    print(f"--micro-rows for the job writer: {micro or 0}  (0 = no accumulation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
