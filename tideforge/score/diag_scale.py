"""Scale diagnostic for tasks where a model is orders of magnitude worse than seasonal naive (GPU job; one per checkpoint).

For each task and series (window row x target), uses the same inputs as the scorer (window cache, prepare = last 8192 points
with gaps filled one season back) and prints:
  context n / min / max / mean / std (float64), the model's input scale (mean and std of the float32 context as ft3_core's
  _causal_stats sees it at the context end, the std after its absolute floor 1e-3, and whether the floor binds), the float32
  rounding step at the context level, and the forecast median min / max for four variants:
    raw_bf16   the old path (autocast bf16 on CUDA)          raw_fp32   the same with autocast off
    res_bf16   rescale on (adapter default)                  res_fp32   rescale on, autocast off
plus the task's MASE relative to seasonal naive for each variant. raw_bf16 vs raw_fp32 separates precision; raw vs res
separates normalisation. Univariate path (groups OFF), as in the report.

    python -m tideforge.score.diag_scale --ckpt CKPT [--model-id ID] [--tier fast] [--tasks a,b,...] [--max-print 20]
-> <out>/<model_id>/diag_scale.csv (every series) and diag_scale.md (task table + the first --max-print series per task).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASKS = ["thermobench_thermobench_pure__5min/medium", "thermobench_thermobench_pure__h/medium",
         "datonic_world_development_indicators_default__ys/short", "juliensimon_goes_xray_flux__min/medium"]
VARIANTS = [("raw_bf16", False, False), ("raw_fp32", False, True), ("res_bf16", True, False), ("res_fp32", True, True)]


@contextlib.contextmanager
def fp32(on: bool):
    """Turn autocast off inside ft3_core (it calls torch.autocast(..., enabled=amp_on) at call time)."""
    import torch
    if not on:
        yield
        return
    orig = torch.autocast

    def off(*a, **k):
        k["enabled"] = False
        return orig(*a, **k)
    torch.autocast = off
    try:
        yield
    finally:
        torch.autocast = orig


def model_view(x) -> dict:
    """What _causal_stats sees at the context end: float32 input, float64 stats, absolute std floor 1e-3."""
    f = np.asarray(x, dtype=np.float32)
    f = f[np.isfinite(f)][-8192:].astype(np.float64)
    if not len(f):
        return dict(m_mean=np.nan, m_std=np.nan, m_sd_used=np.nan, floor_binds=False, f32_step=np.nan)
    sd = float(f.std())
    lvl = float(np.abs(f).max())
    return dict(m_mean=float(f.mean()), m_std=sd, m_sd_used=max(sd, 1e-3), floor_binds=sd < 1e-3,
                f32_step=float(np.spacing(np.float32(lvl))))


def run(a) -> int:
    from tideforge.score import metrics as MT
    from tideforge.score import predict as PR
    from tideforge.score import release as RL
    from tideforge.score import windows as W
    from tideforge.score.adapters.ft3 import Ft3
    from tideforge.score.adapters.naive import SeasonalNaive
    rel = RL.Release(Path(a.release_root or RL.default_release_root()) / a.version)
    want = [t for t in a.tasks.split(",") if t]
    specs = [s for s in rel.tasks(a.tier, "tuning") if s["row"]["task"] in want]
    missing = sorted(set(want) - {s["row"]["task"] for s in specs})
    if missing:
        print(f"not in {a.version} {a.tier}: {missing}", flush=True)
    if not specs:
        return 1
    caches = W.build(specs, rel.path / "_cache" / f"{a.tier}_tuning", ctx=8192, workers=1, log=lambda s: None)
    nv = SeasonalNaive("tfbench")
    qn, _ = PR.run(caches, nv, "off", Path(a.out) / "_naive", "naive_tfbench", log=lambda s: None)
    naive = {c.name: MT.score_task(c, qn[c.name]) for c in caches}
    pred = Ft3(a.ckpt, device=a.device, name=a.model_id)
    rows, task_rows = [], []
    for c in caches:
        R, D, H, S = c.info["n_rows"], len(c.info["target_columns"]), c.info["horizon"], c.info["seasonality"]
        units = [(r, d) for r in range(R) for d in range(D)]
        x = [pred.prepare(c.past(r, d), S) for r, d in units]
        med = MT.point_index(c.info["quantile_levels"])
        base = [dict(task=c.name, row=r, item=int(c.r_item[r]), target=d, n=int(np.isfinite(v).sum()),
                     min=np.nanmin(v), max=np.nanmax(v), mean=np.nanmean(v), std=np.nanstd(v), **model_view(v))
                for (r, d), v in zip(units, x)]
        tr = dict(task=c.name, series=len(units), naive_MASE=naive[c.name]["MASE"])
        for name, res, f32 in VARIANTS:
            pred.rescale = res
            with fp32(f32):
                q = pred.predict(x, H, season=S)                              # (N, Q, H)
            sc = MT.score_task(c, q.reshape(R, D, *q.shape[1:]))
            tr[f"rel_MASE_{name}"] = sc["MASE"] / naive[c.name]["MASE"]
            for b, qi in zip(base, q):
                b[f"fc_min_{name}"], b[f"fc_max_{name}"] = float(qi[med].min()), float(qi[med].max())
        task_rows.append(tr)
        rows += base
        print(json.dumps(tr), flush=True)
    out = Path(a.out) / a.model_id
    out.mkdir(parents=True, exist_ok=True)
    df, tf = pd.DataFrame(rows), pd.DataFrame(task_rows)
    df.to_csv(out / "diag_scale.csv", index=False)
    from tideforge.tfbench.report import _md
    cols = ["row", "target", "n", "min", "max", "mean", "std", "m_std", "floor_binds", "f32_step"] + \
        [f"fc_{k}_{v}" for v, _, _ in VARIANTS for k in ("min", "max")]
    L = [f"# Scale diagnostic: {a.model_id} ({a.ckpt}), tfbench {a.version} {a.tier}", "",
         "rel_MASE = model MASE / seasonal-naive MASE. raw = old path, res = rescale on; bf16 = autocast (CUDA), fp32 = off.",
         "", _md(tf.round(4)), ""]
    for t, g in df.groupby("task", sort=False):
        L += [f"## {t} ({len(g)} series; first {a.max_print})", "",
              _md(g[cols].head(a.max_print).map(lambda v: f"{v:.4g}" if isinstance(v, float) else v)), ""]
    (out / "diag_scale.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L), flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model-id", default="ft3_main")
    ap.add_argument("--version", default="v1.1.0")
    ap.add_argument("--release-root")
    ap.add_argument("--tier", default="fast")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--max-print", type=int, default=20)
    ap.add_argument("--device")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.out is None:
        import os
        a.out = str(Path(os.environ.get("TF_ROOT", ".")) / "results" / "diag_scale")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
