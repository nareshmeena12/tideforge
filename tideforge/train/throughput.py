"""E0.9: end-to-end 5M training throughput from the trainer's metrics.jsonl, generator deterministic vs default mode.

    python -m tideforge.train.throughput report --det <run dir> --default <run dir> [--skip 2]

Per run: median points/s over the logged windows (the first `skip` windows dropped: torch.compile and pool warm-up) and the
median share of wall time spent generating (gen_frac). Generation runs on the main thread between steps (never overlapping the
model), so gen_frac is also what overlapping could at most win back: the cost of not overlapping. Prints the markdown rows for
experiments/E0.9.md and the 100B-point wall time per run; tideforge/slurm.py's E0.7 estimate uses the deterministic rate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def summary(run_dir: str | Path, skip: int = 2) -> dict:
    rows = [json.loads(s) for s in (Path(run_dir) / "metrics.jsonl").read_text(encoding="utf-8").splitlines() if s.strip()]
    tr = [r for r in rows if r.get("kind") == "train"][skip:]
    if not tr:
        raise SystemExit(f"{run_dir}: no training windows after skipping {skip}")
    pps = median(r["pts_per_s"] for r in tr)
    return dict(pts_per_s=pps, gen_frac=median(r["gen_frac"] for r in tr), windows=len(tr), hours_100B=100e9 / pps / 3600)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("report",))
    ap.add_argument("--det", required=True)
    ap.add_argument("--default", required=True)
    ap.add_argument("--skip", type=int, default=2)
    a = ap.parse_args(argv)
    d, n = summary(a.det, a.skip), summary(a.default, a.skip)
    print("| generator mode | points/s (median) | generation share of wall time | 100B points, h per run |")
    print("|---|---|---|---|")
    for name, s in (("deterministic", d), ("default", n)):
        print(f"| {name} | {s['pts_per_s']:.3e} | {100 * s['gen_frac']:.1f}% | {s['hours_100B']:.2f} |")
    r = d["pts_per_s"] / n["pts_per_s"]
    print(f"\ndeterministic / default = {r:.3f} ({'above' if r < 0.9 else 'within'} the 10% the owner set for revisiting)")
    print(f"cost of not overlapping generation (deterministic mode): at most {100 * d['gen_frac']:.1f}% of wall time")
    print(f"for tideforge/slurm.py: TRAIN_PTS_PER_S = {d['pts_per_s']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
