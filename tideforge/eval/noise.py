"""The seed-noise file (E0.7, pre-registered 2026-10-29): runs/noise/e07.json.

    python -m tideforge.eval.noise --out runs/noise/e07.json [--dev-runs <dev run> ...] [--dev-cov-runs <cov run> ...]
                                   [--probe-csvs <results.csv> ...]

The noise floor for EXPERIMENTS.md rule 5 is the RANGE (max - min) over the seeds, per metric. The sd is recorded but not used.
- dev: per metric (MASE, CRPS), each seed's overall score (weighted geometric mean of the ratios to seasonal naive). The range
  is stored in the units devscore compare's gain uses (gain = 1 - A / B): max / min - 1.
- dev_cov (added 2026-10-29 for G1.13's win rule): the same on the dev-covariate set (groups ON score), from the seeds'
  devscore --cov runs.
- probes: per (probe, cell, mode, view, metric) of the probe suite's results.csv, the range of the value (in the metric's units).
Either part can be written alone and the other added later (the dev part from finish --ckpt once the dev set is built): an
existing file is updated, not replaced. Each part keeps the runs it came from.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

STAT = "range"
PROBE_KEY = ("probe", "cell", "mode", "view", "metric")


def _stats(values: list[float], dev: bool) -> dict:
    v = [float(x) for x in values]
    rng = max(v) / min(v) - 1 if dev else max(v) - min(v)
    sd = float(np.std(np.log(v), ddof=1)) if dev else float(np.std(v, ddof=1))
    return dict(range=rng, sd=sd, mean=float(np.mean(v)), values=v)


def dev_part(run_dirs: list[str]) -> dict:
    from tideforge.eval import devmetrics as M
    from tideforge.eval.devscore import load_run, weights_of
    runs = [load_run(d) for d in run_dirs]
    keys = {(r["version"], r["role"], r["set"], r["set_sha256"], r["cov"]) for r, _ in runs}
    if len(keys) != 1:
        raise SystemExit("the seed runs were scored on different dev versions / sets")
    scores = {m: [] for m in M.METRICS}
    for r, w in runs:
        mode = "on" if r["cov"] else "main"
        o = M.overall(M.dataset_scores(M.pair_ratios(w[w["mode"] == mode])), weights_of(r, w), ci=False)
        for m in M.METRICS:
            scores[m].append(o[m][0])
    return dict(runs=[str(d) for d in run_dirs], metrics={m: _stats(v, dev=True) for m, v in scores.items()})


def probe_part(csvs: list[str]) -> dict:
    out = {}
    frames = [pd.read_csv(c) for c in csvs]
    for f in frames:
        f["_key"] = f[list(PROBE_KEY)].astype(str).agg("/".join, axis=1)
    common = set.intersection(*(set(f["_key"]) for f in frames))
    for k in sorted(common):
        vals = [float(f.loc[f["_key"] == k, "value"].iloc[-1]) for f in frames]
        if all(math.isfinite(x) for x in vals):
            out[k] = _stats(vals, dev=False)
    return dict(runs=[str(c) for c in csvs], metrics=out)


def write(out: str | Path, dev_runs=None, probe_csvs=None, dev_cov_runs=None) -> dict:
    p = Path(out)
    d = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    d.update(stat=STAT, note="noise floor = range (max - min) over seeds; sd recorded, not used (E0.7 Part A)")
    for name, items, fn in (("dev", dev_runs, dev_part), ("dev_cov", dev_cov_runs, dev_part), ("probes", probe_csvs, probe_part)):
        if items:
            if len(items) < 2:
                raise SystemExit(f"{name}: at least 2 seeds are needed for a range")
            d[name] = fn(items)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return d


def floor(noise: dict | None, metric: str, part: str = "dev") -> float | None:
    """The noise floor for rule 5 (the range) of part dev | dev_cov, or None if that part is not written yet."""
    try:
        return float(noise[part]["metrics"][metric][STAT])
    except (TypeError, KeyError):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="runs/noise/e07.json")
    ap.add_argument("--dev-runs", nargs="+")
    ap.add_argument("--dev-cov-runs", nargs="+")
    ap.add_argument("--probe-csvs", nargs="+")
    a = ap.parse_args(argv)
    if not (a.dev_runs or a.probe_csvs or a.dev_cov_runs):
        raise SystemExit("give --dev-runs and / or --probe-csvs")
    d = write(a.out, a.dev_runs, a.probe_csvs, a.dev_cov_runs)
    for m, s in d.get("dev", {}).get("metrics", {}).items():
        print(f"dev {m}: range {s['range']:.4f} (sd {s['sd']:.4f}) over {len(s['values'])} seeds")
    print(f"probes: {len(d.get('probes', {}).get('metrics', {}))} metrics -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
