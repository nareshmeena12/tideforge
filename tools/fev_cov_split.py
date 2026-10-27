"""D1: split fev-bench skill into tasks with and without covariates, and measure how the gap to a reference model is spread.

Uses the per-task result files the fev repo publishes (benchmarks/fev_bench/results/*.csv) plus, optionally, your own
results file in the same format (one row per task, columns as produced by fev's summaries).

Usage (from the repo root):
    git clone --depth 1 https://github.com/autogluon/fev third_party_ext/fev        # not committed
    python tools/fev_cov_split.py --fev third_party_ext/fev --ours path/to/ft3_main_fev.csv --name ft3_main
    python tools/fev_cov_split.py --fev $TM_EVAL/fev --ours $TF_REF/scores_long.csv --arm ft3mainon --name ft3_main_on         --out $TF_ROOT/d1/fev_rel_errors.csv                                    # a tidemark board (Perlmutter)
    python tools/fev_cov_split.py --fev third_party_ext/fev                         # published models only

Writes docs/report/data/fev_rel_errors.csv (per-task relative error vs Seasonal Naive, all models) and prints, per model:
skill on all / no-covariate / covariate tasks, the share of the gap to --leader coming from covariate tasks, and how
concentrated that gap is (share held by the top 10 / 25 tasks). Needs: pip install fev pandas.
"""
import argparse
import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--fev", required=True, help="path to a clone of github.com/autogluon/fev")
ap.add_argument("--ours", default=None, help="your fev results csv (same format as the published files), or a tidemark "
                "board scores_long.csv (columns model, arm, seed, benchmark, metric, task, window, err) with --arm")
ap.add_argument("--arm", default=None, help="with a scores_long.csv: the arm to take (ft3mainon = groups ON, ft3main = hidden)")
ap.add_argument("--name", default="ours", help="model_name to give your results")
ap.add_argument("--leader", default="TimeRSI-1.5")
ap.add_argument("--out", default="docs/report/data/fev_rel_errors.csv")
a = ap.parse_args()

import fev  # noqa: E402  (the fev package's own leaderboard / pivot functions)

files = glob.glob(str(Path(a.fev) / "benchmarks" / "fev_bench" / "results" / "*.csv"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
if a.ours:
    o = pd.read_csv(a.ours)
    if "arm" in o.columns:
        # scores_long.csv: one test_error per fev task; the task identity columns are copied from a published model's
        # row of the same task_name (fev's pivot needs them)
        if not a.arm:
            raise SystemExit("--arm is required with a scores_long.csv (e.g. ft3mainon)")
        o = o[(o["arm"] == a.arm) & (o["benchmark"] == "fev") & (o["metric"] == "test_error")]
        o = o.groupby("task", as_index=False)["err"].mean()                     # one seed per arm here; mean if several
        tmpl = df[df.model_name == df.model_name.iloc[0]].drop_duplicates("task_name").set_index("task_name")
        missing = sorted(set(o["task"]) - set(tmpl.index))
        if missing:
            print(f"{len(missing)} of our tasks have no published row (left out): {missing[:10]}")
        o = o[o["task"].isin(tmpl.index)]
        rows = tmpl.loc[o["task"]].reset_index()
        rows["test_error"] = o["err"].to_numpy()
        o = rows
        print(f"{a.ours}: arm {a.arm}, {len(o)} fev tasks")
    o["model_name"] = a.name
    df = pd.concat([df, o], ignore_index=True)

# keep only models with all 100 tasks (fev.leaderboard refuses incomplete ones)
counts = df.groupby("model_name")["dataset_config"].nunique()
full = counts[counts >= counts.max()].index
dropped = sorted(set(counts.index) - set(full))
if dropped:
    print("dropped (missing tasks):", dropped)
df = df[df.model_name.isin(full)]

piv = fev.pivot_table(df, metric_column="test_error", baseline_model="Seasonal Naive")
piv.index = [i if isinstance(i, str) else "|".join(map(str, i)) for i in piv.index]
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
piv.to_csv(a.out)
L = np.log(piv.clip(1e-2, 100))


def has_cov(ix: str) -> bool:
    f = ix.split("|")
    lists = [x for x in f if x.startswith("[")]
    known = lists[-3] if len(lists) >= 3 else "[]"
    past = lists[-2] if len(lists) >= 2 else "[]"
    return known != "[]" or past != "[]"


cov = np.array([has_cov(t) for t in L.index])
skill = lambda col, m=None: 100 * (1 - math.exp(np.nanmean(L[col] if m is None else L[col][m])))
print(f"tasks: {len(L)}, with covariates: {cov.sum()}")
print(f"{'model':22s} {'all':>6s} {'no-cov':>7s} {'cov':>6s} {'gap from cov':>13s} {'top10':>6s} {'top25':>6s}")
for m in sorted(L.columns, key=lambda c: -skill(c)):
    g = L[m] - L[a.leader]
    pos = g.clip(lower=0)
    share = 100 * g[cov].sum() / g.sum() if g.sum() > 0 else float("nan")
    t10 = 100 * pos.nlargest(10).sum() / pos.sum() if pos.sum() > 0 else float("nan")
    t25 = 100 * pos.nlargest(25).sum() / pos.sum() if pos.sum() > 0 else float("nan")
    print(f"{m[:22]:22s} {skill(m):6.1f} {skill(m, ~cov):7.1f} {skill(m, cov):6.1f} {share:12.0f}% {t10:5.0f}% {t25:5.0f}%")
if a.ours:
    g = (L[a.name] - L[a.leader]).sort_values(ascending=False)
    print(f"\nworst 15 tasks for {a.name} vs {a.leader} (error ratio):")
    for ix, v in g.head(15).items():
        print(f"  {ix.split('|')[1]:32s} x{math.exp(v):.2f}  {'covariates' if has_cov(ix) else ''}")
