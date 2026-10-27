"""Build, run and report the probe suite.

    python -m tideforge.probes build  --root $TF_DATA/probes [--probes all] [--n 256]
    python -m tideforge.probes run    --root $TF_DATA/probes --out $TF_ROOT/probes --model ft3main --ckpt $FT3_MAIN_CKPT
                                      [--probes all] [--nan-variant baseline] [--device cuda]
    python -m tideforge.probes report --out $TF_ROOT/probes [--model ft3main[:variant]] [--compare A B]

Each probe module (pNN_*.py) defines PROBE, VERSION, SEED, N_PER_CELL, THRESHOLDS, build(n_per_cell, seed) -> ProbeSet,
metrics(cell, preds) -> [Metric] with preds {(mode, view): (n, k, 9, H)}, and optionally cross(cells, per_cell) ->
[(cell, Metric A, Metric B)] for ratios between cells (a cell vs its control: unpaired bootstrap).

results.csv: one row per (model, nan variant, probe, version, cell, mode, view, metric): value and 95% bootstrap CI over
series (1000 resamples, fixed seed). series/<model>__<variant>/<probe>_v<version>.npz keeps the per-series components and
the raw quantile forecasts, so `report --compare A B` can bootstrap the per-series difference on the same items.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib
import io
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.probes import common, metrics as M
from tideforge.probes.models import make

MODULES = {"P1": "p01_covariates", "P2": "p02_calendar", "P3": "p03_cycles", "P4": "p04_breaks", "P5": "p05_quasiperiodic",
           "P6": "p06_gaps", "P7": "p07_counts", "P8": "p08_hierarchy", "P9": "p09_dynamics", "P10": "p10_tails"}
COLUMNS = ["date", "git", "model", "ckpt_sha", "nan_variant", "probe", "version", "cell", "control", "mode", "view", "metric",
           "value", "ci_lo", "ci_hi", "n"]


def module(probe: str):
    return importlib.import_module(f"tideforge.probes.{MODULES[probe]}")


def probe_ids(spec: str) -> list[str]:
    ids = list(MODULES) if spec == "all" else [s.strip().upper() for s in spec.split(",")]
    have = [p for p in ids if p in MODULES and importlib.util.find_spec(f"tideforge.probes.{MODULES[p]}")]
    return have


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).resolve().parent).stdout.strip()
    except OSError:
        return ""


def evaluate(model, ps: common.ProbeSet, mod) -> tuple[list[tuple], dict]:
    """[(cell, Metric)] for every cell (and cross-cell ratios), plus the raw forecasts {(cell, mode, view): q}."""
    out, fc, per_cell = [], {}, {}
    for cell in ps.cells:
        preds = {(m, v): model.predict_cell(cell, m, v) for m in cell.modes for v in cell.views}
        fc.update({(cell.name, m, v): q for (m, v), q in preds.items()})
        ms = mod.metrics(cell, preds)
        per_cell[cell.name] = ms
        out += [(cell, m) for m in ms]
    if hasattr(mod, "cross"):
        for cell, a, b in mod.cross(ps.cells, per_cell):
            out.append((cell, ("cross", a, b)))
    return out, fc


def run(model, probes: list[str], root: Path, out: Path, nan_variant: str = "baseline", say=print) -> pd.DataFrame:
    out = Path(out)
    rows = []
    key = f"{model.name}__{nan_variant}"
    (out / "series" / key).mkdir(parents=True, exist_ok=True)
    date, git = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), git_head()
    for p in probes:
        mod = module(p)
        ps = common.load(root, p, mod.VERSION)
        res, fc = evaluate(model, ps, mod)
        store = {}
        for cell, m in res:
            if isinstance(m, tuple):
                _, a, b = m
                v, lo, hi = M.unpaired(a, b)
                name, mode, view, n = f"{a.name}_ratio_vs_control", a.mode, a.view, len(a.comps)
            else:
                v, lo, hi = M.bootstrap(m)
                name, mode, view, n = m.name, m.mode, m.view, len(m.comps)
                store[f"{cell.name}|{m.name}|{m.mode}|{m.view}|{m.agg_name}"] = m.comps
            rows.append(dict(date=date, git=git, model=model.name, ckpt_sha=getattr(model, "ckpt_sha", ""), nan_variant=nan_variant,
                             probe=p, version=mod.VERSION, cell=cell.name, control=cell.control, mode=mode, view=view,
                             metric=name, value=v, ci_lo=lo, ci_hi=hi, n=n))
        for (c, mo, vi), q in fc.items():
            store[f"{c}|q|{mo}|{vi}|-"] = q.astype(np.float32)
        buf = io.BytesIO()
        np.savez_compressed(buf, **store)
        (out / "series" / key / f"{p}_v{mod.VERSION}.npz").write_bytes(buf.getvalue())
        say(f"{p} v{mod.VERSION}: {len(ps.cells)} cells done")
    df = pd.DataFrame(rows, columns=COLUMNS)
    f = out / "results.csv"
    df.to_csv(f, mode="a", header=not f.exists(), index=False)
    return df


# ---------------------------------------------------------------------------------------------------- report
def _key(s: str) -> tuple[str, str]:
    name, _, var = s.partition(":")
    return name, var or "baseline"


def table(df: pd.DataFrame) -> str:
    """One block per probe: cells x (metric @ mode/view), 'value [lo, hi]'."""
    lines = []
    for p, g in df.groupby("probe", sort=False):
        g = g.assign(col=g["metric"] + "@" + g["mode"] + ("/" + g["view"]).where(g["view"] != "main", ""),
                     txt=[f"{v:.3f} [{a:.3f}, {b:.3f}]" for v, a, b in zip(g["value"], g["ci_lo"], g["ci_hi"])])
        t = g.pivot_table(index="cell", columns="col", values="txt", aggfunc="last", sort=False)
        lines += [f"== {p} (v{g['version'].iloc[0]}) {g['model'].iloc[0]} / {g['nan_variant'].iloc[0]}", t.to_string(), ""]
        if p == "P7":
            lines.append("note: P(y=0) is read from 9 quantiles, so its resolution is about 0.1 (raw quantiles are in series/).")
    return "\n".join(lines)


def compare(out: Path, a: str, b: str) -> pd.DataFrame:
    """Paired difference agg(A) - agg(B) per (probe, cell, metric, mode, view) present in both runs."""
    (na, va), (nb, vb) = _key(a), _key(b)
    rows = []
    for fa in sorted((Path(out) / "series" / f"{na}__{va}").glob("*.npz")):
        fb = Path(out) / "series" / f"{nb}__{vb}" / fa.name
        if not fb.exists():
            continue
        za, zb = np.load(fa), np.load(fb)
        for k in za.files:
            cell, metric, mode, view, agg = k.split("|")
            if metric == "q" or k not in zb.files:
                continue
            ma, mb = M.Metric(metric, za[k], agg, mode, view), M.Metric(metric, zb[k], agg, mode, view)
            d, lo, hi = M.paired(ma, mb)
            rows.append(dict(probe=fa.stem, cell=cell, metric=metric, mode=mode, view=view, a=ma.agg(ma.comps),
                             b=mb.agg(mb.comps), diff=d, ci_lo=lo, ci_hi=hi, n=len(ma.comps)))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m tideforge.probes")
    ap.add_argument("cmd", choices=["build", "run", "report"])
    ap.add_argument("--root")
    ap.add_argument("--out")
    ap.add_argument("--probes", default="all")
    ap.add_argument("--n", type=int)
    ap.add_argument("--model")
    ap.add_argument("--ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--nan-variant", default="baseline", choices=["baseline", "nofill_delete", "nofill_mask"])
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    a = ap.parse_args(argv)
    probes = probe_ids(a.probes)
    if a.cmd == "build":
        for p in probes:
            mod = module(p)
            ps = mod.build(a.n or mod.N_PER_CELL, mod.SEED)
            meta = common.save(ps, Path(a.root))
            print(f'{p} v{mod.VERSION}: {{"seed": {mod.SEED}, "n_per_cell": {ps.n_per_cell}, "cells": {len(ps.cells)}, '
                  f'"sha256": "{meta["sha256"]}", "fingerprint": "{meta["fingerprint"]}"}}')
        return 0
    if a.cmd == "run":
        model = make(a.model, a.ckpt, a.device, a.nan_variant)
        df = run(model, probes, Path(a.root), Path(a.out), a.nan_variant)
        print(table(df))
        return 0
    if a.compare:
        d = compare(Path(a.out), *a.compare)
        print(d.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        return 0
    df = pd.read_csv(Path(a.out) / "results.csv")
    if a.model:
        n, v = _key(a.model)
        df = df[(df["model"] == n) & (df["nan_variant"] == v)]
    df = df.groupby(["model", "nan_variant", "probe", "cell", "mode", "view", "metric"], sort=False).tail(1)
    print(table(df))
    return 0
