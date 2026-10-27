"""Freeze a dev version: tasks, windows, weights, score set, fast subset, manifest (DATA_COLLECTION.md step 9).

    python -m tideforge.devdata.build --staging $TF_DATA/staging --out $TF_DATA/dev --version dev-v1
    python -m tideforge.devdata.build ... --version batch1-test --pipeline-test      # fewer than 15 domains allowed

- Task = (dataset, frequency, term); horizons from freqs.horizons (TIME's convention). Per series and term: the last
  N_WINDOWS rolling windows (window j: target = the H points ending j*H before the end, context = everything before; the
  context needs >= max(64, H) points and the target >= 50% observed values).
- Weights, per role (tuning / locked) separately: every dataset equal, no domain above DOMAIN_CAP of the weight (water-filling:
  capped domains get the cap, the rest is shared by the others in proportion to their dataset counts). Within a dataset its
  frequencies and terms share its weight equally, and windows share their task's weight.
- Score set and fast subset: per dataset a quota proportional to its weight (at least 1 window), windows picked by
  sha256 of their key (frozen); each picked window carries weight / picked so every dataset keeps its weight.
- The version folder holds copies of the staged files it uses (so later ingests never change it), the window tables and
  manifest.json; catalog/manifests/<version>.json is the committed copy (counts and sha256 only, no data).
Covariate tasks are windowed the same way into covtasks_windows.parquet and scored separately (not in the dataset weights).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.devdata import split as S
from tideforge.devdata.catalog import CAT, REPO
from tideforge.devdata.freqs import HORIZONS_EXTRA, TERMS, TIME_HORIZONS, horizons
from tideforge.devdata.ingest import sha256_file
from tideforge.devdata.schema import SOURCES

N_WINDOWS = 3
DOMAIN_CAP = 0.10
MIN_DOMAINS = 15
SIZES = {"tuning": {"score": 75000, "fast": 15000}, "locked": {"score": 25000, "fast": 5000}}
MANIFESTS = CAT / "manifests"


def domain_weights(n_by_domain: dict[str, int], cap: float = DOMAIN_CAP) -> dict[str, float]:
    """Water-filling: weight proportional to the number of datasets, no domain above cap."""
    if len(n_by_domain) * cap < 1 - 1e-12:
        raise ValueError(f"{len(n_by_domain)} domains cannot satisfy a {cap:.0%} cap")
    w, fixed = {}, set()
    while True:
        free = {d: n for d, n in n_by_domain.items() if d not in fixed}
        left = 1.0 - cap * len(fixed)
        tot = sum(free.values())
        over = {d for d, n in free.items() if left * n / tot > cap + 1e-12}
        if not over:
            w.update({d: cap for d in fixed})
            w.update({d: left * n / tot for d, n in free.items()})
            return w
        fixed |= over


def series_windows(v: np.ndarray, H: int) -> list[int]:
    """Cutoff positions (start of the target) of the last N_WINDOWS valid windows of a series."""
    n = len(v)
    out = []
    for j in range(N_WINDOWS):
        c = n - (j + 1) * H
        if c < max(64, H):
            break
        if np.isfinite(v[c:c + H]).mean() >= 0.5:
            out.append(c)
    return out


def task_windows(df: pd.DataFrame, freq: str, target: str = "value") -> pd.DataFrame:
    rows = []
    hz = horizons(freq)
    for sid, g in df.groupby("series_id"):
        v = g.sort_values("timestamp")[target].to_numpy(dtype=float)
        ts = g.sort_values("timestamp")["timestamp"].to_numpy()
        for term in TERMS:
            H = hz[term]
            for j, c in enumerate(series_windows(v, H)):
                rows.append(dict(series_id=sid, term=term, H=H, window=j, cutoff=int(c), cutoff_time=ts[c], n=len(v)))
    return pd.DataFrame(rows)


def key_hash(*parts) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()


def sample(win: pd.DataFrame, weights: dict[str, float], total: int) -> pd.DataFrame:
    """Per dataset: quota = max(1, round(total * weight)) windows by key hash; window weight = dataset weight / picked."""
    out = []
    for d, g in win.groupby("dataset_id"):
        q = max(1, int(round(total * weights[d])))
        g = g.assign(_h=[key_hash(d, r.freq, r.series_id, r.term, r.window) for r in g.itertuples()])
        pick = g.sort_values("_h").head(q).drop(columns="_h")
        out.append(pick.assign(weight=weights[d] / len(pick)))
    return pd.concat(out, ignore_index=True) if out else win.assign(weight=[])


def git_head() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


def build(staging: Path, out: Path, version: str, pipeline_test: bool = False, say=print) -> dict:
    staging, vdir = Path(staging), Path(out) / version
    if vdir.exists():
        raise FileExistsError(f"{vdir} exists: dev versions are frozen")
    split = S.load().set_index("dataset_id")
    metas = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(staging.glob("*/meta.json"))]
    metas = [m for m in metas if not m.get("covariate_only") and any(t["n_series"] for t in m["tasks"].values())]
    if not metas:
        raise RuntimeError(f"no staged datasets under {staging}")
    wins, datasets = [], []
    for m in metas:
        d = m["dataset_id"]
        sp = split.loc[d]
        for key, t in m["tasks"].items():
            if not t["n_series"]:
                continue
            src = staging / m["slug"] / f"{key}.parquet"
            if sha256_file(src) != t["sha256"]:
                raise RuntimeError(f"{src} changed since ingest (sha256 differs from meta.json)")
            w = task_windows(pd.read_parquet(src), key)
            wins.append(w.assign(dataset_id=d, slug=m["slug"], freq=key, domain=m["domain"], unit=sp["unit_id"], role=sp["role"]))
        datasets.append(dict(dataset_id=d, slug=m["slug"], domain=m["domain"], unit=sp["unit_id"], role=sp["role"],
                             licence=m["licence"], tasks={k: {x: t[x] for x in ("n_series", "n_points", "nan_share", "sha256")}
                                                          for k, t in m["tasks"].items() if t["n_series"]}))
    win = pd.concat(wins, ignore_index=True)
    domains = sorted({x["domain"] for x in datasets})
    if len(domains) < MIN_DOMAINS and not pipeline_test:
        raise ValueError(f"{len(domains)} domains < {MIN_DOMAINS}: a dev version needs >= {MIN_DOMAINS} (use --pipeline-test)")
    vdir.mkdir(parents=True)
    weights, sets = {}, {}
    for role in ("tuning", "locked"):
        ds = [x for x in datasets if x["role"] == role]
        if not ds:
            continue
        n_dom = pd.Series([x["domain"] for x in ds]).value_counts().to_dict()
        try:
            dw = domain_weights(n_dom)
        except ValueError as e:
            if not pipeline_test:
                raise
            say(f"{role}: {e}; pipeline test: weights without the cap")
            dw = {k: v / len(ds) for k, v in n_dom.items()}
        w = {x["dataset_id"]: dw[x["domain"]] / n_dom[x["domain"]] for x in ds}
        weights[role] = dict(datasets=w, domains=dw)
        rw = win[win["role"] == role]
        for name, total in SIZES[role].items():
            s = sample(rw, w, total)
            s.to_parquet(vdir / f"{role}_{name}.parquet", index=False)
            sets[f"{role}_{name}"] = dict(windows=int(len(s)), sha256=sha256_file(vdir / f"{role}_{name}.parquet"))
        rw.to_parquet(vdir / f"{role}_all.parquet", index=False)
    for x in datasets:
        for k in x["tasks"]:
            dst = vdir / "data" / x["slug"] / f"{k}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(staging / x["slug"] / f"{k}.parquet", dst)
        x["weight"] = weights.get(x["role"], {}).get("datasets", {}).get(x["dataset_id"])
        x["windows"] = int((win["dataset_id"] == x["dataset_id"]).sum())
    cov = []
    for p in sorted((staging / "covtasks").glob("*/meta.json")):
        cm = json.loads(p.read_text(encoding="utf-8"))
        for f in sorted(p.parent.glob("*.parquet")):
            cw = task_windows(pd.read_parquet(f), f.stem, target="target").assign(task=cm["task"], freq=f.stem, role=cm["role"])
            cov.append(cw)
            dst = vdir / "covtasks" / cm["task"] / f.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, dst)
            shutil.copyfile(p, dst.parent / "meta.json")
    if cov:
        pd.concat(cov, ignore_index=True).to_parquet(vdir / "covtasks_windows.parquet", index=False)
    man = dict(version=version, pipeline_test=pipeline_test, created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
               git=git_head(), split_sha256=sha256_file(S.SPLIT_V1), sources_sha256=sha256_file(SOURCES),
               horizons_sha256={p.name: sha256_file(p) for p in (TIME_HORIZONS, HORIZONS_EXTRA) if p.exists()},
               rules=dict(n_windows=N_WINDOWS, domain_cap=DOMAIN_CAP, min_domains=MIN_DOMAINS, sizes=SIZES),
               n_domains=len(domains), domains=domains, weights=weights, sets=sets, datasets=datasets,
               covariate_tasks=sorted({c["task"].iloc[0] for c in cov}) if cov else [],
               totals=dict(datasets=len(datasets), series=int(sum(t["n_series"] for x in datasets for t in x["tasks"].values())),
                           points=int(sum(t["n_points"] for x in datasets for t in x["tasks"].values())),
                           windows=int(len(win))))
    (vdir / "manifest.json").write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")
    MANIFESTS.mkdir(exist_ok=True)
    (MANIFESTS / f"{version}.json").write_text(json.dumps(man, indent=1, default=str), encoding="utf-8")
    say(f"{version}: {len(datasets)} datasets, {len(domains)} domains, {man['totals']['points']:,} points, "
        f"{len(win):,} windows; sets {sets}")
    return man


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--pipeline-test", action="store_true")
    a = ap.parse_args(argv)
    build(Path(a.staging), Path(a.out), a.version, a.pipeline_test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
