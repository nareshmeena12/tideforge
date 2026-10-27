"""Fetch -> select -> read -> regularise -> clean -> canonical parquet + meta.json (DATA_COLLECTION.md steps 4-6).

    python -m tideforge.devdata.ingest probe  --ids batch1                       # first bytes of each source's index / first file
    python -m tideforge.devdata.ingest fetch  --ids batch1 --raw $TF_DATA/raw   [--max-files N]
    python -m tideforge.devdata.ingest ingest --ids batch1 --raw $TF_DATA/raw --out $TF_DATA/staging
    python -m tideforge.devdata.ingest status --ids batch1 --raw $TF_DATA/raw --out $TF_DATA/staging

Selection: at most max_series (500) series per dataset, a frozen sample: series ranked by sha256("<dataset id>/<series id>")
within strata, the 500 split over strata in proportion to their sizes (largest remainder). With an index, max_series x
oversample candidates are fetched; after cleaning the first max_series survivors (same ranking) are kept.
Per output frequency: series resampled to the grid (bins without data stay NaN; event counts are 0 between the first and the
last event), the last 16,384 points kept, then dropped if shorter than 3 x the longest horizon + 64, more than 50% missing,
or constant. Gaps are never filled. Output data/<slug>/<freq>.parquet (series_id, timestamp UTC, value) + meta.json.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.devdata import fetch as F
from tideforge.devdata import loaders
from tideforge.devdata.freqs import canonical, min_length
from tideforge.devdata.schema import Source, load_sources, resolve_ids, validate

MAX_POINTS = 16384
MAX_MISSING = 0.5


# ---------------------------------------------------------------------------------------------------- selection
def rank_key(dataset_id: str, series: str) -> str:
    return hashlib.sha256(f"{dataset_id}/{series}".encode("utf-8")).hexdigest()


def quotas(sizes: dict, n: int) -> dict:
    """n split over groups in proportion to their sizes (largest remainder), never more than a group has."""
    total = sum(sizes.values())
    if total <= n:
        return dict(sizes)
    q = {k: min(s, math.floor(n * s / total)) for k, s in sizes.items()}
    rem = sorted(sizes, key=lambda k: (-(n * sizes[k] / total - math.floor(n * sizes[k] / total)), str(k)))
    left = n - sum(q.values())
    while left > 0:
        progressed = False
        for k in rem:
            if left and q[k] < sizes[k]:
                q[k] += 1
                left -= 1
                progressed = True
        if not progressed:
            break
    return q


def select(df: pd.DataFrame, dataset_id: str, n: int, strata: list[str], col: str = "series") -> list[str]:
    """The frozen stratified sample of df[col] (unique ids), in rank order."""
    d = df.drop_duplicates(col).copy()
    d["_r"] = [rank_key(dataset_id, s) for s in d[col].astype(str)]
    strata = [s for s in strata if s in d.columns]
    if not strata:
        return d.sort_values("_r")[col].astype(str).head(n).tolist()
    d["_s"] = d[strata].astype(str).agg("/".join, axis=1)
    q = quotas(d["_s"].value_counts().to_dict(), n)
    picked = pd.concat([g.sort_values("_r").head(q[s]) for s, g in d.groupby("_s")])
    return picked.sort_values("_r")[col].astype(str).tolist()


# ---------------------------------------------------------------------------------------------------- per series
def regularise(g: pd.DataFrame, freq: str, how: str, events: bool) -> pd.Series:
    s = g.set_index("timestamp")["value"].sort_index()
    if events:
        r = s.resample(freq).sum(min_count=0) if how != "count" else s.resample(freq).count()
        return r.astype(float)
    if how == "sum":
        return s.resample(freq).sum(min_count=1)
    return getattr(s.resample(freq), how)()


def clean(s: pd.Series, need: int) -> tuple[pd.Series | None, str]:
    """(kept series or None, reason). Trims leading / trailing NaN, keeps the last MAX_POINTS points."""
    v = s.to_numpy()
    ok = np.isfinite(v)
    if not ok.any():
        return None, "empty"
    s = s.iloc[int(np.argmax(ok)):len(v) - int(np.argmax(ok[::-1]))].iloc[-MAX_POINTS:]
    v = s.to_numpy()
    if len(v) < need:
        return None, "short"
    if np.isnan(v).mean() > MAX_MISSING:
        return None, "missing"
    if np.nanstd(v) == 0:
        return None, "constant"
    return s, "kept"


# ---------------------------------------------------------------------------------------------------- one source
class Run:
    def __init__(self, src: Source, raw: Path, out: Path | None = None, say=print):
        self.src, self.say = src, say
        self.dir = Path(raw) / src.slug
        self.out = Path(out) / src.slug if out else None
        self.log = self.dir / "fetch_log.jsonl"

    def index(self, fetch_missing: bool) -> pd.DataFrame | None:
        ix = self.src.index
        if not ix:
            return None
        urls = F.file_list(ix)
        if fetch_missing:
            files = F.fetch_all(urls, self.dir / "index", ix, self.log, say=self.say)
        else:
            files = [(self.dir / "index" / F.local_name(u), v) for u, v in urls]
            files = [(p, v) for p, v in files if p.exists()]
        return loaders.read_index(self.src, files)

    def candidates(self, ixdf: pd.DataFrame) -> list[str]:
        n = math.ceil(self.src.max_series * float(self.src.select.get("oversample", 1.3)))
        return select(ixdf, self.src.id, int(self.src.select.get("index_max", n)), list(self.src.index.get("strata", [])))

    def urls(self, fetch_missing: bool = True) -> list[tuple[str, dict]]:
        series = None
        if self.src.index:
            series = self.candidates(self.index(fetch_missing))
        return F.file_list(self.src.fetch, series)

    def fetch(self, max_files: int | None = None) -> list[tuple[Path, dict]]:
        urls = self.urls(True)
        self.say(f"{self.src.id} {self.src.slug}: {len(urls)} files")
        return F.fetch_all(urls, self.dir / "files", self.src.fetch, self.log, max_files, say=self.say)

    def files(self) -> list[tuple[Path, dict]]:
        urls = self.urls(False)
        out = [(self.dir / "files" / F.local_name(u), v) for u, v in urls]
        return [(p, v) for p, v in out if p.exists()]

    def ingest(self) -> dict:
        src = self.src
        files = self.files()
        if not files:
            raise RuntimeError(f"{src.id}: no raw files under {self.dir} (run fetch first)")
        df = loaders.read_files(src, files)
        if src.loader != "table" and (src.exclude or {}).get("where"):
            df = df.query(src.exclude["where"], engine="python")
        strata = list(src.select.get("strata", []) or [])
        ids = df[["series_id"] + [c for c in strata if c in df.columns]].drop_duplicates("series_id")
        order = select(ids, src.id, len(ids), strata, col="series_id")        # every series, in frozen rank order
        meta = self.meta_base(files)
        self.out.mkdir(parents=True, exist_ok=True)
        groups = dict(tuple(df.groupby("series_id")))
        for task in src.tasks:
            key = canonical(task["freq"])
            need = min_length(task["freq"])
            kept, why = {}, {"empty": 0, "short": 0, "missing": 0, "constant": 0, "over_cap": 0}
            for sid in order:
                s = regularise(groups[sid], task["freq"], task.get("how", "mean"), bool(src.events))
                s, reason = clean(s, need)
                if s is None:
                    why[reason] += 1
                    continue
                kept[sid] = s
            final = select(ids[ids["series_id"].isin(kept)], src.id, src.max_series, strata, col="series_id")
            why["over_cap"] = len(kept) - len(final)
            parts = [pd.DataFrame({"series_id": sid, "timestamp": kept[sid].index, "value": kept[sid].to_numpy()})
                     for sid in sorted(final)]
            outdf = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=loaders.LONG)
            path = self.out / f"{key}.parquet"
            outdf.to_parquet(path, index=False)
            v = outdf["value"].to_numpy(dtype=float)
            meta["tasks"][key] = dict(freq=key, how=task.get("how", "mean"), min_length=need, n_series=len(final),
                                      n_points=int(len(v)), nan_share=float(np.isnan(v).mean()) if len(v) else None,
                                      dropped=why, sha256=sha256_file(path))
            self.say(f"  {src.id} {key}: {len(final)} series, {len(v):,} points, dropped {why}")
        (self.out / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        return meta

    def meta_base(self, files) -> dict:
        src, row = self.src, self.src.row
        log = {}
        if self.log.exists():
            for line in self.log.read_text(encoding="utf-8").splitlines():
                r = json.loads(line)
                if r.get("sha256"):
                    log[r["file"]] = r
        return dict(dataset_id=src.id, slug=src.slug, name=row["name"], domain=src.domain, licence=src.licence,
                    url=row["url"], overlap=row["overlap"], exclude=src.exclude, covariate_only=src.covariate_only,
                    retrieved=sorted({log[p.name]["time"][:10] for p, _ in files if p.name in log}),
                    raw_files=[dict(file=p.name, sha256=log.get(p.name, {}).get("sha256"), url=log.get(p.name, {}).get("url"))
                               for p, _ in files],
                    rules=dict(max_series=src.max_series, max_points=MAX_POINTS, max_missing=MAX_MISSING,
                               min_length="3 x longest horizon + 64", gaps="kept as NaN"),
                    ingested=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), tasks={})


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "fetch", "ingest", "status", "validate"])
    ap.add_argument("--ids", default="batch1")
    ap.add_argument("--raw")
    ap.add_argument("--out")
    ap.add_argument("--max-files", type=int)
    a = ap.parse_args(argv)
    srcs, batches = load_sources()
    ids = resolve_ids(a.ids, srcs, batches)
    bad = [p for i in ids for p in validate(srcs[i])]
    if a.cmd == "validate" or (bad and a.cmd != "probe"):      # probe runs anyway: it is how entries get completed
        print("\n".join(bad) if bad else f"OK: {len(ids)} sources valid")
        return 1 if bad else 0
    if bad:
        print("NOT VALID YET (the probe output is used to complete these):\n  " + "\n  ".join(bad))
    rc = 0
    for i in ids:
        src = srcs[i]
        try:
            if a.cmd == "probe":
                print(f"\n=== {i} {src.slug}  (confirm: {src.confirm})")
                if src.index:
                    u = F.file_list(src.index)[0][0]
                    print(f"index {F.redact(u, src.index.get('auth'))}\n{F.probe(u, src.index)}")
                try:
                    pv = src.fetch.get("probe_vars")       # an example series for a per-series template
                    u = (F.render(src.fetch["url_template"], pv) if pv else
                         F.file_list(src.fetch, ["<series>"] if src.index else None)[0][0])
                    print(f"first file {F.redact(u, src.fetch.get('auth'))}\n{F.probe(u, src.fetch)}")
                except Exception as e:      # noqa: BLE001
                    print(f"first file: {type(e).__name__}: {e}")
            elif a.cmd == "fetch":
                Run(src, Path(a.raw)).fetch(a.max_files)
            elif a.cmd == "ingest":
                Run(src, Path(a.raw), Path(a.out)).ingest()
            else:
                d = Path(a.raw) / src.slug
                n = len(list((d / "files").glob("*"))) if (d / "files").exists() else 0
                m = Path(a.out or ".") / src.slug / "meta.json"
                t = json.loads(m.read_text())["tasks"] if m.exists() else {}
                print(f"{i} {src.slug:24s} raw files {n:6d}  outputs " +
                      ", ".join(f"{k}: {v['n_series']} series {v['n_points']:,} pts" for k, v in t.items()))
        except Exception as e:      # noqa: BLE001
            rc = 1
            print(f"{i} FAILED: {type(e).__name__}: {e}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
