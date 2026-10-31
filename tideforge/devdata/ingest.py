"""Fetch -> select -> read -> regularise -> clean -> canonical parquet + meta.json (DATA_COLLECTION.md steps 4-6).

    python -m tideforge.devdata.ingest probe  --ids batch1                       # first bytes of each source's index / first file
    python -m tideforge.devdata.ingest fetch  --ids batch1 --raw $TF_DATA/raw   [--max-files N] [--retry-missing]
    python -m tideforge.devdata.ingest ingest --ids batch1 --raw $TF_DATA/raw --out $TF_DATA/staging
    python -m tideforge.devdata.ingest status --ids batch1 --raw $TF_DATA/raw --out $TF_DATA/staging
    python -m tideforge.devdata.ingest register --ids D230 --raw RAW     # log hand-placed files (fetch.manual sources)
    python -m tideforge.devdata.ingest verify   --ids D230 --raw RAW     # every logged file is there with its sha256
Files fetched or placed on another machine (e.g. the owner's PC for sources NERSC cannot reach) are uploaded as the whole
<raw>/<slug>/ folder (files/ + fetch_log.jsonl), unchanged; `verify` on Perlmutter then checks every sha256 of the log.

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
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.devdata import fetch as F
from tideforge.devdata.country import series_country
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
    from tideforge.devdata.freqs import resample_rule
    freq = resample_rule(freq)                                   # index time: the synthetic 1-s step
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
def _log_rows(log: Path) -> list[dict]:
    if not log.exists():
        return []
    return [json.loads(x) for x in log.read_text(encoding="utf-8").splitlines() if x.strip()]


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

    def fetch(self, max_files: int | None = None, retry_missing: bool = False) -> list[tuple[Path, dict]]:
        if self.src.fetch.get("manual"):
            files = self.files()
            self.say(f"{self.src.id} {self.src.slug}: manual download, {len(files)} files in {self.dir / 'files'}")
            return files
        urls = self.urls(True)
        self.say(f"{self.src.id} {self.src.slug}: {len(urls)} files")
        return F.fetch_all(urls, self.dir / "files", self.src.fetch, self.log, max_files, say=self.say,
                           retry_missing=retry_missing)

    def register(self) -> list[dict]:
        """fetch_log rows for files placed in <raw>/<slug>/files/ by hand (manual sources): url manual:<name>, bytes, sha256."""
        done = {r.get("file") for r in _log_rows(self.log)}
        rows = []
        for p in sorted((self.dir / "files").glob("*")):
            if p.is_file() and p.name not in done and not p.name.endswith(".part"):
                rows.append({"url": f"manual:{p.name}", "file": p.name, "bytes": p.stat().st_size, "sha256": sha256_file(p),
                             "status": "ok", "time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")})
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "a", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        self.say(f"{self.src.id} {self.src.slug}: registered {len(rows)} hand-placed file(s); {len(done)} already in the log")
        return rows

    def verify(self) -> int:
        """0 if every file with a sha256 in the log exists with that sha256; 1 otherwise (printed)."""
        bad, n = [], 0
        for r in _log_rows(self.log):
            if not r.get("sha256"):
                continue
            n += 1
            p = self.dir / "files" / r["file"]
            if not p.exists():
                bad.append(f"missing {r['file']}")
            elif sha256_file(p) != r["sha256"]:
                bad.append(f"sha256 differs: {r['file']}")
        self.say(f"{self.src.id} {self.src.slug}: {n} logged files, " + ("all match" if not bad else f"{len(bad)} problems: "
                                                                          + "; ".join(bad[:10])))
        return 1 if bad else 0

    def files(self) -> list[tuple[Path, dict]]:
        if self.src.fetch.get("manual"):          # files put in <raw>/<slug>/files/ by hand (browser-only sources)
            return [(p, {}) for p in sorted((self.dir / "files").glob("*")) if p.is_file()]
        urls = self.urls(False)
        out = [(self.dir / "files" / F.local_name(u), v) for u, v in urls]
        return [(p, v) for p, v in out if p.exists()]

    def groups(self, files: list[tuple[Path, dict]]) -> list[list[tuple[Path, dict]]]:
        """Files in groups that each hold whole series: one group per index series when the URLs carry the series variable
        (per-station / station-year files), in frozen rank order; otherwise one group of every file."""
        var = next((k for k, v in (self.src.fetch.get("vars") or {}).items() if v == "index"), None)
        if var is None or not all(var in v for _, v in files):
            return [files] if files else []
        by: dict = {}
        for f in files:
            by.setdefault(str(f[1][var]), []).append(f)
        rank = sorted(by, key=lambda k: rank_key(self.src.id, k))
        return [by[k] for k in rank]

    def ingest(self) -> dict:
        """Memory-bounded (owner 2026-10-31): files are read one series group at a time, every cleaned series is appended to
        a per-task part file at once and only its id stays in memory; the frozen selection (select over the kept ids, rank
        order, strata quotas) then copies the chosen series to <freq>.parquet. Same output as reading everything at once."""
        import pyarrow as pa
        import pyarrow.parquet as pq
        src = self.src
        files = self.files()
        if not files:
            raise RuntimeError(f"{src.id}: no raw files under {self.dir} (run fetch first)")
        t0 = time.time()
        self.say(f"{src.id} {src.slug}: start, {len(files)} files")
        strata = list(src.select.get("strata", []) or [])
        meta = self.meta_base(files)
        self.out.mkdir(parents=True, exist_ok=True)
        schema = pa.schema([("series_id", pa.string()), ("timestamp", pa.timestamp("ns")), ("value", pa.float64())])
        tasks = [(canonical(t["freq"]), t, min_length(t["freq"])) for t in src.tasks]
        writers = {k: pq.ParquetWriter(self.out / f"{k}.parts.parquet", schema) for k, _, _ in tasks}
        kept = {k: [] for k, _, _ in tasks}
        why = {k: {"empty": 0, "short": 0, "missing": 0, "constant": 0, "over_cap": 0} for k, _, _ in tasks}
        id_rows = []
        try:
            for group in self.groups(files):
                df = loaders.read_files(src, group)
                if src.loader != "table" and (src.exclude or {}).get("where"):
                    from tideforge.devdata.loaders.tabular import where_mask
                    df = df[where_mask(df, src.exclude["where"])]
                id_rows.append(df[["series_id"] + [c for c in strata if c in df.columns]].drop_duplicates("series_id"))
                for sid, g in df.groupby("series_id", sort=True):
                    for key, task, need in tasks:
                        ser = regularise(g, task["freq"], task.get("how", "mean"), bool(src.events))
                        ser, reason = clean(ser, need)
                        if ser is None:
                            why[key][reason] += 1
                            continue
                        writers[key].write_table(pa.table({"series_id": [str(sid)] * len(ser), "timestamp": ser.index.to_numpy(),
                                                           "value": ser.to_numpy(dtype="float64")}, schema=schema))
                        kept[key].append(str(sid))
                del df
        finally:
            for w in writers.values():
                w.close()
        ids = (pd.concat(id_rows, ignore_index=True).drop_duplicates("series_id") if id_rows
               else pd.DataFrame(columns=["series_id"]))
        n_series, n_points = 0, 0
        for key, task, need in tasks:
            part = self.out / f"{key}.parts.parquet"
            final = select(ids[ids["series_id"].isin(set(kept[key]))], src.id, src.max_series, strata, col="series_id")
            why[key]["over_cap"] = len(kept[key]) - len(final)
            tab = pq.read_table(part, filters=[("series_id", "in", final)] if final else None)
            outdf = tab.to_pandas() if final else pd.DataFrame(columns=loaders.LONG)
            if len(outdf):
                outdf = outdf.sort_values(["series_id", "timestamp"], kind="stable").reset_index(drop=True)
            part.unlink()
            path = self.out / f"{key}.parquet"
            outdf.to_parquet(path, index=False)
            v = outdf["value"].to_numpy(dtype=float)
            meta["tasks"][key] = dict(freq=key, how=task.get("how", "mean"), min_length=need, n_series=len(final),
                                      n_points=int(len(v)), nan_share=float(np.isnan(v).mean()) if len(v) else None,
                                      dropped=why[key], sha256=sha256_file(path))
            cc = {sid: series_country(src.country, sid) for sid in sorted(final)}
            meta["series_country"].update(cc)
            meta["tasks"][key]["countries"] = dict(sorted(Counter(cc.values()).items()))
            n_series, n_points = n_series + len(final), n_points + len(v)
            self.say(f"  {src.id} {key}: {len(final)} series, {len(v):,} points, dropped {why[key]}")
        (self.out / "meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
        self.say(f"{src.id} {src.slug}: done, {n_series} series, {n_points:,} points, {time.time() - t0:.0f} s, peak RSS "
                 f"{peak_rss_mb()}")
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
                    ingested=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), tasks={},
                    country=src.country, series_country={})


def peak_rss_mb() -> str:
    """The process's peak resident memory (Linux / macOS; 'n/a' where the resource module is missing, e.g. Windows)."""
    try:
        import resource
    except ImportError:
        return "n/a"
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return f"{r / 1024 / (1024 if sys.platform == 'darwin' else 1):,.0f} MB"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "fetch", "ingest", "status", "validate", "register", "verify"])
    ap.add_argument("--ids", default="batch1")
    ap.add_argument("--raw")
    ap.add_argument("--out")
    ap.add_argument("--max-files", type=int)
    ap.add_argument("--retry-missing", action="store_true", help="fetch: request files logged as missing (404) again")
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
                Run(src, Path(a.raw)).fetch(a.max_files, a.retry_missing)
            elif a.cmd == "register":
                Run(src, Path(a.raw)).register()
            elif a.cmd == "verify":
                rc |= Run(src, Path(a.raw)).verify()
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
