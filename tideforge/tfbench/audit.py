"""A5 leakage audit against fev-bench, GIFT-Eval and TIME. Reuses tideforge.devdata.audit (z-scored 64-point windows,
double binning, benchmark stride 16, dev stride 1, Pearson r > 0.999 verification, aggregation to coarser benchmark
frequencies).

Two rules (owner 2026-11-12), both reported:
  series level   a series with more than audit_series_share (10%) of its windows matched is dropped and listed;
  dataset level  a dataset with more than audit_dataset_share (1%) of all its windows matched, after the series drops,
                 is flagged and excluded until the owner decides.
audit.csv also keeps the old rule (>= 3 verified windows) as a column.
Windows are counted at the benchmark stride (16) on both sides: the benchmark is hashed every 16 points, so a full copy
matches one dev window in 16 at stride 1; share = verified matches / ((length - 64) // 16 + 1), capped at 1.
A series' share is the best over its variants (native, and each coarser benchmark frequency it aggregates to).

Outputs ($TFB_DATA/audit/): refs_db/ (fingerprints), <dataset_id>.json, audit.csv, series_drops.csv,
excluded_audit.csv. Canonical data is never rewritten: load_clean() applies the series drops.
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.devdata import audit as A
from tideforge.tfbench.config import Settings
from tideforge.tfbench.readers import Unreadable, open_parts

log = logging.getLogger("tfbench.audit")
REF_EXT = {"parquet", "arrow", "csv", "tsv", "json", "jsonl", "csv.gz", "zip", "tar.gz"}


# ---- reference database ----------------------------------------------------------------------------------------------
def ref_series(root: Path, bench: str, workdir: Path):
    """Every series of one downloaded reference benchmark: (values, meta). Lists = one series per row (2-D lists: one per
    channel); flat tables = one series per numeric column per id (or per column when there is no id)."""
    from tideforge.tfbench.canon import _kind, roles_for
    from tideforge.tfbench.download import ext
    for f in sorted(p for p in Path(root).rglob("*") if p.is_file() and ext(p.name) in REF_EXT):
        rel = f.relative_to(root).as_posix()
        try:
            parts = open_parts(f, workdir, rel)
        except Unreadable:
            continue
        for p in parts:
            lists = [x.name for x in p.schema if _kind(x.type)[0] == "list_num"]
            fcol = next((c for c in ("freq", "frequency") if c in p.schema.names), None)
            if lists:
                for t in p.batches(lists + ([fcol] if fcol else [])):
                    for j, row in enumerate(t.to_pylist()):
                        for c in lists:
                            if row[c] is None:
                                continue
                            a = np.asarray(row[c], dtype=np.float64)
                            for v, y in enumerate(a[None] if a.ndim == 1 else a):
                                yield y, dict(bench=bench, dataset=p.name, item=f"{j}/{c}/{v}", freq=str(row.get(fcol) or ""))
                continue
            try:
                r, _, _ = roles_for(p.schema, {})
            except Exception:
                continue
            num = [x.name for x in p.schema if _kind(x.type)[0] == "num" and x.name not in r["id_cols"]]
            if not num:
                continue
            d = p.read().to_pandas()
            groups = d.groupby(r["id_cols"]) if r["id_cols"] else [("all", d)]
            for gid, g in groups:
                if r["time_col"] and r["time_col"] in g:
                    g = g.sort_values(r["time_col"])
                for c in num:
                    yield pd.to_numeric(g[c], errors="coerce").to_numpy(np.float64), dict(
                        bench=bench, dataset=p.name, item=f"{gid}/{c}", freq="")


def build_refs_db(refs_root: Path, out: Path, workdir: Path, benches: dict) -> A.DB:
    """benches: {bench name: folder under refs_root}."""
    db = A.DB()
    for bench, sub in benches.items():
        n0 = len(db.values)
        for y, m in ref_series(Path(refs_root) / sub, bench, workdir):
            db.add(y, **m)
        log.info("refs %s: %d series", bench, len(db.values) - n0)
    db.build()
    db.save(out)
    return db


# ---- one dataset ------------------------------------------------------------------------------------------------------
def _variants(ts: pd.Series, freq: str, db: A.DB):
    yield freq, ts.to_numpy()
    for rf in sorted(f for f in db.freqs() if f):
        try:
            a = A.aggregate(ts, freq, rf)
        except Exception:
            a = None
        if a is not None and len(a) >= A.W:
            yield rf, a


def audit_dataset(canon_dir: Path, db: A.DB, s: Settings) -> dict:
    meta = json.loads((Path(canon_dir) / "meta.json").read_text(encoding="utf-8"))
    data = pd.read_parquet(Path(canon_dir) / "data.parquet")
    freq = meta["frequency"]
    rows = []
    old_windows = 0
    for iid, g in data.groupby("item_id", sort=False):
        g = g.set_index("timestamp")
        best = dict(item_id=iid, windows=0, matched=0, share=0.0, at_freq=freq, matched_task="")
        for tc in meta["targets"]:
            for f, x in _variants(g[tc], freq, db):
                if len(x) < A.W:
                    continue
                n = (len(x) - A.W) // A.BENCH_STRIDE + 1   # windows at the benchmark stride (see module docstring)
                m = A.match_series(np.asarray(x, dtype=np.float64), db, freq=f)   # same frequency band only
                old_windows += len(m)
                share = min(len(m) / n, 1.0)
                if f == freq and tc == meta["targets"][0] and best["windows"] == 0:
                    best.update(windows=n)
                if m and share > best["share"]:
                    tasks = Counter(f"{db.meta[k['ref']]['bench']}:{db.meta[k['ref']]['dataset']}" for k in m)
                    best.update(matched=len(m), share=share, at_freq=f, windows=n, matched_task=tasks.most_common(1)[0][0])
        rows.append(best)
    per = pd.DataFrame(rows)
    per["dropped"] = per["share"] > s.audit_series_share
    keep = per[~per["dropped"]]
    tot = keep["windows"].sum()
    ds_share = float(keep["matched"].sum() / tot) if tot else 0.0
    tasks = Counter(keep.loc[keep["matched"] > 0, "matched_task"])
    out = dict(dataset_id=meta["dataset_id"], series=len(per), series_dropped=int(per["dropped"].sum()),
               windows=int(per["windows"].sum()), matched_windows=int(per["matched"].sum()),
               share_after_drops=ds_share, flagged=ds_share > s.audit_dataset_share,
               old_rule_windows=int(old_windows), old_rule_flag=old_windows >= A.FLAG_WINDOWS,
               matched_task=(tasks.most_common(1)[0][0] if tasks else ""),
               dropped_series=per.loc[per["dropped"], ["item_id", "share", "matched_task"]].to_dict("records"))
    return out


def load_clean(canon_dir: Path, audit_dir: Path) -> pd.DataFrame:
    """Canonical data without the series the audit dropped."""
    data = pd.read_parquet(Path(canon_dir) / "data.parquet")
    a = Path(audit_dir) / f"{Path(canon_dir).name}.json"
    if a.exists():
        drop = {d["item_id"] for d in json.loads(a.read_text(encoding="utf-8"))["dropped_series"]}
        data = data[~data["item_id"].isin(drop)]
    return data


def summarise(audit_dir: Path) -> pd.DataFrame:
    rows, drops = [], []
    for f in sorted(Path(audit_dir).glob("*.json")):
        a = json.loads(f.read_text(encoding="utf-8"))
        drops += [{"dataset_id": a["dataset_id"], **d} for d in a.pop("dropped_series")]
        rows.append(a)
    df = pd.DataFrame(rows)
    df.to_csv(Path(audit_dir) / "audit.csv", index=False)
    pd.DataFrame(drops, columns=["dataset_id", "item_id", "share", "matched_task"]).to_csv(
        Path(audit_dir) / "series_drops.csv", index=False)
    if not df.empty:
        df[df["flagged"]][["dataset_id", "share_after_drops", "matched_task"]].to_csv(
            Path(audit_dir) / "excluded_audit.csv", index=False)
    return df
