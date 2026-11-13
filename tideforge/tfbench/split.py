"""B2 split by whole source dataset, written once to splits.json and never regenerated (owner 2026-11-12).

First build: within each domain, datasets sorted by sha256(dataset_id); every 4th (rank % 4 == 3) -> locked, the rest
-> tuning. Later builds are append-only: a dataset not in splits.json goes to locked when int(sha256) % 4 == 0; datasets
already there never move.

Cross-dataset copies (owner 2026-11-12): every tuning series is matched against the locked series with the audit's
z-normalised window hashes (tideforge.devdata.audit, stride 16 on the locked side, stride 1 on the tuning side); a tuning
series with more than crossdup_share of its windows matched is dropped from tuning. Locked stays complete.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.devdata import audit as A


def h(dataset_id: str) -> str:
    return hashlib.sha256(dataset_id.encode("utf-8")).hexdigest()


def assign(datasets: dict[str, str], existing: dict | None = None) -> dict:
    """datasets: {dataset_id: domain}. existing: a loaded splits.json or None. -> splits.json content."""
    now = dt.date.today().isoformat()
    if not existing:
        out = {"rule": "first build: per domain, sort by sha256(dataset_id), rank % 4 == 3 -> locked; "
                       "appended later: int(sha256) % 4 == 0 -> locked; never regenerated",
               "created": now, "datasets": {}}
        by_dom: dict[str, list[str]] = {}
        for d, dom in datasets.items():
            by_dom.setdefault(str(dom), []).append(d)
        for dom, ids in by_dom.items():
            for i, d in enumerate(sorted(ids, key=h)):
                out["datasets"][d] = {"split": "locked" if i % 4 == 3 else "tuning", "domain": dom, "added": now,
                                      "how": "rank"}
        return out
    out = json.loads(json.dumps(existing))
    for d, dom in datasets.items():
        if d not in out["datasets"]:
            out["datasets"][d] = {"split": "locked" if int(h(d), 16) % 4 == 0 else "tuning", "domain": str(dom),
                                  "added": now, "how": "append"}
    return out


def load_or_create(path: Path, datasets: dict[str, str]) -> dict:
    path = Path(path)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    out = assign(datasets, existing)
    if out != existing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    return out


def force_locked(path: Path, ids: list, n: int) -> dict:
    """Smoke splits only: the first n datasets by sha256 go to locked (marked how=smoke-forced)."""
    path = Path(path)
    if path.name == "splits.json":
        raise ValueError("force_locked is for smoke splits only, never the real splits.json")
    sp = json.loads(path.read_text(encoding="utf-8"))
    have = [d for d in ids if sp["datasets"][d]["split"] == "locked"]
    for d in sorted((d for d in ids if d not in have), key=h)[: max(0, n - len(have))]:
        sp["datasets"][d].update(split="locked", how="smoke-forced")
    path.write_text(json.dumps(sp, indent=1, sort_keys=True), encoding="utf-8")
    return sp


def _series(data: pd.DataFrame, targets: list[str]):
    for iid, g in data.groupby("item_id", sort=False, observed=True):   # canon item_id is categorical
        for t in targets:
            yield str(iid), t, g[t].to_numpy(np.float64)


def build_locked_db(locked: list[tuple[str, pd.DataFrame, list[str]]]) -> A.DB:
    db = A.DB()
    for did, data, targets in locked:
        for iid, t, x in _series(data, targets):
            db.add(x, bench="tfbench-locked", dataset=did, item=f"{iid}/{t}", freq="")
    return db.build()


def crossdups(did: str, data: pd.DataFrame, targets: list[str], db: A.DB, share: float) -> list[dict]:
    """Tuning series of one dataset that copy a locked series: [{dataset_id, item_id, share, matched}]."""
    out = []
    for iid, t, x in _series(data, targets):
        if len(x) < A.W:
            continue
        n = (len(x) - A.W) // A.BENCH_STRIDE + 1
        m = A.match_series(x, db)
        sh = min(len(m) / n, 1.0)
        if sh > share:
            top = pd.Series([db.meta[k["ref"]]["dataset"] for k in m]).mode().iloc[0]
            out.append({"dataset_id": did, "item_id": iid, "share": sh, "matched": top})
    return out
