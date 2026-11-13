"""B2 split by whole source group, written once to splits.json and never regenerated once published (owner 2026-11-12;
source groups owner 2026-11-14).

Unit = source group (devdata/tfbench/source_groups.yaml; groups sharing a dataset are merged; an unlisted dataset is its
own group). First build: within each domain (a group's most common member domain), groups sorted by sha256(group);
every 4th (rank % 4 == 3) -> locked, the rest -> tuning; every member takes its group's split. Later builds are
append-only: a new dataset joins its group's split when the group is already there, else the group goes to locked when
int(sha256(group)) % 4 == 0; datasets already there never move. A splits.json made before source groups (no
"unit": "source_group") is regenerated once (it was never published) and the old file kept as splits_by_dataset.json.

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


def group_map(groups: dict[str, list[str]] | None) -> dict[str, str]:
    """{group: [dataset_id]} (source_groups.yaml) -> {dataset_id: group}; groups sharing a dataset are merged (the merged
    name joins the sorted group names with '+')."""
    parent: dict[str, str] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for g, ids in (groups or {}).items():
        for d in ids or []:
            parent[find("g:" + g)] = find("d:" + d)
    members: dict[str, list[str]] = {}
    for k in list(parent):
        members.setdefault(find(k), []).append(k)
    out = {}
    for ks in members.values():
        name = "+".join(sorted(k[2:] for k in ks if k.startswith("g:")))
        for k in ks:
            if k.startswith("d:"):
                out[k[2:]] = name
    return out


def assign(datasets: dict[str, str], existing: dict | None = None, groups: dict[str, list[str]] | None = None) -> dict:
    """datasets: {dataset_id: domain}. existing: a loaded splits.json or None. groups: source_groups.yaml content.
    -> splits.json content (every dataset has source_group)."""
    now = dt.date.today().isoformat()
    gm = group_map(groups)
    grp = {d: gm.get(d, d) for d in datasets}
    if not existing:
        out = {"rule": "first build: unit = source group; per domain, sort groups by sha256(group), rank % 4 == 3 -> "
                       "locked; appended later: a dataset joins its group's split, a new group goes to locked when "
                       "int(sha256(group)) % 4 == 0; never regenerated once published",
               "unit": "source_group", "created": now, "datasets": {}}
        members: dict[str, list[str]] = {}
        for d, g in grp.items():
            members.setdefault(g, []).append(d)
        by_dom: dict[str, list[str]] = {}
        for g, ds in members.items():
            doms = pd.Series([str(datasets[d]) for d in sorted(ds)])
            by_dom.setdefault(str(doms.mode().sort_values().iloc[0]), []).append(g)
        for dom, gs in by_dom.items():
            for i, g in enumerate(sorted(gs, key=h)):
                for d in members[g]:
                    out["datasets"][d] = {"split": "locked" if i % 4 == 3 else "tuning", "domain": str(datasets[d]),
                                          "source_group": g, "group_domain": dom, "added": now, "how": "rank"}
        return out
    out = json.loads(json.dumps(existing))
    have = {v.get("source_group", d): v["split"] for d, v in out["datasets"].items()}
    for d, dom in datasets.items():
        if d not in out["datasets"]:
            g = grp[d]
            sp, how = (have[g], "append, joins its group") if g in have else (
                "locked" if int(h(g), 16) % 4 == 0 else "tuning", "append")
            have[g] = sp
            out["datasets"][d] = {"split": sp, "domain": str(dom), "source_group": g, "added": now, "how": how}
    return out


def load_or_create(path: Path, datasets: dict[str, str], groups: dict[str, list[str]] | None = None) -> dict:
    path = Path(path)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if existing is not None and existing.get("unit") != "source_group":      # made before source groups, unpublished
        path.with_name("splits_by_dataset.json").write_text(json.dumps(existing, indent=1, sort_keys=True),
                                                            encoding="utf-8")
        existing = None
    out = assign(datasets, existing, groups)
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
