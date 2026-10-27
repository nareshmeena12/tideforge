"""Frozen dataset-level split (DATA_COLLECTION.md step 8, owner decisions 2026-10-27).

Unit of the split = a split group (catalog/split_groups.csv: datasets from the same provider and network, or copies of each
other) or a single dataset. A unit's domain is its members' most common domain (ties: the domain of the lowest id). Within each
domain the units are sorted by sha256(unit_id); every 4th (positions 3, 7, 11, ... counting from 0) goes to the locked set,
the rest to tuning. v1 splits the whole catalog (every counted dataset) at once, so the order in which datasets are ingested
never matters.

catalog/split_v1.csv is written once and committed; afterwards it is only appended to (extend): a new dataset of an existing
unit takes that unit's role; new units are ranked among themselves by hash within their domain and continue the domain's
position counter. Existing rows never change (check() enforces it).

    python -m tideforge.devdata.split write     # once (refuses if the file exists)
    python -m tideforge.devdata.split check     # the committed file equals the rule applied to its own rows; no gaps
    python -m tideforge.devdata.split extend    # append counted datasets that are not in the file yet
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import sys
from collections import Counter

import pandas as pd

from tideforge.devdata import catalog as cat

SPLIT_V1 = cat.CAT / "split_v1.csv"
COLUMNS = ["dataset_id", "unit_id", "unit_domain", "position", "role", "added"]
LOCKED_EVERY = 4


def h(unit_id: str) -> str:
    return hashlib.sha256(unit_id.encode("utf-8")).hexdigest()


def units(ids: list[str]) -> pd.DataFrame:
    """dataset_id -> unit_id, unit_domain for the given catalog ids."""
    c = cat.catalog()
    grp = dict(zip(cat.split_groups()["dataset_id"], cat.split_groups()["group_id"]))
    rows = [(i, grp.get(i, i)) for i in ids]
    df = pd.DataFrame(rows, columns=["dataset_id", "unit_id"])
    dom = {}
    for u, g in df.groupby("unit_id"):
        members = sorted(cat.split_groups().query("group_id == @u")["dataset_id"]) if u in set(grp.values()) else [u]
        cnt = Counter(c.loc[m, "domain"] for m in members)
        best = max(cnt.values())
        dom[u] = next(c.loc[m, "domain"] for m in members if cnt[c.loc[m, "domain"]] == best)
    df["unit_domain"] = df["unit_id"].map(dom)
    return df


def assign(df: pd.DataFrame, start: dict[str, int] | None = None) -> pd.DataFrame:
    """Rank the units of df within each domain by hash, starting at start[domain] (0 for v1); role by position."""
    start = start or {}
    u = df[["unit_id", "unit_domain"]].drop_duplicates().copy()
    u["hash"] = u["unit_id"].map(h)
    u = u.sort_values(["unit_domain", "hash"])
    u["position"] = u.groupby("unit_domain").cumcount() + u["unit_domain"].map(lambda d: start.get(d, 0))
    u["role"] = ["locked" if p % LOCKED_EVERY == LOCKED_EVERY - 1 else "tuning" for p in u["position"]]
    out = df.merge(u[["unit_id", "position", "role"]], on="unit_id")
    return out.sort_values(["unit_domain", "position", "dataset_id"]).reset_index(drop=True)


def build_v1(date: str | None = None) -> pd.DataFrame:
    out = assign(units(sorted(cat.counted().index)))
    out["added"] = date or dt.date.today().isoformat()
    return out[COLUMNS]


def load() -> pd.DataFrame:
    return pd.read_csv(SPLIT_V1, dtype={"position": int}).fillna("")


def extend(existing: pd.DataFrame, ids: list[str], date: str | None = None) -> pd.DataFrame:
    """Rows for the ids not in `existing` (to append)."""
    new = [i for i in ids if i not in set(existing["dataset_id"])]
    if not new:
        return pd.DataFrame(columns=COLUMNS)
    u = units(new)
    known = existing.drop_duplicates("unit_id").set_index("unit_id")
    old = u[u["unit_id"].isin(known.index)].copy()
    old["position"] = old["unit_id"].map(known["position"])
    old["role"] = old["unit_id"].map(known["role"])
    old["unit_domain"] = old["unit_id"].map(known["unit_domain"])
    fresh = u[~u["unit_id"].isin(known.index)]
    start = (existing.drop_duplicates("unit_id").groupby("unit_domain")["position"].max() + 1).to_dict()
    out = pd.concat([old, assign(fresh, start)] if len(fresh) else [old], ignore_index=True)
    out["added"] = date or dt.date.today().isoformat()
    return out[COLUMNS]


def check(df: pd.DataFrame) -> list[str]:
    """Problems with a split file: roles that do not follow from (position), a unit split across roles or positions, positions
    that are not 0..n-1 per domain, ids not in the catalog."""
    bad = []
    for _, r in df.iterrows():
        want = "locked" if r["position"] % LOCKED_EVERY == LOCKED_EVERY - 1 else "tuning"
        if r["role"] != want:
            bad.append(f"{r['dataset_id']}: role {r['role']} at position {r['position']}")
    for u, g in df.groupby("unit_id"):
        if g[["position", "role", "unit_domain"]].drop_duplicates().shape[0] != 1:
            bad.append(f"unit {u} has rows with different position / role / domain")
    for d, g in df.drop_duplicates("unit_id").groupby("unit_domain"):
        if sorted(g["position"]) != list(range(len(g))):
            bad.append(f"domain {d}: positions are not 0..{len(g) - 1}")
    missing = sorted(set(df["dataset_id"]) - set(cat.catalog().index))
    if missing:
        bad.append(f"ids not in the catalog: {missing}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "check", "extend"])
    a = ap.parse_args(argv)
    if a.cmd == "write":
        if SPLIT_V1.exists():
            print(f"{SPLIT_V1} exists: it is frozen; use extend")
            return 1
        df = build_v1()
        df.to_csv(SPLIT_V1, index=False, lineterminator="\n")
        print(f"wrote {SPLIT_V1}: {len(df)} datasets, {df['unit_id'].nunique()} units, "
              f"locked {int((df.drop_duplicates('unit_id')['role'] == 'locked').sum())} units")
        return 0
    df = load()
    if a.cmd == "check":
        bad = check(df)
        print("\n".join(bad) if bad else f"OK: {len(df)} datasets, {df['unit_id'].nunique()} units")
        return 1 if bad else 0
    rows = extend(df, sorted(cat.counted().index))
    if rows.empty:
        print("nothing to append")
        return 0
    pd.concat([df, rows]).to_csv(SPLIT_V1, index=False, lineterminator="\n")
    print(f"appended {len(rows)} datasets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
