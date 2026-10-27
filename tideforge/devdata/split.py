"""Frozen dataset-level split (DATA_COLLECTION.md step 8, owner decisions 2026-10-27).

Unit of the split = a split group (catalog/split_groups.csv: datasets from the same provider and network, or copies of each
other) or a single dataset. A unit's domain is its members' most common domain (ties: the domain of the lowest id). Within each
domain the units are ranked by sha256(unit_id) (position 0, 1, ...) and the first locked_quota(n) go to the locked set:
round(0.25 * n) (halves round up), at least 1 when the domain has >= 2 units; a domain with 1 unit is tuning-only. v1 splits
the whole catalog (every counted dataset) at once, so the order in which datasets are ingested never matters.
(Owner, 2026-10-27: this rule replaced "every 4th unit" before anything was scored; split_v1.csv was regenerated.)

catalog/split_v1.csv is committed and afterwards only appended to (extend): a new dataset of an existing unit takes that
unit's role; new units continue the domain's positions in hash order, and the first of them are locked until the domain's
locked count reaches locked_quota(total units). Existing rows never change (check() enforces the counts).

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
SPLIT_REPORT = cat.CAT / "split_v1_report.md"
COLUMNS = ["dataset_id", "unit_id", "unit_domain", "position", "role", "added"]
LOCKED_SHARE = 0.25


def locked_quota(n: int) -> int:
    """Locked units of a domain with n units: round(0.25 n), halves up; >= 1 when n >= 2; 0 when n == 1."""
    if n < 2:
        return 0
    return max(1, int(LOCKED_SHARE * n + 0.5))


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


def assign(df: pd.DataFrame, start: dict[str, int] | None = None, locked_before: dict[str, int] | None = None) -> pd.DataFrame:
    """Rank the units of df within each domain by hash, starting at start[domain] (0 for v1). Locked: the first units in rank
    order until the domain holds locked_quota(start + n) locked units (locked_before[domain] of them already)."""
    start, locked_before = start or {}, locked_before or {}
    u = df[["unit_id", "unit_domain"]].drop_duplicates().copy()
    u["hash"] = u["unit_id"].map(h)
    u = u.sort_values(["unit_domain", "hash"])
    k = u.groupby("unit_domain").cumcount()
    u["position"] = k + u["unit_domain"].map(lambda d: start.get(d, 0))
    n_total = u.groupby("unit_domain")["position"].transform("max") + 1
    need = [max(0, locked_quota(int(n)) - locked_before.get(d, 0)) for d, n in zip(u["unit_domain"], n_total)]
    u["role"] = ["locked" if i < q else "tuning" for i, q in zip(k, need)]
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
    eu = existing.drop_duplicates("unit_id")
    start = (eu.groupby("unit_domain")["position"].max() + 1).to_dict()
    locked = eu[eu["role"] == "locked"].groupby("unit_domain").size().to_dict()
    out = pd.concat([old, assign(fresh, start, locked)] if len(fresh) else [old], ignore_index=True)
    out["added"] = date or dt.date.today().isoformat()
    return out[COLUMNS]


def check(df: pd.DataFrame) -> list[str]:
    """Problems with a split file: a unit split across roles / positions / domains, positions that are not 0..n-1 per domain,
    a domain whose locked count is not locked_quota(n), ids not in the catalog."""
    bad = []
    for u, g in df.groupby("unit_id"):
        if g[["position", "role", "unit_domain"]].drop_duplicates().shape[0] != 1:
            bad.append(f"unit {u} has rows with different position / role / domain")
    for d, g in df.drop_duplicates("unit_id").groupby("unit_domain"):
        if sorted(g["position"]) != list(range(len(g))):
            bad.append(f"domain {d}: positions are not 0..{len(g) - 1}")
        if (g["role"] == "locked").sum() != locked_quota(len(g)):
            bad.append(f"domain {d}: {(g['role'] == 'locked').sum()} locked units, the rule says {locked_quota(len(g))}")
    missing = sorted(set(df["dataset_id"]) - set(cat.catalog().index))
    if missing:
        bad.append(f"ids not in the catalog: {missing}")
    return bad


def report(df: pd.DataFrame) -> str:
    """Markdown: locked share overall and per domain (units and datasets), tuning-only domains."""
    u = df.drop_duplicates("unit_id")
    lines = ["# Split v1 report", "", f"Rule: per domain, units ranked by sha256(unit_id); locked_quota(n) = round(0.25 n), "
             ">= 1 when n >= 2; one-unit domains are tuning-only.", "",
             f"Overall: {(u['role'] == 'locked').sum()} of {len(u)} units locked ({(u['role'] == 'locked').mean():.1%}); "
             f"{(df['role'] == 'locked').sum()} of {len(df)} datasets ({(df['role'] == 'locked').mean():.1%}).", "",
             "| Domain | Units | Locked units | Locked share (units) | Datasets | Locked datasets |", "|---|---|---|---|---|---|"]
    for d, g in u.groupby("unit_domain"):
        dd = df[df["unit_domain"] == d]
        nl = int((g["role"] == "locked").sum())
        lines.append(f"| {d} | {len(g)} | {nl} | {nl / len(g):.0%} | {len(dd)} | {int((dd['role'] == 'locked').sum())} |")
    solo = sorted(d for d, g in u.groupby("unit_domain") if len(g) == 1)
    lines += ["", f"Tuning-only domains (one unit): {', '.join(solo) if solo else 'none'}."]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["write", "check", "extend", "report"])
    ap.add_argument("--force", action="store_true", help="replace split_v1.csv (only with the owner's OK, before any scoring)")
    a = ap.parse_args(argv)
    if a.cmd == "write":
        if SPLIT_V1.exists() and not a.force:
            print(f"{SPLIT_V1} exists: it is frozen; use extend")
            return 1
        df = build_v1()
        df.to_csv(SPLIT_V1, index=False, lineterminator="\n")
        SPLIT_REPORT.write_text(report(df), encoding="utf-8")
        print(f"wrote {SPLIT_V1}: {len(df)} datasets, {df['unit_id'].nunique()} units, "
              f"locked {int((df.drop_duplicates('unit_id')['role'] == 'locked').sum())} units")
        return 0
    df = load()
    if a.cmd == "report":
        print(report(df))
        return 0
    if a.cmd == "check":
        bad = check(df)
        print("\n".join(bad) if bad else f"OK: {len(df)} datasets, {df['unit_id'].nunique()} units")
        return 1 if bad else 0
    rows = extend(df, sorted(cat.counted().index))
    if rows.empty:
        print("nothing to append")
        return 0
    both = pd.concat([df, rows])
    both.to_csv(SPLIT_V1, index=False, lineterminator="\n")
    SPLIT_REPORT.write_text(report(both), encoding="utf-8")
    print(f"appended {len(rows)} datasets")
    return 0


if __name__ == "__main__":
    sys.exit(main())
