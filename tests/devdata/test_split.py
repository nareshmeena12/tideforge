"""Frozen split: the committed file follows the rule, reproduces from the catalog, keeps groups together, and extending only
appends."""
import pandas as pd

from tideforge.devdata import catalog as cat
from tideforge.devdata import split


def test_committed_split_is_valid_and_reproducible():
    df = split.load()
    assert split.check(df) == []
    assert set(df["dataset_id"]) == set(cat.counted().index)
    again = split.build_v1(date="x")
    cols = [c for c in split.COLUMNS if c != "added"]
    assert again[cols].sort_values("dataset_id").reset_index(drop=True).equals(
        df[cols].sort_values("dataset_id").reset_index(drop=True))


def test_groups_share_one_role():
    df = split.load().set_index("dataset_id")
    for g, rows in cat.split_groups().groupby("group_id"):
        ids = [i for i in rows["dataset_id"] if i in df.index]
        assert df.loc[ids, "role"].nunique() == 1, g


def test_locked_quota_per_domain():
    assert [split.locked_quota(n) for n in (1, 2, 3, 4, 5, 6, 7, 10)] == [0, 1, 1, 1, 1, 2, 2, 3]
    u = split.load().drop_duplicates("unit_id")
    for d, g in u.groupby("unit_domain"):
        locked = g[g["role"] == "locked"]
        assert len(locked) == split.locked_quota(len(g)), d
        assert sorted(locked["position"]) == list(range(len(locked))), d       # the first units in hash order


def test_report_lists_tuning_only_domains():
    r = split.report(split.load())
    assert "Tuning-only domains (one unit): Physics-reference, Utilities-waste, Utilities-water." in r


def test_extend_only_appends():
    full = split.build_v1(date="v1")
    singles = full[full["unit_id"] == full["dataset_id"]]
    held = list(singles.groupby("unit_domain").head(1)["dataset_id"][:6])         # pretend these came later
    grouped = full[(full["unit_id"] != full["dataset_id"])].drop_duplicates("unit_id")["dataset_id"][:2].tolist()
    existing = full[~full["dataset_id"].isin(held + grouped)]
    existing = split.assign(existing[["dataset_id", "unit_id", "unit_domain"]]).assign(added="v1")[split.COLUMNS]
    new = split.extend(existing, sorted(cat.counted().index), date="v2")
    assert set(new["dataset_id"]) == set(held + grouped)
    both = pd.concat([existing, new], ignore_index=True)
    assert split.check(both) == []
    old = existing.set_index("dataset_id")
    assert both.set_index("dataset_id").loc[old.index, ["role", "position"]].equals(old[["role", "position"]])
