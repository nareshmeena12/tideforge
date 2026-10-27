"""The committed catalog files (catalog/): dataset list, benchmark exclusions and lists, split groups, aggregator rules."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CAT = REPO / "catalog"
CATALOG = CAT / "dataset_catalog_v2.csv"
EXCLUDED = CAT / "excluded_benchmark.csv"
BENCHMARKS = CAT / "benchmark_datasets.csv"
SPLIT_GROUPS = CAT / "split_groups.csv"
AGGREGATOR_RULES = CAT / "aggregator_rules.csv"
AUDIT_DECISIONS = CAT / "audit_decisions.csv"


@lru_cache(maxsize=None)
def catalog() -> pd.DataFrame:
    d = pd.read_csv(CATALOG, dtype=str).fillna("")
    assert d["id"].is_unique, "duplicate catalog ids"
    return d.set_index("id", drop=False)


def counted() -> pd.DataFrame:
    """The datasets that count towards the dev set (catalog column counted == yes)."""
    c = catalog()
    return c[c["counted"] == "yes"]


@lru_cache(maxsize=None)
def excluded() -> pd.DataFrame:
    return pd.read_csv(EXCLUDED, dtype=str).fillna("")


@lru_cache(maxsize=None)
def benchmarks() -> pd.DataFrame:
    return pd.read_csv(BENCHMARKS, dtype=str).fillna("")


@lru_cache(maxsize=None)
def split_groups() -> pd.DataFrame:
    g = pd.read_csv(SPLIT_GROUPS, dtype=str).fillna("")
    dup = g["dataset_id"][g["dataset_id"].duplicated()]
    assert dup.empty, f"datasets in more than one split group: {sorted(dup)}"
    unknown = sorted(set(g["dataset_id"]) - set(catalog().index))
    assert not unknown, f"split_groups.csv names datasets not in the catalog: {unknown}"
    return g


@lru_cache(maxsize=None)
def aggregator_rules() -> pd.DataFrame:
    r = pd.read_csv(AGGREGATOR_RULES, dtype=str).fillna("")
    unknown = sorted((set(r["aggregator_id"]) | set(r["source_id"])) - set(catalog().index))
    assert not unknown, f"aggregator_rules.csv names datasets not in the catalog: {unknown}"
    return r
