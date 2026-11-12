"""B2 tiers and weights.

full: every task. core: one task per (dataset, frequency variant): medium horizon if it exists else short, covariates on
where the variant has them, univariate. fast: fast_n[split] core tasks, seeded, stratified by (domain, band, horizon):
strata are visited round-robin in a seeded order, a seeded random task from each, until the quota is reached.
weights: in each tier and split every source dataset has weight 1, shared equally by its tasks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def core(tasks: pd.DataFrame) -> pd.Series:
    """Boolean mask over tasks (columns dataset_id, config, horizon_class, variant)."""
    t = tasks[tasks["variant"] != "joint"].copy()
    t["_h"] = t["horizon_class"].map({"medium": 0, "short": 1, "long": 2})
    t["_c"] = (t["variant"] != "cov").astype(int)
    key = "freq_config" if "freq_config" in t else "config"
    pick = t.sort_values([key, "_h", "_c"]).groupby(key).head(1).index
    return tasks.index.isin(pick)


def fast(tasks: pd.DataFrame, core_mask, n: int, seed: int) -> pd.Series:
    c = tasks[core_mask].copy()
    if len(c) <= n:
        return pd.Series(core_mask, index=tasks.index)
    rng = np.random.default_rng(seed)
    c["_s"] = c["domain"].astype(str) + "|" + c["band"].astype(str) + "|" + c["horizon_class"].astype(str)
    groups = {k: list(rng.permutation(sorted(g.index))) for k, g in c.groupby("_s")}
    order = list(rng.permutation(sorted(groups)))
    picked = []
    while len(picked) < n:
        for k in order:
            if groups[k] and len(picked) < n:
                picked.append(groups[k].pop())
    return pd.Series(tasks.index.isin(picked), index=tasks.index)


def weights(tasks: pd.DataFrame, mask) -> pd.Series:
    t = tasks[mask]
    cnt = t.groupby("dataset_id")["dataset_id"].transform("size")
    return (1.0 / cnt).reindex(tasks.index)
