"""B1 tasks of one source dataset, in fev's task format.

Input: canon/<id> without the audit's dropped series and without cross-dataset copies (split.crossdups).
Series subset: one per dataset (canon's seeded sample minus those drops), recorded in subset.json, shared by every
variant. Inside one frequency variant all tasks (horizons, covariates on / off, joint) use the same series:
  - a series too short for the variant is dropped from the variant;
  - medium / long horizons exist only when >= horizon_min_share of the series are long enough for them; the shorter
    series are then dropped from the variant (so every horizon has the same series);
  - NaN is allowed in context only: series with NaN in any target inside the variant's evaluation span (the largest
    n_windows x H over its tasks) are dropped; so are series whose in-sample seasonal-difference scale is zero (MASE
    undefined).
Every drop is counted in the variant's record.

Per frequency variant x horizon: a target-only task; a with-covariates task when covariates exist (past + known_candidate
as past_dynamic, known as known_dynamic, statics); a joint multivariate task when the dataset has 2..max_channels targets
(the univariate tasks then use fev's generate_univariate_targets_from).

build(...) writes <out>/<config>.parquet per variant, <out>/tasks.json, <out>/subset.json, and returns the task records.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from tideforge.tfbench import horizons as HZ
from tideforge.tfbench import rules
from tideforge.tfbench.config import Settings
from tideforge.tfbench.variants import freq_variants, to_fev

LEN_BANDS = ((0, 100, "<100"), (100, 500, "100-500"), (500, 2000, "500-2k"), (2000, 10000, "2k-10k"),
             (10000, 10 ** 12, ">=10k"))


def len_band(n: float) -> str:
    return next(b for lo, hi, b in LEN_BANDS if lo <= n < hi)


def freq_slug(alias: str) -> str:
    return alias.lower().replace("-", "")


def seasonal_scale(y: np.ndarray, m: int) -> float:
    m = m if len(y) > m else 1
    d = np.abs(y[m:] - y[:-m])
    d = d[np.isfinite(d)]
    return float(d.mean()) if len(d) else 0.0


def pivot_wide(data: pd.DataFrame, s: Settings) -> tuple[pd.DataFrame, list[str]]:
    """Wide table with `multivariate: true`: the first max_channels items (sorted) become channels of one series."""
    items = sorted(data["item_id"].unique())[: s.max_channels]
    p = data[data["item_id"].isin(items)].pivot_table(index="timestamp", columns="item_id", values="target", dropna=False)
    names = [f"target_{k}" for k in range(len(items))]
    p.columns = names
    p = p.reset_index()
    p.insert(0, "item_id", "all")
    return p, names


def eligible(df: pd.DataFrame, targets: list[str], R: int, m: int) -> tuple[pd.DataFrame, dict]:
    keep, nan_eval, zero_scale = [], 0, 0
    for iid, g in df.groupby("item_id", sort=False):
        y = g[targets].to_numpy(np.float64)
        if np.isnan(y[-R:]).any():
            nan_eval += 1
            continue
        if any(seasonal_scale(y[:-R, k], m) == 0 for k in range(y.shape[1])):
            zero_scale += 1
            continue
        keep.append(iid)
    return df[df["item_id"].isin(keep)], {"nan_in_eval": nan_eval, "zero_scale": zero_scale}


def intermittency(y: np.ndarray) -> float:
    f = y[np.isfinite(y)]
    return float((f == 0).mean()) if len(f) else 0.0


def task_sha(t: dict) -> str:
    return hashlib.sha256(json.dumps(t, sort_keys=True, default=str).encode()).hexdigest()


def build(data: pd.DataFrame, meta: dict, ov: dict, s: Settings, out: Path, dataset_path: str = "{repo}") -> list[dict]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    did = meta["dataset_id"]
    targets = list(meta["targets"])
    multivar = len(targets) >= 2
    if ov.get("multivariate") and meta.get("layout") == "wide" and len(targets) == 1:
        data, targets = pivot_wide(data, s)
        meta = {**meta, "targets": targets, "covariates": {}, "static": []}
        multivar = True
    if len(targets) > s.max_channels:
        targets = targets[: s.max_channels]
    covs = {c: r for c, r in (meta.get("covariates") or {}).items() if c in data.columns}
    static = [c for c in meta.get("static", []) if c in data.columns]
    subset = sorted(map(str, data["item_id"].unique()))
    record = {"dataset_id": did, "seed": meta.get("seed", s.seed), "series": subset, "variants": []}
    tasks = []
    for v in freq_variants(data, {**meta, "targets": targets, "covariates": covs, "static": static}, ov, s):
        alias, df = v["freq"], v["data"]
        cfg = f"{did}__{freq_slug(alias)}"
        rec = {"config": cfg, "freq": alias, "agg": v["agg"], "dropped_short": v["dropped_short"]}
        lens = df.groupby("item_id").size()
        hz = HZ.horizons(alias)
        H0 = hz["short"]
        short_ok = lens[lens >= HZ.context_needed(alias, H0, s) + H0]
        rec["dropped_short"] += int(len(lens) - len(short_ok))
        chosen = {"short": H0}
        for cls in ("medium", "long"):
            if cls in hz:
                H = hz[cls]
                ok = short_ok[short_ok >= HZ.context_needed(alias, H, s) + H]
                if len(short_ok) and len(ok) >= s.horizon_min_share * len(short_ok):
                    chosen[cls] = H
        need = max(HZ.context_needed(alias, H, s) + H for H in chosen.values())
        ok_ids = short_ok[short_ok >= need].index
        rec["dropped_for_long_horizons"] = int(len(short_ok) - len(ok_ids))
        df = df[df["item_id"].isin(ok_ids)]
        if df.empty:
            rec["skipped"] = "no series long enough"
            record["variants"].append(rec)
            continue
        lens = df.groupby("item_id").size()
        nw = {cls: HZ.n_windows(lens.to_numpy(), H, s) for cls, H in chosen.items()}
        R = max(nw[c] * chosen[c] for c in chosen)
        m = HZ.seasonality(alias)
        df, drops = eligible(df, targets, R, m)
        rec.update(drops)
        if df.empty:
            rec["skipped"] = "no series left after the NaN / scale rules"
            record["variants"].append(rec)
            continue
        vcols = targets + [c for c in covs if c in df.columns]
        vcols = [c for c in vcols if c in targets or df[c].notna().any()]
        tbl = to_fev(df, vcols, static)
        path = out / f"{cfg}.parquet"
        pq.write_table(tbl, path, compression="zstd")
        lens = df.groupby("item_id").size()
        y = df[targets].to_numpy(np.float64)
        rec.update(series=int(len(lens)), points=int(len(df)), series_ids=sorted(map(str, lens.index)))
        record["variants"].append(rec)
        past = [c for c, r in covs.items() if r != "known" and c in vcols]
        known = [c for c, r in covs.items() if r == "known" and c in vcols]
        base_tags = dict(domain=meta.get("domain"), band=rules.band(alias), freq=alias,
                         intermittency=round(intermittency(y), 3),
                         series_length_band=len_band(float(lens.median())))
        for cls, H in chosen.items():
            ctx = HZ.context_needed(alias, H, s)
            common = dict(dataset_path=dataset_path, dataset_config=cfg, horizon=int(H), num_windows=int(nw[cls]),
                          window_step_size=int(H), min_context_length=int(ctx), seasonality=int(m),
                          eval_metric="MASE", extra_metrics=["WQL", "SQL"], quantile_levels=list(s.quantile_levels),
                          id_column="id", timestamp_column="timestamp")
            hist = float(lens.median()) - nw[cls] * H
            variants = [("target", [], [], [])]
            if past or known:
                variants.append(("cov", past, known, static))
            for name, p_, k_, st_ in variants:
                t = dict(common, past_dynamic_columns=p_, known_dynamic_columns=k_, static_columns=st_)
                if multivar:
                    t.update(target="target", generate_univariate_targets_from=list(targets))
                else:
                    t.update(target=targets[0])
                t["task_name"] = f"{cfg}/{cls}/{name}"
                tasks.append(dict(task=t, dataset_id=did, config=cfg, horizon_class=cls, variant=name, multivariate=False,
                                  n_series=int(len(lens)) * (len(targets) if multivar else 1), n_windows=int(nw[cls]),
                                  tags=dict(base_tags, horizon=cls, covariates="known" if (name == "cov" and k_) else
                                            "past" if name == "cov" else "none", multivariate=False,
                                            history_length_band=len_band(max(hist, 0)))))
            if multivar:
                t = dict(common, target=list(targets), past_dynamic_columns=[], known_dynamic_columns=[], static_columns=[],
                         task_name=f"{cfg}/{cls}/joint")
                tasks.append(dict(task=t, dataset_id=did, config=cfg, horizon_class=cls, variant="joint", multivariate=True,
                                  n_series=int(len(lens)), n_windows=int(nw[cls]),
                                  tags=dict(base_tags, horizon=cls, covariates="none", multivariate=True,
                                            history_length_band=len_band(max(hist, 0)))))
    for t in tasks:
        t["task_sha256"] = task_sha(t["task"])
    (out / "tasks.json").write_text(json.dumps(tasks, indent=1, default=str), encoding="utf-8")
    (out / "subset.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return tasks
