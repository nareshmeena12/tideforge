"""B1 tasks of one source dataset, in fev's task format.

Input: canon/<id> without the audit's dropped series and without cross-dataset copies (split.crossdups).
Series subset: one per dataset (canon's seeded sample minus those drops), recorded in subset.json, shared by every
variant. Inside one frequency variant and horizon, covariates on / off and joint tasks use the same series file:
  - a series too short for the variant is dropped from the variant;
  - the short horizon keeps every series that has min_context + one H; medium / long tasks (owner 2026-11-12) get their
    own series file with only the series long enough for them (config <freq config>__medium / __long; the short file is
    reused when the series are the same), created when >= horizon_min_share (75%) of the variant's series qualify;
  - NaN is allowed in context only: per horizon file, series with NaN in any target inside that horizon's evaluation
    span (n_windows x H) are dropped; so are series whose in-sample seasonal-difference scale is zero (MASE undefined).
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
        rec["horizons"] = {}
        if short_ok.empty:
            rec["skipped"] = "no series long enough"
            record["variants"].append(rec)
            continue
        m = HZ.seasonality(alias)
        written: dict[tuple, str] = {}           # series set -> config already written (medium reuses short's file)
        for cls, H in hz.items():
            ok = short_ok[short_ok >= HZ.context_needed(alias, H, s) + H]
            hr = {"H": int(H), "too_short": int(len(short_ok) - len(ok))}
            rec["horizons"][cls] = hr
            if cls != "short" and len(ok) < s.horizon_min_share * len(short_ok):
                hr["skipped"] = f"{len(ok)} of {len(short_ok)} series long enough (< {s.horizon_min_share:.0%})"
                continue
            d = df[df["item_id"].isin(ok.index)]
            nw = HZ.n_windows(ok.to_numpy(), H, s)
            d, drops = eligible(d, targets, nw * H, m)
            hr.update(drops)
            if d.empty:
                hr["skipped"] = "no series left after the NaN / scale rules"
                continue
            vcols = targets + [c for c in covs if c in d.columns]
            vcols = [c for c in vcols if c in targets or d[c].notna().any()]
            hl = d.groupby("item_id").size()
            key = tuple(sorted(map(str, hl.index)))
            if key in written:
                hcfg = written[key]
            else:
                hcfg = cfg if cls == "short" else f"{cfg}__{cls}"
                pq.write_table(to_fev(d, vcols, static), out / f"{hcfg}.parquet", compression="zstd")
                written[key] = hcfg
            hr.update(config=hcfg, series=int(len(hl)), points=int(len(d)), n_windows=int(nw), series_ids=list(key))
            y = d[targets].to_numpy(np.float64)
            past = [c for c, r in covs.items() if r != "known" and c in vcols]
            known = [c for c, r in covs.items() if r == "known" and c in vcols]
            tags0 = dict(domain=meta.get("domain"), band=rules.band(alias), freq=alias, intermittency=round(intermittency(y), 3),
                         series_length_band=len_band(float(hl.median())), horizon=cls, multivariate=False,
                         history_length_band=len_band(max(float(hl.median()) - nw * H, 0)))
            common = dict(dataset_path=dataset_path, dataset_config=hcfg, horizon=int(H), num_windows=int(nw),
                          window_step_size=int(H), min_context_length=int(HZ.context_needed(alias, H, s)), seasonality=int(m),
                          eval_metric="MASE", extra_metrics=["WQL", "SQL"], quantile_levels=list(s.quantile_levels),
                          id_column="id", timestamp_column="timestamp")
            base = dict(dataset_id=did, config=hcfg, freq_config=cfg, horizon_class=cls, n_windows=int(nw))
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
                tasks.append(dict(base, task=t, variant=name, multivariate=False,
                                  n_series=int(len(hl)) * (len(targets) if multivar else 1),
                                  tags=dict(tags0, covariates="known" if (name == "cov" and k_) else
                                            "past" if name == "cov" else "none")))
            if multivar:
                t = dict(common, target=list(targets), past_dynamic_columns=[], known_dynamic_columns=[], static_columns=[],
                         task_name=f"{cfg}/{cls}/joint")
                tasks.append(dict(base, task=t, variant="joint", multivariate=True, n_series=int(len(hl)),
                                  tags=dict(tags0, covariates="none", multivariate=True)))
        if "short" in rec["horizons"] and "series" in rec["horizons"]["short"]:
            rec.update(series=rec["horizons"]["short"]["series"], series_ids=rec["horizons"]["short"]["series_ids"])
        else:
            rec["skipped"] = rec.get("skipped") or "short horizon has no series"
        record["variants"].append(rec)
    for t in tasks:
        t["task_sha256"] = task_sha(t["task"])
    (out / "tasks.json").write_text(json.dumps(tasks, indent=1, default=str), encoding="utf-8")
    (out / "subset.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return tasks
