"""B1 tasks of one source dataset, in fev's task format.

Input: canon/<id> without the audit's dropped series and without cross-dataset copies (split.crossdups).
Series subset: one per dataset (canon's seeded sample minus those drops), recorded in subset.json, shared by every
variant. Inside one frequency variant and horizon, covariates on / off and joint tasks use the same series file:
  - a series too short for the variant is dropped from the variant;
  - the short horizon keeps every series that has min_context + one H; medium / long tasks (owner 2026-11-12) get their
    own series file with only the series long enough for them (config <freq config>__medium / __long; the short file is
    reused when the series are the same), created when >= horizon_min_share (75%) of the variant's series qualify;
  - leading / trailing rows where every target is NaN (padding) are trimmed first (trim_edges);
  - daily data never observed on weekends -> business days (B); covariates that are model outputs are dropped;
  - NaN in the evaluation span (n_windows x H) is masked by fev's metrics (owner 2026-11-14): allowed in structurally
    closed slots and up to eval_nan_max (10%) otherwise, each window keeps >= window_min_finite; near-constant contexts
    are dropped (MASE undefined);
  - tasks with n_series x n_windows x H < min_task_points, or a NaN / infinite / < 0.01 seasonal-naive score, are
    dropped (record['dropped_tasks']).
Every drop is counted in the variant's record.

Per frequency variant x horizon: a target-only task; a with-covariates task when covariates exist (past + known_candidate
as past_dynamic, known as known_dynamic, statics); a joint multivariate task when the dataset has 2..max_channels targets
(the univariate tasks then use fev's generate_univariate_targets_from).

build(...) writes <out>/<config>.parquet per variant, <out>/tasks.json, <out>/subset.json, and returns the task records.
"""
from __future__ import annotations

import hashlib
import json
import re
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


def slots(ts: pd.Series, alias: str) -> np.ndarray:
    """Calendar slot of each timestamp: time of day for sub-daily data, weekday for daily data, else one slot."""
    b = rules.band(alias)
    t = pd.DatetimeIndex(ts)
    if b in ("subhourly", "hourly", "subdaily"):
        return np.asarray(t.hour * 60 + t.minute)
    if b == "daily":
        return np.asarray(t.dayofweek)
    return np.zeros(len(t), int)


def closed_slot_set(fin: np.ndarray, sl: np.ndarray, share: float) -> set:
    """Slots that are NaN on >= share of the context rows (a structural closure: nights, weekends)."""
    if not len(sl):
        return set()
    u, inv = np.unique(sl, return_inverse=True)
    nan_share = 1 - np.bincount(inv, weights=fin.astype(float), minlength=len(u)) / np.bincount(inv, minlength=len(u))
    return set(u[nan_share >= share].tolist())


def near_constant(y: np.ndarray, m: int, max_distinct: int = 2) -> bool:
    """Zero seasonal-difference scale, or at most `max_distinct` distinct finite values in the context. Intermittent
    series (mostly zeros) are kept (owner 2026-11-14); tasks with a degenerate baseline go by baseline_filter."""
    m = m if len(y) > m else 1
    d = y[m:] - y[:-m]
    d = d[np.isfinite(d)]
    if not len(d) or not np.abs(d).mean():
        return True
    return len(np.unique(y[np.isfinite(y)])) <= max_distinct


def eligible(df: pd.DataFrame, targets: list[str], R: int, m: int, alias: str = "D", H: int | None = None,
             s: Settings | None = None) -> tuple[pd.DataFrame, dict]:
    """Series kept for one horizon file. NaN in the evaluation span (last R points) is allowed (fev masks it): NaN in
    structurally closed slots, plus up to s.eval_nan_max of the span otherwise, and every window of H points keeps
    >= s.window_min_finite finite values. Near-constant contexts are dropped (MASE undefined or meaningless)."""
    s = s or Settings()
    H = H or R
    keep, n = [], {"nan_in_eval": 0, "zero_scale": 0, "masked_eval_nan": 0, "closed_eval_nan": 0}
    for iid, g in df.groupby("item_id", sort=False):
        y = g[targets].to_numpy(np.float64)
        ctx_n = len(y) - R
        fin = np.isfinite(y)
        sl = slots(g["timestamp"], alias)
        ok, masked, closed = True, 0, 0
        for k in range(y.shape[1]):
            shut = closed_slot_set(fin[:ctx_n, k], sl[:ctx_n], s.closed_slot_nan) if ctx_n > 0 else set()
            ev_fin = fin[ctx_n:, k]
            ev_closed = np.isin(sl[ctx_n:], list(shut)) if shut else np.zeros(len(ev_fin), bool)
            unexpected = int((~ev_fin & ~ev_closed).sum())
            w = (len(ev_fin) // H) * H
            wins = ev_fin[len(ev_fin) - w:].reshape(-1, H) if w else ev_fin[None, :]
            if unexpected > s.eval_nan_max * len(ev_fin) or (wins.mean(axis=1) < s.window_min_finite).any():
                ok = False
                break
            masked += unexpected
            closed += int((~ev_fin & ev_closed).sum())
        if not ok:
            n["nan_in_eval"] += 1
            continue
        if any(near_constant(y[:ctx_n, k], m, s.near_constant_distinct) for k in range(y.shape[1])):
            n["zero_scale"] += 1
            continue
        n["masked_eval_nan"] += masked
        n["closed_eval_nan"] += closed
        keep.append(iid)
    return df[df["item_id"].isin(keep)], n


# owner 2026-11-14: covariates that are other models' forecasts, or the target itself shifted (a leak), are never used
MODEL_OUTPUT = re.compile(r"chronos|bolt|gbm|lightgbm|xgb|moirai|timesfm|prophet|arima|persistence|(^|[_\s-])seed\d*"
                          r"|(^|[_\s-])pred(s|ict|iction|ictions|icted)?($|[_\s-])|_pred$|(^|_)lag_?\d|rolling_|ewma_"
                          r"|target_next|next_day", re.I)


def model_output_cols(covs: list[str], targets: list[str]) -> list[str]:
    """Covariates named like model outputs (chronos, bolt, gbm, persistence, seed, pred, lag_k, rolling_), and
    forecast / fcst columns that share a name token with a target (a forecast of the target)."""
    from tideforge.devdata.hf_derive import tokens
    tt = {t for c in targets for t in tokens(c) if len(t) > 2}
    out = []
    for c in covs:
        tk = set(tokens(c))
        if MODEL_OUTPUT.search(c) or (tk & {"forecast", "fcst", "forecasted", "predicted"} and tk & tt):
            out.append(c)
    return out


def business_days(data: pd.DataFrame, targets: list[str], alias: str, s: Settings) -> tuple[pd.DataFrame, str, int]:
    """Daily data that is (almost) never observed on Saturday / Sunday (trading days) -> the business-day calendar B:
    weekend rows removed. -> (data, alias, rows removed)."""
    if alias != "D" or data.empty:
        return data, alias, 0
    wk = np.asarray(pd.DatetimeIndex(data["timestamp"]).dayofweek) >= 5
    fin = data[targets].notna().any(axis=1).to_numpy()
    if not wk.any() or wk.all() or fin[wk].mean() > 1 - s.weekend_nan or fin[~wk].mean() < 0.5:
        return data, alias, 0
    return data[~wk], "B", int(wk.sum())


def trim_edges(data: pd.DataFrame, targets: list[str]) -> tuple[pd.DataFrame, int]:
    """Drop each series' leading and trailing rows where every target is NaN (padding, e.g. LOTSA): fev cuts the windows
    from the end of a series, so a NaN tail would put every window on padding. -> (data, rows removed)."""
    if data.empty:
        return data, 0
    data = data.sort_values(["item_id", "timestamp"], kind="stable")
    fin = pd.Series(data[targets].notna().any(axis=1).to_numpy(), index=data.index)
    g = fin.groupby(data["item_id"].to_numpy(), sort=False)
    head = g.cummax()                                         # True from the first observed row on
    tail = fin[::-1].groupby(data["item_id"].to_numpy()[::-1], sort=False).cummax()[::-1]
    keep = (head & tail).to_numpy()
    return data[keep], int((~keep).sum())


def zero_reasons(record: dict) -> str:
    """Why a dataset produced no task, from its subset.json record: counts per drop rule over every variant x horizon."""
    n = {"too_short": 0, "nan_in_eval": 0, "zero_scale": 0}
    skipped = []
    for v in record.get("variants", []):
        n["too_short"] += v.get("dropped_short", 0)
        if v.get("skipped") and not v.get("horizons"):
            skipped.append(f"{v['config']}: {v['skipped']}")
        for cls, hr in (v.get("horizons") or {}).items():
            for k in ("too_short", "nan_in_eval", "zero_scale"):
                n[k] += hr.get(k, 0)
            if hr.get("skipped"):
                skipped.append(f"{v['config']}/{cls} (H={hr['H']}): {hr['skipped']}")
    head = (f"{record.get('input_series', len(record.get('series', [])))} series in, "
            f"{record.get('trimmed_rows', 0)} all-NaN edge rows trimmed, {len(record.get('variants', []))} variants; drops "
            + ", ".join(f"{k} {v}" for k, v in n.items()))
    return head + ("; " + "; ".join(skipped) if skipped else "")


def baseline_filter(tasks: list[dict], out: Path, s: Settings, record: dict) -> list[dict]:
    """Drop tasks whose seasonal-naive MASE or WQL is NaN / infinite, or whose MASE is below s.min_baseline_mase (a
    near-zero in-sample scale); each drop is recorded in record['dropped_tasks']."""
    from tideforge.tfbench.baseline import score
    keep = []
    for t in tasks:
        r = score(Path(out) / f"{t['config']}.parquet", t["task"])
        why = ("seasonal naive " + r["flag"]) if r["flag"] else (
            f"seasonal naive MASE {r['MASE']:.4f} < {s.min_baseline_mase}" if r["MASE"] < s.min_baseline_mase else "")
        if why:
            record["dropped_tasks"].append({"task": t["task"]["task_name"], "why": why})
        else:
            keep.append(t)
    return keep


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
    if isinstance(data["item_id"].dtype, pd.CategoricalDtype):    # canon's fixed categories: no empty groups below
        data = data.assign(item_id=data["item_id"].astype(str))
    n_in = int(data["item_id"].nunique())
    data, trimmed = trim_edges(data, targets)
    leak = model_output_cols(list(covs), targets)
    covs = {c: r for c, r in covs.items() if c not in leak}
    data, native, n_weekend = business_days(data, targets, meta["frequency"], s)
    meta = {**meta, "frequency": native}
    subset = sorted(map(str, data["item_id"].unique()))
    record = {"dataset_id": did, "seed": meta.get("seed", s.seed), "series": subset, "input_series": n_in,
              "trimmed_rows": trimmed, "dropped_covariates_model_output": leak,
              "calendar": "business days (weekend rows removed)" if native == "B" else "as built",
              "weekend_rows_removed": n_weekend, "variants": [], "dropped_tasks": []}
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
            d, drops = eligible(d, targets, nw * H, m, alias, H, s)
            hr.update(drops)
            if d.empty:
                hr["skipped"] = "no series left after the NaN / scale rules"
                continue
            n_ser = int(d["item_id"].nunique()) * len(targets)
            if n_ser * nw * H < s.min_task_points:
                hr["skipped"] = f"task too small: {n_ser} series x {nw} windows x H {H} < {s.min_task_points}"
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
    tasks = baseline_filter(tasks, out, s, record)
    for t in tasks:
        t["task_sha256"] = task_sha(t["task"])
    (out / "tasks.json").write_text(json.dumps(tasks, indent=1, default=str), encoding="utf-8")
    (out / "subset.json").write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    return tasks
