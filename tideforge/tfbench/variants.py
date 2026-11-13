"""B1 frequency-aggregation variants and the fev table writer.

Variants of one dataset: the native frequency, then hourly (from sub-hourly), daily (from sub-daily) and weekly (W-SUN,
from daily or finer), each kept only for the series still at least rules.min_length long after aggregation. Every variant
is aggregated from the native data (not chained).

Aggregation per column (agg_rules): overrides.yaml `agg` first; event counts / sums -> sum; categorical or flag
covariates (bool-like, or integer-valued with <= flag_max_distinct values, or calendar "known" columns) -> last; else
mean. A bucket with fewer than agg_coverage of its expected points (non-NaN) becomes NaN (never filled).

to_fev(df, ...) -> pa.Table in fev's native form: one row per series, id, timestamp list, float32 lists, static scalars.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from tideforge.tfbench import rules
from tideforge.tfbench.config import Settings

LADDER = (("subhourly", "h"), ("subdaily", "D"), ("daily", "W-SUN"))


def agg_rules(data: pd.DataFrame, meta: dict, ov: dict, s: Settings) -> dict[str, str]:
    cols = list(meta["targets"]) + list(meta.get("covariates", {}))
    out = {}
    for c in cols:
        if c in (ov.get("agg") or {}):
            out[c] = ov["agg"][c]
        elif c == "count" or c.startswith("sum_"):
            out[c] = "sum"
        elif c in meta.get("covariates", {}) and is_categorical(data[c], meta["covariates"][c], s):
            out[c] = "last"
        else:
            out[c] = "mean"
    return out


def is_categorical(x: pd.Series, role: str, s: Settings) -> bool:
    v = pd.to_numeric(x, errors="coerce").dropna()
    if v.empty:
        return False
    if role == "known":
        return True
    vals = np.unique(v.to_numpy()[:200_000])
    return bool(np.all(vals == np.round(vals)) and len(vals) <= s.flag_max_distinct)


def targets_for(alias: str) -> list[str]:
    """Coarser frequencies a native frequency is aggregated to."""
    sec = rules.freq_seconds(alias)
    out = []
    if sec < 3600:
        out.append("h")
    if sec < 86400:
        out.append("D")
    if sec <= 86400:
        out.append("W-SUN")
    return out


def aggregate(df: pd.DataFrame, src: str, dst: str, how: dict[str, str], coverage: float) -> pd.DataFrame:
    """One series (timestamp + value columns) on the src grid -> dst buckets; NaN where coverage is too low."""
    ratio = rules.freq_seconds(dst) / rules.freq_seconds(src)
    if src.upper().endswith("B") and dst.startswith("W"):     # business days: 5 per week, not 7
        ratio *= 5 / 7
    g = df.set_index("timestamp").resample(dst, label="left" if not dst.startswith("W") else "right",
                                           closed="left" if not dst.startswith("W") else "right")
    cnt = g.count()
    out = {}
    for c, h in how.items():
        if c not in df.columns:
            continue
        if h == "sum":
            v = g[c].sum(min_count=1)
        elif h == "last":
            v = g[c].last()
        else:
            v = g[c].mean()
        v[cnt[c] < coverage * ratio] = np.nan
        out[c] = v
    res = pd.DataFrame(out)
    res.index.name = "timestamp"
    return res.reset_index()


def freq_variants(data: pd.DataFrame, meta: dict, ov: dict, s: Settings) -> list[dict]:
    """[{"freq", "data", "agg": rules or None, "dropped_short": n}] starting with the native frequency."""
    native = meta["frequency"]
    out = [{"freq": native, "data": data, "agg": None, "dropped_short": 0}]
    how = agg_rules(data, meta, ov, s)
    static = [c for c in meta.get("static", []) if c in data.columns]
    for dst in targets_for(native):
        need = rules.min_length(dst, s)
        parts, short = [], 0
        for iid, g in data.groupby("item_id", sort=False):
            a = aggregate(g.drop(columns=["item_id"] + static), native, dst, how, s.agg_coverage)
            fin = a[list(meta["targets"])].notna().any(axis=1).to_numpy()
            if fin.any():
                a = a.iloc[int(np.argmax(fin)): len(fin) - int(np.argmax(fin[::-1]))]
            if len(a) < need:
                short += 1
                continue
            a.insert(0, "item_id", iid)
            for c in static:
                a[c] = g[c].iloc[0]
            parts.append(a)
        if parts:
            out.append({"freq": dst, "data": pd.concat(parts, ignore_index=True), "agg": how, "dropped_short": short})
    return out


def to_fev(df: pd.DataFrame, value_cols: list[str], static: list[str]) -> pa.Table:
    ids, ts, vals, st = [], [], {c: [] for c in value_cols}, {c: [] for c in static}
    for iid, g in df.groupby("item_id", sort=True):
        ids.append(str(iid))
        ts.append(g["timestamp"].to_numpy("datetime64[ms]"))
        for c in value_cols:
            vals[c].append(g[c].to_numpy(np.float32))
        for c in static:
            st[c].append(None if pd.isna(g[c].iloc[0]) else str(g[c].iloc[0]))
    cols = {"id": pa.array(ids, pa.string()),
            "timestamp": pa.array(ts, pa.list_(pa.timestamp("ms")))}
    for c in value_cols:
        cols[c] = pa.array(vals[c], pa.list_(pa.float32()))
    for c in static:
        cols[c] = pa.array(st[c], pa.string())
    return pa.table(cols)
