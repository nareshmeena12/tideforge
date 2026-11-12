"""Timestamps, frequency detection and regularisation (A3). Pure pandas, no I/O.

parse_time(series)       any time column -> datetime64[ns] (UTC, tz dropped); NaT where it cannot be read.
infer_freq(ts_by_item)   mode of the timestamp differences -> (alias, share of differences at that step).
regularise(df, alias)    snap to the grid, average duplicates, reindex: real gaps become NaN rows, never filled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86400


def parse_time(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s
    elif pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        mx = v.abs().max()
        if mx == mx and 1000 <= v.min() and mx <= 2200 and (v.dropna() % 1 == 0).all():   # calendar years
            out = pd.to_datetime(v.astype("Int64").astype(str) + "-01-01", errors="coerce")
        elif mx == mx and 19000101 <= v.min() and mx <= 22001231 and (v.dropna() % 1 == 0).all():  # yyyymmdd
            out = pd.to_datetime(v.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        else:
            unit = "s" if mx < 1e11 else "ms" if mx < 1e14 else "us" if mx < 1e17 else "ns"
            out = pd.to_datetime(v, unit=unit, errors="coerce")
    else:
        s = s.astype("string")
        out = pd.to_datetime(s, errors="coerce", utc=True, format="mixed")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert("UTC").dt.tz_localize(None)
    return out.astype("datetime64[ns]")


def _calendar_alias(kind: str, ts: pd.Series) -> str:
    if kind == "W":
        return "W-" + pd.Series(ts.dt.day_name().str[:3].str.upper()).mode().iloc[0]
    if (ts.dt.day == 1).mean() >= 0.5:
        return {"M": "MS", "Q": "QS", "Y": "YS"}[kind]
    if ts.dt.is_month_end.mean() >= 0.5:
        return {"M": "ME", "Q": "QE", "Y": "YE"}[kind]
    return {"M": "MS", "Q": "QS", "Y": "YS"}[kind]   # mid-period stamps: snapped to the period start


def infer_freq(groups) -> tuple[str | None, float, float]:
    """groups: iterable of timestamp Series (one per item). -> (alias, share at the mode, mode in seconds)."""
    diffs, stamps = [], []
    for ts in groups:
        t = np.sort(pd.Series(ts).dropna().unique())
        if len(t) > 1:
            diffs.append(np.diff(t).astype("timedelta64[s]").astype(np.int64))
            stamps.append(pd.Series(t[-500:]))
    if not diffs:
        return None, 0.0, 0.0
    d = np.concatenate(diffs)
    d = d[d > 0]
    if not len(d):
        return None, 0.0, 0.0
    days = d / DAY
    ts = pd.concat(stamps, ignore_index=True)
    for kind, lo, hi in (("M", 28, 31), ("Q", 89, 92), ("Y", 365, 366)):
        share = ((days >= lo) & (days <= hi)).mean()
        if share >= 0.5:
            return _calendar_alias(kind, ts), float(share), float(np.median(d[(days >= lo) & (days <= hi)]))
    vals, cnt = np.unique(d, return_counts=True)
    mode = int(vals[cnt.argmax()])
    share = float(cnt.max() / len(d))
    if mode % (7 * DAY) == 0 and mode // (7 * DAY) == 1:
        return _calendar_alias("W", ts), share, float(mode)
    alias = pd.tseries.frequencies.to_offset(pd.Timedelta(seconds=mode)).freqstr
    return alias, share, float(mode)


def snap(ts: pd.Series, alias: str) -> pd.Series:
    a = alias.upper()
    if a.startswith("W"):
        return ts.dt.to_period(alias).dt.end_time.dt.normalize()
    if a in ("MS", "QS", "YS"):
        return ts.dt.to_period(a[0]).dt.start_time
    if a in ("ME", "QE", "YE"):
        return ts.dt.to_period(a[0]).dt.end_time.dt.normalize()
    return ts.dt.floor(alias)


def regularise(df: pd.DataFrame, alias: str, value_cols: list[str], agg: dict | None = None, fill_zero: bool = False,
               max_grid: int = 5_000_000, keep_last: int | None = None) -> tuple[pd.DataFrame, dict]:
    """One item: df(timestamp, value cols...) -> regular grid. Duplicate stamps after snapping are averaged (or summed
    where agg[col] == "sum"). Missing grid steps are NaN rows (0 when fill_zero, used only for event counts)."""
    agg = agg or {}
    t = df["timestamp"]
    st = snap(t, alias)
    stats = {"snapped": int((st != t).sum()), "duplicates": 0, "grid_added": 0}
    d = df.assign(timestamp=st).dropna(subset=["timestamp"])
    if d["timestamp"].duplicated().any():
        stats["duplicates"] = int(d["timestamp"].duplicated().sum())
        how = {c: ("sum" if agg.get(c) == "sum" else "mean" if pd.api.types.is_numeric_dtype(d[c]) else "first")
               for c in d.columns if c != "timestamp"}
        d = d.groupby("timestamp", sort=True).agg(how).reset_index()
    else:
        d = d.sort_values("timestamp")
    if d.empty:
        return d, stats
    end = d["timestamp"].iloc[-1]
    start = d["timestamp"].iloc[0]
    n = keep_last or max_grid
    grid = pd.date_range(end=end, periods=min(n, max_grid), freq=alias) if keep_last else None
    full = pd.date_range(start, end, freq=alias) if grid is None else grid[grid >= start]
    if len(full) > max_grid:
        full = full[-max_grid:]
    out = d.set_index("timestamp").reindex(full)
    stats["grid_added"] = int(len(full) - out[value_cols].notna().any(axis=1).sum())
    if fill_zero:
        out[value_cols] = out[value_cols].fillna(0.0)
    out.index.name = "timestamp"
    return out.reset_index(), stats
