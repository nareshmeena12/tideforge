"""Timestamps, frequency detection and regularisation (A3). Pure pandas, no I/O.

parse_time(series)       any time column -> datetime64[ns] (UTC, tz dropped); NaT where it cannot be read.
infer_freq(ts_by_item)   mode of the timestamp differences -> (alias, share of differences at that step).
regularise(df, alias)    snap to the grid, average duplicates, reindex: real gaps become NaN rows, never filled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86400


# Tried in order on a sample of the distinct values; month-first before day-first (what format="mixed" did before).
FORMATS = ["ISO8601", "%m/%d/%Y", "%d/%m/%Y", "%m/%d/%Y %H:%M", "%d/%m/%Y %H:%M", "%m/%d/%Y %H:%M:%S",
           "%d/%m/%Y %H:%M:%S", "%m/%d/%y", "%d/%m/%y", "%d-%m-%Y", "%d.%m.%Y", "%d.%m.%Y %H:%M", "%d-%m-%Y %H:%M",
           "%Y/%m/%d", "%Y/%m/%d %H:%M", "%Y/%m/%d %H:%M:%S", "%Y.%m.%d", "%Y%m%d", "%d-%b-%Y", "%d %b %Y", "%b %d, %Y",
           "%B %d, %Y", "%d %B %Y", "%b %Y", "%B %Y", "%m/%Y", "%Y%m%d%H%M%S", "%Y%m%d %H:%M"]


_FMT: dict[str, str] = {}   # column key -> format that fitted its first batch (tried first on the next batches)


def guess_format(u: pd.Series, n: int = 500, key: str | None = None) -> str | None:
    """The first format that reads >= 99% of a sample of distinct strings (None: no single format fits). With a key
    (the column being read batch by batch), the format found for its first batch is tried first: one parse per batch,
    and the same reading of ambiguous d/m vs m/d dates in every batch."""
    sm = u.iloc[:: max(1, len(u) // n)][:n] if len(u) > n else u
    for f in ([_FMT[key]] if key in _FMT else []) + FORMATS:
        try:
            ok = pd.to_datetime(sm, format=f, errors="coerce", utc=True).notna().mean()
        except (ValueError, TypeError):
            continue
        if ok >= 0.99:
            if key is not None:
                _FMT[key] = f
            return f
    return None


def parse_strings(s: pd.Series, key: str | None = None) -> pd.Series:
    """Strings -> UTC timestamps. Parses each distinct value once, with one explicit format guessed from a sample;
    only when no format fits are the distinct values read with format="mixed" (owner 2026-11-13: per-element dateutil
    on full columns stalled canon for hours)."""
    s = s.astype("string").str.strip()
    codes, uniq = pd.factorize(s)
    u = pd.Series(uniq, dtype="string")
    if len(u) == 0:
        return pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns, UTC]")
    f = guess_format(u, key=key)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        if f is not None:
            p = pd.to_datetime(u, format=f, errors="coerce", utc=True)
            bad = p.isna() & u.notna()
            if bad.any():           # a few stragglers in another format: those only
                p[bad] = pd.to_datetime(u[bad], format="mixed", errors="coerce", utc=True)
        else:
            p = pd.to_datetime(u, format="mixed", errors="coerce", utc=True)
    vals = pd.DatetimeIndex(p).tz_convert(None).to_numpy(dtype="datetime64[ns]")   # naive UTC
    out = np.where(codes >= 0, vals[np.clip(codes, 0, None)], np.datetime64("NaT", "ns"))
    return pd.Series(out, index=s.index, dtype="datetime64[ns]")


def years_to_time(v: pd.Series) -> pd.Series:
    """Integer calendar years -> Jan 1, built directly (no string parsing); years outside 1678..2261 -> NaT."""
    y = v.to_numpy(dtype="float64")
    ok = np.isfinite(y) & (y >= 1678) & (y <= 2261)
    out = np.full(len(y), np.datetime64("NaT"), dtype="datetime64[ns]")
    out[ok] = (y[ok].astype("int64") - 1970).astype("datetime64[Y]").astype("datetime64[ns]")
    return pd.Series(out, index=v.index)


def parse_time(s: pd.Series, key: str | None = None) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        out = s
        if getattr(out.dt, "tz", None) is not None:
            out = out.dt.tz_convert("UTC").dt.tz_localize(None)
        if out.dtype != "datetime64[ns]":       # ms / us / s columns can hold dates outside the ns range (1615 quakes)
            v = out.to_numpy()
            ok = ~np.isnat(v) & (v >= np.datetime64("1678-01-02")) & (v <= np.datetime64("2262-04-10"))
            a = np.full(len(v), np.datetime64("NaT", "ns"))
            a[ok] = v[ok].astype("datetime64[ns]")
            out = pd.Series(a, index=s.index)    # rows out of range become NaT: dropped and counted by canon
    elif pd.api.types.is_numeric_dtype(s):
        v = pd.to_numeric(s, errors="coerce")
        mx = v.abs().max()
        if mx == mx and 1678 <= v.min() and mx <= 2200 and (v.dropna() % 1 == 0).all():   # calendar years
            out = years_to_time(v)
        elif mx == mx and 19000101 <= v.min() and mx <= 22001231 and (v.dropna() % 1 == 0).all():  # yyyymmdd
            out = pd.to_datetime(v.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")
        else:
            unit = "s" if mx < 1e11 else "ms" if mx < 1e14 else "us" if mx < 1e17 else "ns"
            out = pd.to_datetime(v, unit=unit, errors="coerce")
    else:
        out = parse_strings(s, key)
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
    from tideforge.tfbench.rules import freq_seconds
    n = min(keep_last or max_grid, max_grid)
    span = int((end - start).total_seconds() // max(freq_seconds(alias), 1.0)) + 2   # steps from start to end (approx.)
    full = pd.date_range(end=end, periods=min(n, span), freq=alias)                  # never reaches before start - 1 step
    full = full[full >= start]
    out = d.set_index("timestamp").reindex(full)
    stats["grid_added"] = int(len(full) - out[value_cols].notna().any(axis=1).sum())
    if fill_zero:
        out[value_cols] = out[value_cols].fillna(0.0)
    out.index.name = "timestamp"
    return out.reset_index(), stats
