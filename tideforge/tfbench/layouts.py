"""Generic layouts for tables without a usable time column (owner 2026-11-12). Each turns the main table's parts into
parts with a `_time` column (or a long table), so the long / wide loaders in canon run unchanged.

Tried in order, after the main table is chosen, only when the profiler's roles find no time column (or a numeric one):
  string time  a text column whose values read as dates (e.g. "data_fato", "23/08/2020 03:00")
  (b) melt     >= 3 column names are years or dates (wide-by-time tables): one row per (row, date column), ids = text cols
  (a) path     the file path carries the date (partition folders / file names), plus an hour / minute column if any
  (c) numeric  a numeric time column (JD / BJD / MJD days, seconds, steps): snapped to its median cadence; synthetic
               calendar (2000-01-01 + elapsed) unless the values are real Julian dates
  (d) index    no time axis in the data, rows taken in order, frequency from overrides / profile / the dataset name;
               synthetic calendar. Frequency unknown -> Fail("no time axis")
  (e) events   a time column and no numeric value column is canon's existing events layout (daily counts, 0-filled);
               the string-time rule above is what lets event tables like crime records reach it.

adapt(parts, ov, row) -> (parts, ov, notes): ov gains time_col / targets / id_cols / synthetic_time as needed.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pyarrow as pa

from tideforge.tfbench.readers import Part

TIME = "_time"
SYN_START = pd.Timestamp("2000-01-01")
JD_UNIX = 2440587.5

META_PATH = re.compile(r"(^|/|!)(metadata|meta|missing|docs?|documentation)/", re.I)
META_NAME = re.compile(r"(^|[_\-\s.])(readme|license|licence|citation|dictionary|codebook|data_?dictionary|schema|"
                       r"description|descriptions|variables|columns|metadata|changelog|notes?)([_\-\s.]|$)", re.I)
NUM_TIME = re.compile(r"^(b?t?jd|hjd|mjd|bjd_?tdb|btjd|tbjd|jd_?obs|mjd_?obs|time|t|times|time_?s|time_?sec|seconds?|"
                      r"secs?|elapsed(_\w+)?|time_?step|timestep|step|steps|t_?sec|time_?index|tick)$", re.I)
UNNAMED = re.compile(r"^(Unnamed: ?\d+|index|level_0|__index_level_\d+__)$", re.I)
HOUR = re.compile(r"^(hour|hr|hour_of_day|hora|heure|stunde)$", re.I)
MINUTE = re.compile(r"^(minute|min|minute_of_hour)$", re.I)


def is_meta(name: str) -> bool:
    """Paths under metadata/ missing/ docs/, or README-like file names: never a data table."""
    base = name.replace("!", "/").rsplit("/", 1)[-1]
    return bool(META_PATH.search(name) or META_NAME.search(re.sub(r"\.[a-z0-9.]+$", "", base, flags=re.I)))


# ---- helpers ------------------------------------------------------------------------------------------------------------
def _first(p: Part, cols=None, n: int = 2000) -> pd.DataFrame:
    for t in p.batches(cols):
        return t.slice(0, n).to_pandas()
    return pd.DataFrame(columns=cols or p.schema.names)


def _derived(p: Part, extra: list[tuple[str, pa.DataType]], need: list[str], fn) -> Part:
    """Part with extra columns computed by fn(df_of_need_cols, state) -> {name: values}. state persists over one pass."""
    schema = p.schema
    for n, t in extra:
        schema = schema.append(pa.field(n, t))
    names = {n for n, _ in extra}

    def gen(cols):
        state: dict = {}
        base = None if cols is None else list(dict.fromkeys([c for c in cols if c not in names] + need))
        for t in p.batches(base):
            add = fn(t.select(need).to_pandas() if need else pd.DataFrame(index=range(t.num_rows)), state)
            for n, ty in extra:
                t = t.append_column(pa.field(n, ty), pa.array(add[n], type=ty))
            yield t if cols is None else t.select(cols)
    return Part(p.name, schema, gen)


def _ts(v) -> pa.Array:
    return pa.array(pd.to_datetime(pd.Series(v)).astype("datetime64[ns]"), type=pa.timestamp("ns"))


# ---- string time --------------------------------------------------------------------------------------------------------
def string_time(p: Part) -> str | None:
    d = _first(p)
    for f in p.schema:
        if not (pa.types.is_string(f.type) or pa.types.is_large_string(f.type)):
            continue
        v = d[f.name].dropna().astype(str).head(300)
        if len(v) < 5 or not v.str.contains(r"\d[-/.:]\d|\d{4}-\d|[A-Za-z]{3}\s+\d", regex=True).mean() >= 0.9:
            continue
        if v.str.len().median() < 6:
            continue
        ok = pd.to_datetime(v, errors="coerce", format="mixed", dayfirst=bool(v.str.match(r"^\d{1,2}[/.]").mean() > .5))
        if ok.notna().mean() >= 0.9:
            return f.name
    return None


# ---- (b) dates as column names ------------------------------------------------------------------------------------------
def name_date(n: str) -> pd.Timestamp | None:
    s = str(n).strip()
    m = re.fullmatch(r"(?:\w*?[_ ])?((?:1[6-9]|20)\d{2})", s)
    if m:
        return pd.Timestamp(int(m.group(1)), 1, 1)
    if re.fullmatch(r"(1[6-9]|20)\d{2}[-/._](0?[1-9]|1[0-2])([-/._](0?[1-9]|[12]\d|3[01]))?", s) or \
            re.fullmatch(r"\d{1,2}/\d{1,2}/\d{2,4}", s):
        x = pd.to_datetime(s.replace("_", "-").replace(".", "-"), errors="coerce")
        return None if pd.isna(x) else x
    return None


def date_columns(schema: pa.Schema) -> dict:
    return {n: t for n in schema.names if (t := name_date(n)) is not None}


def melt(parts: list[Part], dates: dict, ids: list[str]) -> list[Part]:
    order = sorted(dates, key=lambda c: dates[c])
    times = [dates[c] for c in order]
    schema = pa.schema([(c, pa.string()) for c in ids] + [(TIME, pa.timestamp("ns")), ("value", pa.float64())])
    out = []
    for p in parts:
        def gen(cols, p=p):
            k = 0
            for t in p.batches([c for c in ids + order if c in p.schema.names]):
                d = t.to_pandas()
                key = d[ids].astype("string").fillna("<NA>") if ids else pd.DataFrame(
                    {"_row": [f"row{k + j}" for j in range(len(d))]})
                k += len(d)
                vals = d[order].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
                n, m = vals.shape
                o = {c: np.repeat(key[c].to_numpy(), m) for c in (ids or ["_row"])}
                o[TIME] = np.tile(np.asarray(times, dtype="datetime64[ns]"), n)
                o["value"] = vals.reshape(-1)
                tb = pa.table({c: pa.array(o[c], type=pa.string()) for c in (ids or ["_row"])} |
                              {TIME: pa.array(o[TIME], type=pa.timestamp("ns")), "value": pa.array(o["value"])})
                yield tb if cols is None else tb.select([c for c in cols if c in tb.column_names])
        sch = schema if ids else pa.schema([("_row", pa.string()), (TIME, pa.timestamp("ns")), ("value", pa.float64())])
        out.append(Part(p.name, sch, gen))
    return out


# ---- (a) time from the path ---------------------------------------------------------------------------------------------
def path_time(parts: list[Part]) -> list[Part] | None:
    from tideforge.tfbench.download import partition_date
    dates = [partition_date(p.name.replace("!", "/")) for p in parts]
    if sum(d is not None for d in dates) < 0.9 * len(parts):
        return None
    names = parts[0].schema.names
    hc = next((c for c in names if HOUR.match(c)), None)
    mc = next((c for c in names if MINUTE.match(c)), None)
    if len({d[0] for d in dates if d}) < 2 and not hc:
        return None
    out = []
    for p, d in zip(parts, dates):
        if d is None:
            continue
        need = [c for c in (hc, mc) if c and c in p.schema.names]

        def fn(df, state, base=d[0], hc=hc, mc=mc):
            t = pd.Series(base, index=df.index)
            if hc and hc in df:
                t = t + pd.to_timedelta(pd.to_numeric(df[hc], errors="coerce").fillna(0), unit="h")
            if mc and mc in df:
                t = t + pd.to_timedelta(pd.to_numeric(df[mc], errors="coerce").fillna(0), unit="m")
            return {TIME: t.to_numpy(dtype="datetime64[ns]")}
        out.append(_derived(p, [(TIME, pa.timestamp("ns"))], need, fn))
    return out


# ---- (c) numeric time ---------------------------------------------------------------------------------------------------
def numeric_time_col(schema: pa.Schema, current: str | None) -> str | None:
    num = [f.name for f in schema if pa.types.is_integer(f.type) or pa.types.is_floating(f.type)]
    if current in num and NUM_TIME.match(current):
        return current
    if current:
        return None
    return next((c for c in num if NUM_TIME.match(c)), None)


def numeric_time(parts: list[Part], col: str) -> tuple[list[Part], dict] | None:
    v = pd.to_numeric(_first(parts[0], [col], 20000)[col], errors="coerce").dropna().to_numpy()
    if len(v) < 3:
        return None
    if 1e9 <= np.nanmax(v) < 1e11 or (19000101 <= v.min() and v.max() <= 22001231) or (1000 <= v.min() and v.max() <= 2200
                                                                                        and (v % 1 == 0).all()):
        return None                                         # epoch seconds / yyyymmdd / years: parse_time reads these
    dv = np.diff(v)
    dv = dv[dv > 0]
    if not len(dv):
        return None
    jdname = bool(re.search(r"jd", col, re.I))
    days = jdname or (v.min() >= 1000 and v.max() < 1e5 and np.median(dv) < 1)
    real = days and v.min() >= 2.4e6
    unit = 86400.0 if days else 1.0
    if re.search(r"step|tick|index", col, re.I):
        unit = 1.0
    cad = max(1.0, round(float(np.median(dv)) * unit))
    out = []
    for p in parts:
        lo = min(np.nanmin(pd.to_numeric(t.column(0).to_pandas(), errors="coerce")) for t in p.batches([col]))

        def fn(df, state, lo=lo):
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
            if real:
                sec = (x - JD_UNIX) * 86400.0
                sec = np.round(sec / cad) * cad
                return {TIME: pd.to_datetime(sec, unit="s").to_numpy(dtype="datetime64[ns]")}
            sec = np.round((x - lo) * unit / cad) * cad
            return {TIME: (SYN_START + pd.to_timedelta(sec, unit="s")).to_numpy(dtype="datetime64[ns]")}
        out.append(_derived(p, [(TIME, pa.timestamp("ns"))], [col], fn))
    return out, {"synthetic_time": not real, "cadence_s": cad, "unit": "days" if days else "seconds",
                 "freq": pd.tseries.frequencies.to_offset(pd.Timedelta(seconds=cad)).freqstr}


# ---- (d) index only -----------------------------------------------------------------------------------------------------
def index_freq(ov: dict, row: dict) -> tuple[str | None, str]:
    from tideforge.devdata.hf_derive import freq_from_name
    if ov.get("freq"):
        return ov["freq"], "overrides.yaml"
    from tideforge.tfbench.canon import _alias_from_profile
    a = _alias_from_profile(row)
    if a:
        return a, "profile"
    for src in (row.get("dataset_id"), row.get("config_or_subset"), row.get("repo_id")):
        for tok in re.split(r"[_\-/\s]", str(src or "")):
            fs = freq_from_name(tok) if tok else None
            if fs:
                return _alias_from_profile({"freq_seconds": fs}), f"dataset name word '{tok}'"
    return None, ""


def index_time(parts: list[Part], ids: list[str], alias: str) -> list[Part]:
    step = pd.tseries.frequencies.to_offset(alias)
    out = []
    for p in parts:
        need = [c for c in ids if c in p.schema.names]

        def fn(df, state):
            if need:
                key = df[need].astype("string").fillna("<NA>").agg("|".join, axis=1) if len(need) > 1 else \
                    df[need[0]].astype("string").fillna("<NA>")
                k = key.groupby(key, sort=False).cumcount().to_numpy() + key.map(lambda x: state.get(x, 0)).to_numpy()
                for x, c in key.value_counts().items():
                    state[x] = state.get(x, 0) + int(c)
            else:
                k = np.arange(len(df)) + state.get("_", 0)
                state["_"] = state.get("_", 0) + len(df)
            return {TIME: pd.date_range(SYN_START, periods=int(k.max()) + 1 if len(k) else 0, freq=step)[k]
                    .to_numpy(dtype="datetime64[ns]")}
        out.append(_derived(p, [(TIME, pa.timestamp("ns"))], need, fn))
    return out


# ---- driver -------------------------------------------------------------------------------------------------------------
def adapt(parts: list[Part], ov: dict, row: dict, roles_for) -> tuple[list[Part], dict, list[dict]]:
    """roles_for: canon.roles_for (passed in to avoid an import cycle)."""
    from tideforge.tfbench.canon import Fail
    notes: list[dict] = []
    if ov.get("time_col") or ov.get("layout") == "nested":
        if ov.get("time_col") and (c := numeric_time_col(parts[0].schema, ov["time_col"])):
            res = numeric_time(parts, c)
            if res:
                parts, info = res
                notes.append({"step": "numeric time column", "column": c, **info})
                return parts, {**ov, "time_col": TIME, "synthetic_time": info["synthetic_time"] or ov.get("synthetic_time"),
                               "freq": ov.get("freq") or info["freq"]}, notes
        return parts, ov, notes
    try:
        r, layout, _ = roles_for(parts[0].schema, ov)
    except Fail:
        r, layout = {"time_col": None, "id_cols": [], "target_cols": []}, "unknown"
    if layout == "nested":
        return parts, ov, notes
    if r["time_col"] and str(r.get("time_rule", "")).startswith("calendar column name"):
        pt = path_time(parts)          # an hour / day column alone is not a time axis when the path carries the date
        if pt:
            notes.append({"step": "time from the file path", "parts": len(pt), "dropped_undated": len(parts) - len(pt),
                          "calendar_column": r["time_col"]})
            return pt, {**ov, "time_col": TIME}, notes
    if r["time_col"]:
        c = numeric_time_col(parts[0].schema, r["time_col"])
        if c:
            res = numeric_time(parts, c)
            if res:
                parts, info = res
                notes.append({"step": "numeric time column", "column": c, **info})
                return parts, {**ov, "time_col": TIME, "synthetic_time": info["synthetic_time"],
                               "freq": ov.get("freq") or info["freq"]}, notes
        return parts, ov, notes
    st = string_time(parts[0])
    if st:
        notes.append({"step": "time column from values", "column": st, "rule": ">= 90% of the first rows read as dates"})
        return parts, {**ov, "time_col": st}, notes
    dates = date_columns(parts[0].schema)
    if len(dates) >= 3:
        ids = [f.name for f in parts[0].schema if f.name not in dates and (pa.types.is_string(f.type) or
               pa.types.is_large_string(f.type)) and not UNNAMED.match(f.name)][:4]
        notes.append({"step": "melt: dates as column names", "date_columns": len(dates), "id_cols": ids,
                      "first": str(min(dates.values()).date()), "last": str(max(dates.values()).date())})
        return melt(parts, dates, ids), {**ov, "time_col": TIME, "targets": ["value"], "covariates": [], "static": [],
                                         "id_cols": ids or ["_row"], "layout": "long"}, notes
    pt = path_time(parts)
    if pt:
        notes.append({"step": "time from the file path", "parts": len(pt), "dropped_undated": len(parts) - len(pt)})
        return pt, {**ov, "time_col": TIME}, notes
    c = numeric_time_col(parts[0].schema, None)
    if c:
        res = numeric_time(parts, c)
        if res:
            parts2, info = res
            notes.append({"step": "numeric time column", "column": c, **info})
            return parts2, {**ov, "time_col": TIME, "synthetic_time": info["synthetic_time"],
                            "freq": ov.get("freq") or info["freq"]}, notes
    if not r.get("target_cols"):
        raise Fail("no time axis (no time column, no dates in names or paths, no numeric value column)")
    alias, how = index_freq(ov, row)
    if not alias:
        raise Fail("no time axis (no time column and the frequency is unknown: set freq in overrides.yaml)")
    ids = list(ov.get("id_cols") or r["id_cols"]) or [f.name for f in parts[0].schema if (
        pa.types.is_string(f.type) or pa.types.is_large_string(f.type)) and not UNNAMED.match(f.name)][:3]
    notes.append({"step": "synthetic time from row order", "freq": alias, "freq_source": how, "id_cols": ids,
                  "rule": f"rows in file order per series, from {SYN_START.date()}"})
    extra = {"id_cols": ids, "targets": r["target_cols"], "layout": "long"} if ids else {}
    return index_time(parts, ids, alias), {**ov, "time_col": TIME, "synthetic_time": True, "freq": alias,
                                           **extra}, notes
