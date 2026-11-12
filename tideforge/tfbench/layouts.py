"""Generic layouts for tables without a usable time column (owner 2026-11-12). Each turns the main table's parts into
parts with a `_time` column (or a long table), so the long / wide loaders in canon run unchanged.

Tried in order, after the main table is chosen, only when the profiler's roles find no time column (or a numeric one):
  calendar     overrides.yaml `calendar`, or found by name + values (owner 2026-11-13): year + month / week / quarter
               (+ day, hour), a yyyymm integer, a year-week text (2014/01) or a financial year (1997-98). Weeks: MMWR
               (CDC / Brazil epi weeks, labelled by the Saturday that ends them) or ISO (Monday). Never replaces a real
               timestamp / text / epoch time column.
  hour + path  a numeric 'time'-like column holding 0..23 with the date in the file path (gladcolor)
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

import functools
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
@functools.lru_cache(maxsize=1 << 16)
def name_date(n: str) -> pd.Timestamp | None:
    s = str(n).strip()
    if re.fullmatch(r"(1[6-9]|20)\d{2}[-/.](0?[1-9]|1[0-2])[-/.](0?[1-9]|[12]\d|3[01])[ T]\d{1,2}:\d{2}(:\d{2})?", s):
        x = pd.to_datetime(s.replace("/", "-").replace(".", "-", 2), errors="coerce")   # hourly columns (gladcolor)
        return None if pd.isna(x) else x
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


ITEM = "_item"


def path_item(name: str, pattern: str) -> str:
    m = re.search(pattern, name.replace("!", "/"))
    return (m.group(1) if m.lastindex else m.group(0)) if m else name          # the first group is the id


def melt(parts: list[Part], ids: list[str], item_re: str | None = None, sum_rows: bool = False) -> list[Part]:
    """Dates as column names -> long rows. Each file uses its own date columns (months differ). item_re: the series id
    comes from the file path (regex, first group); sum_rows: rows of a file are summed per date column (e.g. block groups
    -> county), so each file gives one row per time step. Streams batch by batch."""
    idc = ([ITEM] if item_re and sum_rows else ([ITEM] if item_re else []) + (ids or ([] if item_re else ["_row"])))
    sch = pa.schema([(c, pa.string()) for c in idc] + [(TIME, pa.timestamp("ns")), ("value", pa.float64())])
    out = []
    for p in parts:
        dts = date_columns(p.schema)
        order = sorted(dts, key=lambda c: dts[c])
        times = np.asarray([dts[c] for c in order], dtype="datetime64[ns]")
        item = path_item(p.name, item_re) if item_re else None

        def gen(cols, p=p, order=order, times=times, item=item):
            k = 0
            for t in p.batches([c for c in ids + order if c in p.schema.names]):
                d = t.to_pandas()
                vals = d[order].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
                n, m = vals.shape
                if sum_rows:
                    s = np.nansum(vals, axis=0)
                    s[np.isnan(vals).all(axis=0)] = np.nan
                    o = {ITEM: np.repeat(item if item is not None else p.name, m), TIME: times, "value": s}
                else:
                    key = d[ids].astype("string").fillna("<NA>") if ids else pd.DataFrame(
                        {"_row": [f"row{k + j}" for j in range(n)]})
                    o = {c: np.repeat(key[c].to_numpy(), m) for c in key.columns if c in idc}
                    if item_re:
                        o[ITEM] = np.repeat(item, n * m)
                    o[TIME] = np.tile(times, n)
                    o["value"] = vals.reshape(-1)
                k += n
                tb = pa.table({c: pa.array(o[c], type=pa.string()) for c in idc} |
                              {TIME: pa.array(o[TIME], type=pa.timestamp("ns")), "value": pa.array(o["value"])})
                yield tb if cols is None else tb.select([c for c in cols if c in tb.column_names])
        out.append(Part(p.name, sch, gen, item=item if sum_rows else None))
    return out


# ---- (a) time from the path ---------------------------------------------------------------------------------------------
def path_time(parts: list[Part], hour_col: str | None = None) -> list[Part] | None:
    from tideforge.tfbench.download import partition_date
    dates = [partition_date(p.name.replace("!", "/")) for p in parts]
    if sum(d is not None for d in dates) < 0.9 * len(parts):
        return None
    names = parts[0].schema.names
    hc = hour_col or next((c for c in names if HOUR.match(c)), None)
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


# ---- calendar columns (owner 2026-11-13: real time instead of synthetic) -------------------------------------------------
YEAR_W = r"year|yr|ano|anno|año|jahr|annee|année"
CAL_WORDS = {"year": YEAR_W, "month": r"month|mes|mois|monat|mese", "week": r"week|wk|semana|semaine|woche|settimana",
             "quarter": r"quarter|qtr|trimestre|quartal", "day": r"day|dia|día|jour|tag|giorno", "hour": r"hour|hr|hora|heure"}
CAL_RANGE = {"year": (1800, 2200), "month": (1, 12), "week": (1, 53), "quarter": (1, 4), "day": (1, 31), "hour": (0, 23)}
YEARWEEK = re.compile(r"^\s*((?:18|19|20|21)\d{2})\s*[-/_]?\s*W?(\d{1,2})\s*$", re.I)
FISCAL = re.compile(r"^\s*((?:18|19|20|21)\d{2})\s*[-/]\s*(\d{2})\s*$")


def _word(kind: str, name: str) -> bool:
    return bool(re.search(rf"(^|[^a-z])({CAL_WORDS[kind]})([^a-z]|$)", str(name), re.I))


def _ints_in(v: pd.Series, lo: int, hi: int) -> bool:
    x = pd.to_numeric(v, errors="coerce").dropna()
    return len(x) >= 3 and bool(((x % 1 == 0) & (x >= lo) & (x <= hi)).all())


def mmwr_week_end(year, week) -> pd.Series:
    """Saturday that ends MMWR week `week` of `year` (CDC: Sunday-Saturday weeks; week 1 is the first week with at least
    four days in the year). Brazil's epidemiological weeks (semana epidemiológica) follow the same rule."""
    y = pd.to_numeric(pd.Series(year), errors="coerce")
    w = pd.to_numeric(pd.Series(week), errors="coerce")
    jan1 = pd.to_datetime(y.astype("Int64").astype("string") + "-01-01", errors="coerce")
    dow = (jan1.dt.dayofweek + 1) % 7                                  # Sunday = 0
    start = jan1 - pd.to_timedelta(dow, unit="D") + pd.to_timedelta(np.where(dow > 3, 7, 0), unit="D")
    return start + pd.to_timedelta((w - 1) * 7 + 6, unit="D")


def iso_week_start(year, week) -> pd.Series:
    y = pd.to_numeric(pd.Series(year), errors="coerce").astype("Int64").astype("string")
    w = pd.to_numeric(pd.Series(week), errors="coerce").astype("Int64").astype("string").str.zfill(2)
    return pd.to_datetime(y + "-W" + w + "-1", format="%G-W%V-%u", errors="coerce")


def find_calendar(p: Part) -> dict | None:
    """Calendar columns that together give a time axis: {year + month / week / quarter (+ day, hour)}, a yyyymm integer, a
    year-week string (2014/01, 2014-W01) or a financial year string (1997-98). None when absent or ambiguous."""
    d = _first(p)
    num = [f.name for f in p.schema if pa.types.is_integer(f.type) or pa.types.is_floating(f.type)]
    txt = [f.name for f in p.schema if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)]
    found = {k: [c for c in num if _word(k, c) and _ints_in(d[c], *CAL_RANGE[k])] for k in CAL_WORDS}
    if len(found["year"]) == 1:
        for k in ("month", "week", "quarter"):
            if len(found[k]) == 1:
                cal = {"year": found["year"][0], k: found[k][0]}
                if k == "month" and len(found["day"]) == 1:
                    cal["day"] = found["day"][0]
                    if len(found["hour"]) == 1:
                        cal["hour"] = found["hour"][0]
                return cal
    ym = [c for c in num if (_word("month", c) or re.search(r"yyyymm|(^|_)ym($|_)|period", c, re.I))
          and _ints_in(d[c], 180001, 210012) and _ints_in(pd.to_numeric(d[c], errors="coerce") % 100, 1, 12)]
    if len(ym) == 1:
        return {"yyyymm": ym[0]}
    for c in txt:
        v = d[c].dropna().astype(str).head(300)
        if len(v) < 3:
            continue
        if _word("week", c) and v.str.match(YEARWEEK).all():
            return {"yearweek": c}
        m = v.str.extract(FISCAL)
        if m.notna().all().all() and ((m[0].astype(int) + 1) % 100 == m[1].astype(int)).all():
            return {"fiscal_year": c}
    return None


def week_rule(cal: dict) -> str:
    if cal.get("week_rule"):
        return cal["week_rule"]
    names = " ".join(str(v) for k, v in cal.items() if k in ("week", "yearweek"))
    return "mmwr" if re.search(r"mmwr|epi|semana", names, re.I) else "iso"


def calendar_time(parts: list[Part], cal: dict) -> tuple[list[Part], str, list[str]]:
    """-> (parts with _time, pandas alias, the calendar columns used). Rows whose calendar cells do not read -> NaT."""
    used = [cal[k] for k in ("year", "month", "week", "quarter", "day", "hour", "yyyymm", "yearweek", "fiscal_year")
            if cal.get(k)]
    rule = week_rule(cal)
    if cal.get("hour"):
        alias = "h"
    elif cal.get("day"):
        alias = "D"
    elif cal.get("month") or cal.get("yyyymm"):
        alias = "MS"
    elif cal.get("quarter"):
        alias = "QS"
    elif cal.get("week") or cal.get("yearweek"):
        alias = "W-SAT" if rule == "mmwr" else "W-MON"
    else:
        alias = "YS"

    def fn(df, state):
        num = {c: pd.to_numeric(df[c], errors="coerce") for c in used if c not in (cal.get("yearweek"), cal.get("fiscal_year"))}
        if cal.get("yyyymm"):
            v = num[cal["yyyymm"]]
            y, m = v // 100, v % 100
            t = pd.to_datetime(dict(year=y, month=m.where((m >= 1) & (m <= 12)), day=1), errors="coerce")
        elif cal.get("fiscal_year"):
            ex = df[cal["fiscal_year"]].astype("string").str.extract(FISCAL)
            t = pd.to_datetime(ex[0] + "-01-01", errors="coerce")
        elif cal.get("yearweek"):
            ex = df[cal["yearweek"]].astype("string").str.extract(YEARWEEK)
            t = mmwr_week_end(ex[0], ex[1]) if rule == "mmwr" else iso_week_start(ex[0], ex[1])
        else:
            y = num[cal["year"]]
            if cal.get("week"):
                t = mmwr_week_end(y, num[cal["week"]]) if rule == "mmwr" else iso_week_start(y, num[cal["week"]])
            elif cal.get("quarter"):
                t = pd.to_datetime(dict(year=y, month=(num[cal["quarter"]] - 1) * 3 + 1, day=1), errors="coerce")
            else:
                t = pd.to_datetime(dict(year=y, month=num[cal["month"]] if cal.get("month") else 1,
                                        day=num[cal["day"]] if cal.get("day") else 1), errors="coerce")
                if cal.get("hour"):
                    t = t + pd.to_timedelta(num[cal["hour"]], unit="h")
        return {TIME: pd.Series(t).reset_index(drop=True).to_numpy(dtype="datetime64[ns]")}
    return [_derived(p, [(TIME, pa.timestamp("ns"))], used, fn) for p in parts], alias, used


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
def _is_time_typed(schema: pa.Schema, col: str | None) -> bool:
    """A timestamp / date column, or text (parse_time reads it): a real time column the calendar rule must not replace."""
    if not col or col not in schema.names:
        return False
    t = schema.field(col).type
    return pa.types.is_timestamp(t) or pa.types.is_date(t) or pa.types.is_string(t) or pa.types.is_large_string(t)


def _hour_path(parts, c, ov, notes):
    """A numeric 'time' column holding 0..23 is an hour of day when the file path carries the date (gladcolor)."""
    if not _ints_in(_first(parts[0], [c])[c], 0, 23):
        return None
    pt = path_time(parts, hour_col=c)
    if not pt:
        return None
    notes.append({"step": "time from the file path", "parts": len(pt), "dropped_undated": len(parts) - len(pt),
                  "hour_column": c})
    return pt, {**ov, "time_col": TIME, "freq": ov.get("freq") or "h"}, notes


def _with_calendar(parts, ov, cal, roles_for, notes, how):
    parts2, alias, used = calendar_time(parts, cal)
    ov2 = {**ov, "time_col": TIME}
    r, _, _ = roles_for(parts2[0].schema, ov2)
    tg = ov.get("targets") or [c for c in r["target_cols"] if c not in used]
    if not tg:     # the profiler took the calendar column as the target: the other value columns are the targets
        tg = [c for c in dict.fromkeys(r.get("value_cols", []) + r["covariate_cols"])
              if c not in used and c not in r["id_cols"] and c != TIME and not UNNAMED.match(c)]
    cov = ov["covariates"] if "covariates" in ov else [c for c in r["covariate_cols"] if c not in used and c not in tg]
    notes.append({"step": "time from calendar columns", "columns": {k: v for k, v in cal.items()}, "freq": alias,
                  "week_rule": week_rule(cal) if (cal.get("week") or cal.get("yearweek")) else None, "found_by": how,
                  "rule": "MMWR: week ending Saturday; ISO: week starting Monday; months / quarters / years: first day; "
                          "financial year 1997-98 -> 1997-01-01"})
    return parts2, {**ov2, "targets": tg, "covariates": cov, "freq": ov.get("freq") or alias,
                    "synthetic_time": False}, notes
def adapt(parts: list[Part], ov: dict, row: dict, roles_for) -> tuple[list[Part], dict, list[dict]]:
    """roles_for: canon.roles_for (passed in to avoid an import cycle)."""
    from tideforge.tfbench.canon import Fail
    notes: list[dict] = []
    if ov.get("calendar"):
        cal = ov["calendar"] if isinstance(ov["calendar"], dict) else find_calendar(parts[0])
        if not cal:
            raise Fail("calendar: no year / month / week columns found (name them in overrides.yaml calendar:)")
        miss = [v for k, v in cal.items() if k != "week_rule" and v not in parts[0].schema.names]
        if miss:
            raise Fail(f"calendar: columns not in the table: {miss}")
        return _with_calendar(parts, ov, cal, roles_for, notes, "overrides.yaml")
    if ov.get("time_col") or ov.get("layout") == "nested":
        if ov.get("time_col") and (c := numeric_time_col(parts[0].schema, ov["time_col"])):
            if hp := _hour_path(parts, c, ov, notes):
                return hp
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
    dates = date_columns(parts[0].schema)
    if len(dates) >= 3 and not _is_time_typed(parts[0].schema, r["time_col"]):
        # (b) first: dates / datetimes as column names are the time axis, ahead of any path or synthetic fallback
        ids = list(ov.get("id_cols") or [f.name for f in parts[0].schema if f.name not in dates and (
            pa.types.is_string(f.type) or pa.types.is_large_string(f.type)) and not UNNAMED.match(f.name)][:4])
        item_re, sum_rows = ov.get("id_from_path"), bool(ov.get("sum_rows"))
        m = melt(parts, [] if sum_rows else ids, item_re, sum_rows)
        idc = [c for c in m[0].schema.names if c not in (TIME, "value")]
        notes.append({"step": "melt: dates as column names", "date_columns_first_file": len(dates), "id_cols": idc,
                      "first": str(min(dates.values())), "last_first_file": str(max(dates.values())),
                      "id_from_path": item_re, "sum_rows": sum_rows,
                      "rule": "every file uses its own date columns; " + ("rows of a file summed per time step"
                                                                          if sum_rows else "one series per row")})
        return m, {**ov, "time_col": TIME, "targets": ["value"], "covariates": [], "static": [], "id_cols": idc,
                   "layout": "long", "synthetic_time": False,
                   "agg": {**(ov.get("agg") or {}), **({"value": "sum"} if sum_rows else {})}}, notes
    if not _is_time_typed(parts[0].schema, r["time_col"]) and (cal := find_calendar(parts[0])):
        if not r["time_col"] or r["time_col"] in cal.values():   # never replaces a real (e.g. epoch) time column
            return _with_calendar(parts, ov, cal, roles_for, notes, "column names and values")
    if r["time_col"] and str(r.get("time_rule", "")).startswith("calendar column name"):
        pt = path_time(parts)          # an hour / day column alone is not a time axis when the path carries the date
        if pt:
            notes.append({"step": "time from the file path", "parts": len(pt), "dropped_undated": len(parts) - len(pt),
                          "calendar_column": r["time_col"]})
            return pt, {**ov, "time_col": TIME}, notes
    if r["time_col"]:
        c = numeric_time_col(parts[0].schema, r["time_col"])
        if c and (hp := _hour_path(parts, c, ov, notes)):
            return hp
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
    ids = [] if ov.get("layout") == "wide" else (list(ov.get("id_cols") or r["id_cols"]) or [
        f.name for f in parts[0].schema if (pa.types.is_string(f.type) or pa.types.is_large_string(f.type))
        and not UNNAMED.match(f.name)][:3])          # wide (overrides.yaml): every numeric column is a series
    notes.append({"step": "synthetic time from row order", "freq": alias, "freq_source": how, "id_cols": ids,
                  "rule": f"rows in file order per series, from {SYN_START.date()}"})
    extra = {"id_cols": ids, "targets": r["target_cols"], "layout": "long"} if ids else {}
    return index_time(parts, ids, alias), {**ov, "time_col": TIME, "synthetic_time": True, "freq": alias,
                                           **extra}, notes
