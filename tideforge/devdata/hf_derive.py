"""Derived fields of the Hugging Face catalog profile: pure functions, no network (tideforge.devdata.hf_profile calls them).

Everything is derived from metadata: column names and types (viewer features or parquet schema), row counts, and
per-column statistics (parquet row-group statistics, viewer /statistics). Every derived value carries a confidence
("exact" / "estimated" / "unknown") and the name of the source or rule that produced it.

A "table" is the unit of derivation: one set of files (or one viewer config) with one schema:
    {"name", "cols": [Col], "rows": int|None, "rows_conf", "rows_src", "bytes": int, "n_files": int,
     "stats": {column: {"min", "max", "nulls", "num_values", "n_unique", "len_min", "len_median", "len_max",
                        "len_mean", "values": {value: count}}}, "complete": bool}
`complete` = every file of the table was read (False when footers were sampled or the viewer answer is partial).
"""
from __future__ import annotations

import datetime as dt
import math
import re
from dataclasses import dataclass

EXACT, EST, UNK = "exact", "estimated", "unknown"


# ---- columns -----------------------------------------------------------------------------------------------------------
@dataclass
class Col:
    name: str
    kind: str  # time | num | str | cat | bool | list_num | list_time | list_other | other
    dtype: str = ""
    depth: int = 0  # list nesting
    width: int | None = None  # fixed outer length of a 2-level list (= variates of a gluonts-style target)

    @property
    def label(self) -> str:
        return f"{self.name}:{self.dtype or self.kind}"


def _scalar_kind(dtype: str) -> str:
    d = (dtype or "").lower()
    if d.startswith(("timestamp", "date")):
        return "time"
    if d.startswith(("int", "uint", "float", "double", "decimal", "halffloat")):
        return "num"
    if d in ("string", "large_string", "str"):
        return "str"
    if d == "bool":
        return "bool"
    return "other"


def col_from_feature(name: str, f) -> Col:
    """A `datasets` feature (viewer /info, dataset_info.json) -> Col."""
    depth, width, lengths = 0, None, []
    while True:
        if isinstance(f, list) and f:
            depth, f = depth + 1, f[0]
            lengths.append(None)
        elif isinstance(f, dict) and f.get("_type") in ("Sequence", "List", "LargeList"):
            depth += 1
            lengths.append(f.get("length"))
            f = f.get("feature")
        elif isinstance(f, dict) and str(f.get("_type", "")).startswith("Array"):
            depth += len(f.get("shape") or [1])
            lengths += list(f.get("shape") or [None])
            f = {"_type": "Value", "dtype": f.get("dtype", "")}
        else:
            break
    if depth >= 2 and isinstance(lengths[0], int) and lengths[0] > 0:
        width = lengths[0]
    if not isinstance(f, dict):
        return Col(name, "list_other" if depth else "other", "", depth, width)
    t = f.get("_type")
    if t == "ClassLabel":
        kind, dtype = "cat", "class_label"
    elif t == "Value":
        dtype = str(f.get("dtype", ""))
        kind = _scalar_kind(dtype)
    else:
        kind, dtype = "other", str(t or "struct")
    if depth:
        kind = {"num": "list_num", "time": "list_time"}.get(kind, "list_other")
        dtype = "list<" * depth + dtype + ">" * depth
    return Col(name, kind, dtype, depth, width)


def cols_from_features(features) -> list[Col]:
    if isinstance(features, dict):
        return [col_from_feature(k, v) for k, v in features.items()]
    if isinstance(features, list):  # card form: [{"name": ..., "dtype": ... | "sequence": ...}]
        out = []
        for f in features:
            if isinstance(f, dict) and "name" in f:
                if "dtype" in f and isinstance(f["dtype"], str):
                    out.append(Col(f["name"], _scalar_kind(f["dtype"]), f["dtype"]))
                elif "sequence" in f or "list" in f:
                    inner = f.get("sequence", f.get("list"))
                    d, depth = inner, 1
                    while isinstance(d, dict) and ("sequence" in d or "list" in d):
                        d, depth = d.get("sequence", d.get("list")), depth + 1
                    dtype = d if isinstance(d, str) else (d.get("dtype", "") if isinstance(d, dict) else "")
                    k = {"num": "list_num", "time": "list_time"}.get(_scalar_kind(str(dtype)), "list_other")
                    out.append(Col(f["name"], k, "list<" * depth + str(dtype) + ">" * depth, depth))
                else:
                    out.append(Col(f["name"], "other", "struct"))
        return out
    return []


def _leaf_kind(physical: str, logical: str) -> str:
    lg, ph = (logical or "None"), (physical or "").upper()
    if lg.startswith(("Timestamp", "Date")) or ph == "INT96":
        return "time"
    if lg.startswith(("String", "Enum", "JSON", "UUID")):
        return "str"
    if lg.startswith("Time"):
        return "other"
    if ph == "BOOLEAN":
        return "bool"
    if ph in ("INT32", "INT64", "FLOAT", "DOUBLE") or lg.startswith(("Decimal", "Int", "Float16")):
        return "num"
    if ph == "BYTE_ARRAY":
        return "str"
    return "other"


_LIST_MARK = re.compile(r"\.(list|bag|array)(\.|$)")


def leaf_column_name(path: str, rep: int) -> str:
    """Top-level column of a parquet leaf. Flat columns keep their full path (a column name may hold dots)."""
    if rep <= 0:
        return path
    m = _LIST_MARK.search(path)
    return path[:m.start()] if m else path.split(".")[0]


def cols_from_leaves(leaves: list[dict]) -> list[Col]:
    """Parquet leaves [{"path", "physical", "logical", "rep"}] -> columns (a list column = its one repeated leaf)."""
    order, groups = [], {}
    for lf in leaves:
        name = leaf_column_name(lf["path"], lf.get("rep", 0))
        if name not in groups:
            groups[name] = []
            order.append(name)
        groups[name].append(lf)
    out = []
    for name in order:
        g = groups[name]
        if len(g) > 1:
            out.append(Col(name, "list_other" if g[0].get("rep", 0) else "other", "struct", g[0].get("rep", 0)))
            continue
        lf = g[0]
        kind, rep = _leaf_kind(lf.get("physical", ""), lf.get("logical", "")), int(lf.get("rep", 0))
        dtype = (lf.get("physical") or "").lower()
        lg = lf.get("logical") or "None"
        if lg != "None":
            dtype = lg.split("(")[0].lower()
        if rep:
            kind = {"num": "list_num", "time": "list_time"}.get(kind, "list_other")
            dtype = "list<" * rep + dtype + ">" * rep
        out.append(Col(name, kind, dtype, rep))
    return out


# ---- names ---------------------------------------------------------------------------------------------------------------
def tokens(name: str) -> list[str]:
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return [t.lower() for t in re.findall(r"[A-Za-z]+|\d+", s)]


TIME_STRONG = {"date", "time", "timestamp", "datetime", "ds", "dt", "period", "utc", "ts", "datum", "fecha"}
TIME_WEAK = {"year", "month", "week", "day", "hour", "quarter", "minute"}
ID_TOKENS = {"id", "ids", "item", "series", "station", "site", "store", "sku", "symbol", "ticker", "code", "key", "uuid",
             "sensor", "device", "meter", "node", "turbine", "building", "household", "customer", "client", "user",
             "country", "market", "commodity", "product", "entity", "cell", "well", "filename", "file", "name",
             "location", "city", "region", "state", "county", "zone", "fips", "iso", "iso3", "stock", "asset",
             "pair", "route", "line", "link", "road", "segment", "plant", "farm", "machine", "vehicle", "vessel", "mmsi"}
INT_ID_TOKENS = {"id", "ids", "code", "key", "fips", "uuid", "mmsi", "sku"}
GEO_TOKENS = {"lat", "latitude", "lon", "lng", "long", "longitude", "elevation", "altitude", "x", "y"} - {"x", "y"}
TARGET_NAMES = {"target", "targets", "value", "values", "y", "label", "target_value"}
FREQ_NAMES = {"freq", "frequency"}
EVENT_TOKENS = {"incident", "incidents", "transaction", "transactions", "trade", "trades", "trip", "trips", "tweet",
                "message", "headline", "session", "sessions", "event", "events", "order", "orders", "ticket", "alert",
                "outage", "earthquake", "accident", "crime", "claim", "tick", "ticks"}
FUTURE_TOKENS = {"holiday", "holidays", "promo", "promotion", "promotions", "weekday", "weekend", "dow", "calendar",
                 "forecast", "forecasts", "nwp", "scheduled", "schedule", "planned", "dayofweek", "season", "tariff"}
FUTURE_NAMES = {"feat_dynamic_real", "feat_dynamic_cat", "hour", "month", "day", "year", "week", "quarter"}
PAST_TOKENS = {"past", "observed", "actual", "measured", "lag", "lagged", "realized", "realised"}


def is_time_name(name: str) -> int:
    """2 = strong time name, 1 = weak (year, month, ...), 0 = no."""
    tk = tokens(name)
    if not tk:
        return 0
    if tk[0] in TIME_STRONG or tk[-1] in TIME_STRONG or "".join(tk) in ("datetime", "timestamp", "starttime"):
        return 2
    if len(tk) == 1 and tk[0] in TIME_WEAK:
        return 1
    return 0


def is_id_name(name: str) -> bool:
    return bool(set(tokens(name)) & ID_TOKENS)


def event_tokens(name: str) -> set[str]:
    """Event words in a column that names the event itself ("trade_id", "incident_date", "trips"), not a covariate
    that merely mentions one ("event_name_1" in a sales table)."""
    tk = tokens(name)
    hit = set(tk) & EVENT_TOKENS
    if hit and (len(tk) == 1 or set(tk) & {"id", "ids", "time", "date", "timestamp", "datetime", "type", "uuid", "hash"}):
        return hit
    return set()


def assign_roles(cols: list[Col], stats: dict | None = None) -> dict:
    """Time column, id columns, value / target / covariate / static columns, with the rule for each choice."""
    stats = stats or {}
    typed = [c for c in cols if c.kind in ("time", "list_time")]
    nested = any(c.kind == "list_num" for c in cols)
    time_col, time_rule = None, "none"
    if typed:
        pick = sorted(typed, key=lambda c: (-is_time_name(c.name), cols.index(c)))[0]
        time_col, time_rule = pick.name, "typed timestamp/date column"
    else:
        named = [c for c in cols if c.kind in ("str", "num") and is_time_name(c.name) == 2]
        weak = [c for c in cols if c.kind in ("str", "num") and is_time_name(c.name) == 1]
        if named:
            time_col, time_rule = named[0].name, f"column name ({named[0].kind} type, not a timestamp type)"
        elif weak:
            time_col, time_rule = weak[0].name, f"calendar column name ({weak[0].kind} type)"
    weak_time = {c.name for c in cols if c.kind in ("str", "num") and is_time_name(c.name) >= 1}
    freq_cols = [c.name for c in cols if c.name.lower() in FREQ_NAMES and c.kind in ("str", "cat")]
    ids, seen_stems = [], set()
    for c in cols:
        if c.name == time_col or c.name in freq_cols or c.depth:
            continue
        tk = set(tokens(c.name))
        hit = (c.kind in ("str", "cat") and tk & ID_TOKENS) or (c.kind == "num" and tk & INT_ID_TOKENS)
        if not hit:
            continue
        stem = "".join(t for t in tokens(c.name) if t not in ("id", "ids", "code", "name", "key")) or c.name.lower()
        if stem in seen_stems:  # market + market_id: one identifier, not two
            continue
        seen_stems.add(stem)
        ids.append(c.name)
    id_all = {c.name for c in cols if not c.depth and c.name != time_col and (
        (c.kind in ("str", "cat") and set(tokens(c.name)) & ID_TOKENS)
        or (c.kind == "num" and set(tokens(c.name)) & INT_ID_TOKENS))}
    geo = {c.name for c in cols if c.kind == "num" and set(tokens(c.name)) & GEO_TOKENS}
    if nested:
        value = [c.name for c in cols if c.kind == "list_num"]
        tgt = [n for n in value if n.lower() in TARGET_NAMES]
        target_rule = "list column named target/value"
        if not tgt:
            tgt, target_rule = value[:1], "first numeric list column (no column named target)"
        cov = [n for n in value if n not in tgt] + [c.name for c in cols if c.kind == "list_other"]
        static = [c.name for c in cols if not c.depth and c.name != time_col and c.name not in freq_cols
                  and c.name not in ids and c.kind in ("num", "str", "cat", "bool")]
    else:
        value = [c.name for c in cols if c.kind == "num" and c.name != time_col and c.name not in id_all
                 and c.name not in geo and c.name not in weak_time]
        tgt = [n for n in value if n.lower() in TARGET_NAMES]
        target_rule = "column named target/value/y"
        if not tgt and len(value) == 1:
            tgt, target_rule = value[:], "the only numeric value column"
        elif not tgt and ids:
            tgt, target_rule = value[:1], "first numeric value column (no column named target)"
        elif not tgt:
            tgt, target_rule = value[:], "all numeric columns (no id column: every column is a series)"
        cov = [n for n in value if n not in tgt] + [c.name for c in cols if c.kind == "bool"]
        static = sorted(geo) + [c.name for c in cols if c.kind in ("str", "cat") and c.name != time_col
                                and c.name not in ids and c.name not in freq_cols and c.name not in weak_time]
    return {"time_col": time_col, "time_rule": time_rule, "id_cols": ids, "freq_cols": freq_cols, "value_cols": value,
            "target_cols": tgt, "target_rule": target_rule, "covariate_cols": cov, "static_cols": static,
            "nested": nested}


def covariate_timing(names: list[str]) -> tuple[list[str], list[str]]:
    """-> (known in the future, past only), only where the column name makes it clear."""
    fut, past = [], []
    for n in names:
        tk = set(tokens(n))
        if tk & PAST_TOKENS:
            past.append(n)
        elif n.lower() in FUTURE_NAMES or tk & FUTURE_TOKENS or (tokens(n)[:1] == ["is"]):
            fut.append(n)
    return fut, past


# ---- layout --------------------------------------------------------------------------------------------------------------
def decide_layout(cols: list[Col], roles: dict) -> tuple[str, str, str]:
    """-> (layout, the rule that fired, confidence). Layouts: nested, long, wide, events, unknown."""
    lists = [c.name for c in cols if c.kind == "list_num"]
    if lists:
        return "nested", f"R1 numeric list column ({', '.join(lists[:3])}): one row per series", EXACT
    if not cols:
        return "unknown", "R0 no schema", UNK
    value, ids, tcol = roles["value_cols"], roles["id_cols"], roles["time_col"]
    ev = sorted({t for c in cols for t in event_tokens(c.name)})
    if not value:
        if tcol:
            return "events", "R2 time column but no numeric value column: one row per event", EST
        return "unknown", "R0 no numeric value column and no time column", UNK
    if ev and tcol:
        return "events", f"R3 event-like column names ({', '.join(ev[:4])}) with a time column", EST
    if ids:
        how = "time column + " if tcol else "no time column; "
        return "long", f"R4 {how}id column(s) ({', '.join(ids[:4])}) + {len(value)} numeric value column(s)", EST
    if tcol:
        return "wide", f"R5 time column, no id column, {len(value)} numeric column(s): one column per series", EST
    return "wide", f"R6 no time or id column, {len(value)} numeric column(s): assumed one column per series", EST


# ---- time values ---------------------------------------------------------------------------------------------------------
def to_datetime(v, name: str = "") -> tuple[dt.datetime | None, str]:
    """A statistics value -> (naive UTC datetime, confidence). Strings and plausible epoch numbers are 'estimated'."""
    if v is None:
        return None, UNK
    if isinstance(v, dt.datetime):
        if v.tzinfo is not None:
            v = v.astimezone(dt.timezone.utc).replace(tzinfo=None)
        return dt.datetime(v.year, v.month, v.day, v.hour, v.minute, v.second), EXACT
    if isinstance(v, dt.date):
        return dt.datetime(v.year, v.month, v.day), EXACT
    if isinstance(v, (bytes, bytearray)):
        try:
            v = v.decode("utf-8")
        except Exception:
            return None, UNK
    if isinstance(v, str):
        s = v.strip()
        m = re.match(r"^(\d{4})[-/.](\d{1,2})(?:[-/.](\d{1,2}))?(?:[T ](\d{1,2}):(\d{2})(?::(\d{2}))?)?", s)
        if m:
            y, mo, d, h, mi, se = (int(x) if x else None for x in m.groups())
            try:
                return dt.datetime(y, mo, d or 1, h or 0, mi or 0, se or 0), EST
            except ValueError:
                return None, UNK
        m = re.match(r"^(\d{4})(\d{2})(\d{2})$", s)
        if m and 1700 <= int(m.group(1)) <= 2100:
            try:
                return dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))), EST
            except ValueError:
                return None, UNK
        if re.fullmatch(r"\d{4}", s) and 1500 <= int(s) <= 2100:
            return dt.datetime(int(s), 1, 1), EST
        return None, UNK
    if isinstance(v, bool):
        return None, UNK
    if isinstance(v, (int, float)) and math.isfinite(v):
        x = float(v)
        if 1500 <= x <= 2100 and "year" in tokens(name):
            return dt.datetime(int(x), 1, 1), EST
        for div in (1.0, 1e3, 1e6, 1e9):  # epoch seconds, ms, us, ns
            sec = x / div
            if 0 < sec < 4.2e9 and (div == 1.0 and sec > 1e8 or div > 1.0 and sec > 1e8):
                try:
                    return dt.datetime(1970, 1, 1) + dt.timedelta(seconds=sec), EST
                except OverflowError:
                    return None, UNK
    return None, UNK


def date_range(tables: list[dict], roles_by_table: list[dict]) -> dict:
    """First / last timestamp over the tables' time columns (footer or viewer statistics)."""
    lo = hi = None
    conf, srcs = EXACT, set()
    for t, roles in zip(tables, roles_by_table):
        tc = roles.get("time_col")
        st = (t.get("stats") or {}).get(tc) if tc else None
        if not st or st.get("min") is None or st.get("max") is None:
            continue
        a, ca = to_datetime(st["min"], tc)
        b, cb = to_datetime(st["max"], tc)
        if a is None or b is None or b < a:
            continue
        lo = a if lo is None or a < lo else lo
        hi = b if hi is None or b > hi else hi
        if EST in (ca, cb) or not t.get("complete", True):
            conf = EST
        srcs.add(f"{t.get('stats_src', 'statistics')} min/max of {tc}")
    if lo is None:
        return {"first": None, "last": None, "span_years": None, "conf": UNK, "src": ""}
    return {"first": lo, "last": hi, "span_years": (hi - lo).total_seconds() / (365.25 * 86400.0), "conf": conf,
            "src": "; ".join(sorted(srcs))}


# ---- frequency -------------------------------------------------------------------------------------------------------------
_WORDS = {"secondly": 1, "minutely": 60, "hourly": 3600, "daily": 86400, "weekly": 604800, "monthly": 2629800,
          "quarterly": 7889400, "yearly": 31557600, "annual": 31557600, "annually": 31557600, "hour": 3600,
          "minute": 60, "min": 60, "day": 86400, "week": 604800, "month": 2629800, "year": 31557600, "quarter": 7889400}
_UNIT_CI = [(("s", "sec", "secs", "second", "seconds"), 1), (("min", "mins", "minute", "minutes"), 60),
            (("h", "hr", "hrs", "hour", "hours"), 3600), (("d", "day", "days"), 86400),
            (("w", "wk", "week", "weeks"), 604800), (("mo", "mon", "month", "months"), 2629800),
            (("q", "qtr", "quarter"), 7889400), (("y", "yr", "year", "years"), 31557600)]
_ALIAS = {"S": 1, "T": 60, "H": 3600, "D": 86400, "B": 86400, "W": 604800, "M": 2629800, "MS": 2629800, "ME": 2629800,
          "Q": 7889400, "QS": 7889400, "QE": 7889400, "Y": 31557600, "A": 31557600, "YS": 31557600, "YE": 31557600,
          "AS": 31557600, "min": 60, "h": 3600, "s": 1, "ms": 0.001, "L": 0.001, "us": 1e-6, "U": 1e-6}


def freq_label(seconds: float) -> str:
    for unit, name in ((31557600, "Y"), (7889400, "Q"), (2629800, "M"), (604800, "W"), (86400, "D"), (3600, "h"),
                       (60, "min"), (1, "s")):
        n = seconds / unit
        if n >= 1 and abs(n - round(n)) < 1e-9:
            return f"{int(round(n))}{name}"
    return f"{seconds:g}s"


def parse_freq_alias(value: str) -> float | None:
    """A pandas-style offset alias as written in a `freq` column or feature ("15T", "H", "1D", "W-SUN", "5min")."""
    s = str(value).strip()
    m = re.fullmatch(r"(\d+)?\s*([A-Za-z]+)(?:-[A-Za-z]{3})?", s)
    if not m:
        return None
    n, unit = int(m.group(1) or 1), m.group(2)
    sec = _ALIAS.get(unit)
    if sec is None:
        sec = _ALIAS.get(unit.upper()) if len(unit) <= 2 else None
    if sec is None:
        sec = _WORDS.get(unit.lower())
    return None if sec is None else n * sec


def freq_from_name(name: str) -> float | None:
    """Frequency stated in a config / folder / file name, in seconds; None when the name does not state one.

    Accepted: words (hourly, daily, monthly, ...), number + unit tokens ("15min", "1h", "30T", "5_minutes"), and a
    bare pandas alias as the LAST token ("ds-10-T" = 1 minute, "x_H"). A lone lowercase "m" after a number is minutes ("1m"), an
    uppercase "M" is months ("1M"). A bare letter elsewhere is never read as a frequency ("M_DENSE" is not monthly).
    """
    raw = [t for t in re.split(r"[^A-Za-z0-9]+", str(name)) if t]
    found = []
    for i, t in enumerate(raw):
        last = i == len(raw) - 1
        prev_num = int(raw[i - 1]) if i > 0 and raw[i - 1].isdigit() and len(raw[i - 1]) <= 3 else None
        m = re.fullmatch(r"(\d{1,4})([A-Za-z]+)", t)
        sec = None
        if m:
            n, unit = int(m.group(1)), m.group(2)
            if unit == "m":
                sec = n * 60
            elif unit == "M":
                sec = n * 2629800
            elif unit in ("T",):
                sec = n * 60
            else:
                for names, mult in _UNIT_CI:
                    if unit.lower() in names:
                        sec = n * mult
                        break
        elif t.lower() in _WORDS and (len(t) > 3 or t.lower() in ("min", "day")):
            base = _WORDS[t.lower()]
            sec = base * (prev_num if prev_num and not t.lower().endswith("ly") else 1)
        elif last and len(raw) > 1 and t in ("T", "H", "D", "W", "M", "Q", "Y", "S"):
            sec = _ALIAS[t]  # "ds-10-T": the number before a bare alias is an id, not a multiple
        if sec:
            found.append(sec)
    found = sorted(set(found))
    return found[0] if len(found) == 1 else None


_CANON = [1, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600, 43200, 86400, 604800, 2629800,
          7889400, 31557600]


def freq_implied(span_seconds: float | None, length: float | None) -> tuple[float | None, str]:
    """Step implied by (span, length) for regular data, snapped to a standard step; (None, why) when it does not snap."""
    if not span_seconds or not length or length < 3 or span_seconds <= 0:
        return None, "no span or length"
    step = span_seconds / (length - 1.0)
    for c in _CANON:
        tol = 0.06 if c >= 2629800 else 0.03
        if abs(step / c - 1.0) <= tol:
            return float(c), f"span / (length - 1) = {step:.4g} s"
    if 1.36 <= step / 86400.0 <= 1.52:  # 5 trading days a week, with holidays
        return 86400.0, f"span / (length - 1) = {step / 86400.0:.3f} days: business days"
    return None, f"span / (length - 1) = {step:.4g} s is not a standard step"


BANDS = ["sub-minute", "minute", "hourly", "daily", "weekly", "monthly", "quarterly / yearly", "irregular / events",
         "mixed", "unknown"]


def freq_band(seconds: float | None, layout: str = "", mixed: bool = False) -> str:
    if layout == "events":
        return "irregular / events"
    if mixed:
        return "mixed"
    if not seconds:
        return "unknown"
    if seconds < 60:
        return "sub-minute"
    if seconds < 3600:
        return "minute"
    if seconds < 86400:
        return "hourly"
    if seconds < 604800:
        return "daily"
    if seconds < 28 * 86400:
        return "weekly"
    if seconds < 80 * 86400:
        return "monthly"
    return "quarterly / yearly"


LEN_BANDS = ["under 100", "100-1k", "1k-10k", "10k-100k", "over 100k", "unknown"]


def length_band(n: float | None) -> str:
    if n is None:
        return "unknown"
    if n < 100:
        return "under 100"
    if n < 1000:
        return "100-1k"
    if n < 10000:
        return "1k-10k"
    if n < 100000:
        return "10k-100k"
    return "over 100k"


# ---- counts ----------------------------------------------------------------------------------------------------------------
def _worse(*confs: str) -> str:
    return UNK if UNK in confs else EST if EST in confs else EXACT


def table_counts(t: dict, roles: dict, layout: str) -> dict:
    """n_series, n_points (all numeric value columns), n_points_target and series length for one table.

    nested: series = rows; points = sum of the list lengths (parquet leaf value counts), else rows x mean list length.
    long:   series = n_unique(id) (several id columns: the product, capped at rows, estimated); points = rows x value
            columns.
    wide:   series = numeric columns; points = rows x numeric columns.
    events: series = n_unique(id) when known; points = rows x value columns (event rows, not a regular series).
    """
    rows, rconf, stats = t.get("rows"), t.get("rows_conf", UNK), t.get("stats") or {}
    out = {"n_series": None, "n_series_conf": UNK, "n_series_src": "", "n_points": None, "n_points_conf": UNK,
           "n_points_src": "", "n_points_target": None, "n_points_target_conf": UNK, "len_min": None,
           "len_median": None, "len_max": None, "len_conf": UNK, "len_src": "", "channels": 1}
    value, tgt, ids = roles["value_cols"], roles["target_cols"], roles["id_cols"]
    if rows is None:
        return out
    scale = t.get("scale", 1.0)
    if layout == "nested":
        out.update(n_series=rows, n_series_conf=rconf, n_series_src="rows (one row per series)")
        width = {c.name: c.width for c in t["cols"]}

        def total(names):
            tot, conf, src = 0.0, rconf, []
            for n in names:
                st = stats.get(n) or {}
                if st.get("num_values") is not None:
                    tot += st["num_values"] * scale
                    src.append("parquet leaf value count")
                    conf = _worse(conf, EXACT if t.get("complete", True) and scale == 1.0 else EST)
                elif st.get("len_mean") is not None:
                    tot += rows * st["len_mean"] * (width.get(n) or 1)
                    src.append("rows x mean list length (viewer)")
                    conf = _worse(conf, EST)
                else:
                    return None, UNK, f"no length statistics for {n}"
            return (int(round(tot)) if names else None), (conf if names else UNK), "; ".join(sorted(set(src)))

        p, pc, ps = total(value)
        out.update(n_points=p, n_points_conf=pc, n_points_src=ps)
        p, pc, _ = total(tgt)
        out.update(n_points_target=p, n_points_target_conf=pc)
        first = tgt[0] if tgt else None
        st = stats.get(first) or {}
        w = width.get(first) or 1
        out["channels"] = max(1, len(value))
        if st.get("len_median") is not None and w == 1:
            out.update(len_min=st.get("len_min"), len_median=st.get("len_median"), len_max=st.get("len_max"),
                       len_conf=EXACT if t.get("complete", True) and not t.get("viewer_partial") else EST,
                       len_src="viewer list-length statistics")
        elif out["n_points_target"] and rows:
            deep = any(c.name == first and c.depth >= 2 and not c.width for c in t["cols"])
            out.update(len_median=out["n_points_target"] / rows / w, len_conf=EST,
                       len_src="target points / rows (a mean, not a median)"
                       + ("; 2-level list with unknown variate count: this is length x variates" if deep else ""))
        return out
    nval, ntgt = len(value), len(tgt)
    if layout == "wide":
        out.update(n_series=nval, n_series_conf=EXACT, n_series_src="numeric columns",
                   n_points=int(rows * nval), n_points_conf=rconf, n_points_src="rows x numeric columns",
                   n_points_target=int(rows * nval), n_points_target_conf=rconf,
                   len_min=rows, len_median=rows, len_max=rows, len_conf=rconf, len_src="rows (every column)")
        return out
    out.update(n_points=int(rows * nval), n_points_conf=rconf, n_points_src="rows x value columns",
               n_points_target=int(rows * ntgt), n_points_target_conf=rconf, channels=max(1, nval))
    prod, known, how = 1, bool(ids), []
    for n in ids:
        st = stats.get(n) or {}
        if st.get("n_unique") is not None:
            prod *= max(1, int(st["n_unique"]))
            how.append(f"n_unique({n})")
        elif isinstance(st.get("min"), int) and isinstance(st.get("max"), int) and not isinstance(st["min"], bool):
            prod *= max(1, st["max"] - st["min"] + 1)
            how.append(f"max-min+1 of {n}")
        else:
            known = False
    if t.get("partition_series"):
        out.update(n_series=int(t["partition_series"]), n_series_conf=EST,
                   n_series_src=f"distinct values of the path partition {t.get('partition_key', '')}")
    elif known:
        single_exact = len(ids) == 1 and how[0].startswith("n_unique") and t.get("complete", True) \
            and not t.get("viewer_partial") and not t.get("n_unique_merged")
        out.update(n_series=int(min(prod, rows)), n_series_conf=EXACT if single_exact and rconf == EXACT else EST,
                   n_series_src=" x ".join(how) + ("" if len(ids) == 1 else " (product, capped at rows)"))
    if out["n_series"]:
        out.update(len_median=rows / out["n_series"], len_conf=EST, len_src="rows / n_series (a mean)")
    return out


# ---- slices, ingest --------------------------------------------------------------------------------------------------------
SLICES = (("slice500", 500, 16_000), ("slice5k", 5_000, 64_000))


def slice_estimates(n_series, n_points, bytes_all: int | None, channels: float = 1.0, partial_fetch: bool = True) -> dict:
    """Points and bytes for (i) 500 series x last 16k points, (ii) 5,000 x 64k, (iii) everything. Estimates.

    points = min(S, n_series) x min(L x channels, points per series). Bytes are proportional to points when the files
    are parquet (the real fetch is rounded up to whole row groups or files); other formats cannot be cut, so every
    slice costs all bytes.
    """
    out = {"slice_all_points": n_points, "slice_all_bytes": bytes_all}
    for name, s, length in SLICES:
        if not n_series or not n_points:
            out[f"{name}_points"], out[f"{name}_bytes"] = None, (None if partial_fetch else bytes_all)
            continue
        per = n_points / n_series
        pts = min(s, n_series) * min(length * max(1.0, channels), per)
        pts = int(min(pts, n_points))
        out[f"{name}_points"] = pts
        if bytes_all is None:
            out[f"{name}_bytes"] = None
        else:
            out[f"{name}_bytes"] = int(round(bytes_all * pts / n_points)) if partial_fetch else bytes_all
    return out


HARD_EXT = {"zip", "rar", "7z", "tar", "tar.gz", "tgz", "nc", "nc4", "netcdf", "h5", "hdf5", "hdf", "npy", "npz", "mat",
            "pkl", "pickle", "pt", "pth", "zarr", "grib", "grib2", "tsf", "ts", "bin", "dat", "db", "sqlite", "feather",
            "xlsx", "xls", "tfrecord", "lmdb", "gz", "zst", "bz2", "xz"}
MEDIUM_EXT = {"csv", "tsv", "json", "jsonl", "ndjson", "txt", "arrow", "csv.gz", "tsv.gz", "json.gz", "jsonl.gz",
              "csv.zst", "jsonl.zst", "csv.bz2", "txt.gz", "csv.xz"}


def ingest_difficulty(ext_bytes: dict, n_schemas: int, layout: str, has_loading_script: bool = False) -> tuple[str, str]:
    """easy = parquet with one schema; medium = csv / several tables / nested arrays; hard = archives, netcdf, h5, npy,
    custom formats, event tables that need aggregation. -> (level, reason)."""
    exts = {e for e, b in ext_bytes.items() if b >= 0}
    hard = sorted(e for e in exts if e in HARD_EXT)
    other = sorted(e for e in exts if e not in HARD_EXT and e not in MEDIUM_EXT and e != "parquet")
    if not exts:
        return "hard", "no data files found"
    if hard:
        return "hard", "formats that need unpacking or a custom reader: " + ", ".join(hard)
    if other:
        return "hard", "unrecognised file formats: " + ", ".join(other)
    if layout == "events":
        return "hard", "event-level table: needs aggregation to a regular step"
    medium = []
    if exts & MEDIUM_EXT:
        medium.append("text / arrow files: " + ", ".join(sorted(exts & MEDIUM_EXT)))
    if n_schemas > 1:
        medium.append(f"{n_schemas} different table schemas")
    if layout == "nested":
        medium.append("nested arrays (one row per series)")
    if has_loading_script:
        medium.append("repo ships a loading script")
    if layout == "unknown":
        medium.append("layout not determined from metadata")
    if medium:
        return "medium", "; ".join(medium)
    return "easy", "parquet, one schema"


# ---- formatting ------------------------------------------------------------------------------------------------------------
def fmt_bytes(n) -> str:
    if n is None or n == "":
        return "unknown"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.2f} {unit}"
        n /= 1000.0
    return f"{n:.2f} TB"


def fmt_count(n) -> str:
    if n is None or n == "":
        return "unknown"
    n = float(n)
    for div, unit in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "k")):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:.0f}"
