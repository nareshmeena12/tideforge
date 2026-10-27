"""The generic `table` loader: csv / whitespace / fwf / parquet / json files (plain, .gz or .zip) -> long format, driven by the
read / time / id / value / events / filter / missing / scale keys of a sources.yaml entry."""
from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd

READ_KEYS = {"format", "zip_member", "strip_columns", "drop_first_rows", "record_path", "normalize", "encoding", "melt"}


def _frames(path: Path, read: dict):
    """One DataFrame per table in the file (a zip may hold several members)."""
    opts = {k: v for k, v in read.items() if k not in READ_KEYS}
    fmt = read.get("format", "csv")
    if path.suffix == ".zip":
        pat = re.compile(read.get("zip_member", r".*\.(csv|txt|tsv|parquet|json)$"), re.I)
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                if pat.search(name) and not name.endswith("/") and "__MACOSX" not in name:
                    with z.open(name) as fh:
                        yield _parse(io.BytesIO(fh.read()), fmt, opts, read, None)
        return
    comp = "gzip" if path.suffix == ".gz" else None
    yield _parse(path, fmt, opts, read, comp)


def _parse(src, fmt, opts, read, comp):
    if fmt == "parquet":
        df = pd.read_parquet(src, **opts)
    elif fmt == "json":
        if isinstance(src, Path):
            raw = (__import__("gzip").open(src, "rt", encoding=read.get("encoding", "utf-8")) if comp
                   else open(src, encoding=read.get("encoding", "utf-8"))).read()
        else:
            raw = src.read().decode(read.get("encoding", "utf-8"))
        obj = json.loads(raw)
        for k in (read.get("record_path") or []):
            obj = obj[k]
        df = pd.json_normalize(obj) if read.get("normalize") else pd.DataFrame(obj)
    elif fmt == "fwf":
        df = pd.read_fwf(src, compression=comp, **opts)
    else:
        if fmt == "whitespace":
            opts = dict(opts, sep=r"\s+")
        df = pd.read_csv(src, compression=comp, encoding=read.get("encoding", "utf-8"), low_memory=False, **opts)
    if read.get("strip_columns"):
        df.columns = [str(c).strip() for c in df.columns]
    n = int(read.get("drop_first_rows", 0) or 0)
    df = df.iloc[n:].reset_index(drop=True) if n else df
    if read.get("melt"):            # wide (one column per series) -> long: melt: {id_vars: [time], var_name: series}
        m = read["melt"]
        df = df.melt(id_vars=m["id_vars"], var_name=m.get("var_name", "series"), value_name=m.get("value_name", "value"))
    return df


def apply_filter(df: pd.DataFrame, flt: dict) -> pd.DataFrame:
    """Keep rows whose column value is in the list (None in the list keeps missing values)."""
    for col, allowed in (flt or {}).items():
        allowed = allowed if isinstance(allowed, list) else [allowed]
        keep = df[col].isin([a for a in allowed if a is not None])
        if None in allowed:
            keep |= df[col].isna() | (df[col].astype(str).str.strip() == "")
        df = df[keep]
    return df


def derive(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """{new_col: "<pandas expression over columns>"}, e.g. {"country": "station.str[:2]"}."""
    for k, expr in (spec or {}).items():
        df[k] = df.eval(expr, engine="python")
    return df


def parse_time(df: pd.DataFrame, t: dict, tz: str) -> pd.Series:
    """time spec -> UTC naive timestamps. {col}, {cols: [..]} (joined with a space) or {parts: {year: Y, month: M, ...}};
    format, unit (epoch numbers), tz (default the source's tz); strings with an offset are converted using it."""
    if "parts" in t:
        p = {k: pd.to_numeric(df[v], errors="coerce") for k, v in t["parts"].items()}
        ts = pd.to_datetime(pd.DataFrame(p), errors="coerce")
    else:
        s = df[t["col"]] if "col" in t else df[t["cols"]].astype(str).agg(" ".join, axis=1)
        if "unit" in t:
            ts = pd.to_datetime(pd.to_numeric(s, errors="coerce"), unit=t["unit"], utc=True)
        else:
            ts = pd.to_datetime(s.astype(str), format=t.get("format"), errors="coerce", utc=bool(t.get("has_offset")))
    zone = t.get("tz", tz)
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(zone, ambiguous="NaT", nonexistent="NaT") if zone and zone != "UTC" else ts.dt.tz_localize("UTC")
    return ts.dt.tz_convert("UTC").dt.tz_localize(None)


def to_long(df: pd.DataFrame, src, file_vars: dict | None = None) -> pd.DataFrame:
    """One parsed table -> long rows (series_id, timestamp, value, strata...)."""
    for k, v in (file_vars or {}).items():
        df[f"_var_{k}"] = v
    df = apply_filter(df, src.filter)
    strata = list(src.select.get("strata", []) or [])
    df = derive(df, src.select.get("derive", {}))
    where = (src.exclude or {}).get("where")
    if where:
        df = df.query(where, engine="python")
    ts = parse_time(df, src.time, src.tz)
    sid = df[src.id_cols].astype(str).agg("/".join, axis=1) if src.id_cols else pd.Series("", index=df.index)
    keep = [c for c in strata if c in df.columns]
    if src.events:
        val = pd.to_numeric(df[src.events["value"]], errors="coerce") if src.events.get("value") else 1.0
        out = pd.DataFrame({"series_id": sid, "timestamp": ts, "value": val, **{c: df[c] for c in keep}})
        return out.dropna(subset=["timestamp"])
    cols = src.value if isinstance(src.value, list) else [src.value]
    parts = []
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").astype("float64")
        if src.missing:
            v = v.mask(v.isin([float(m) for m in src.missing]))
        if src.scale is not None:
            v = v * float(src.scale)
        name = sid + "/" + c if isinstance(src.value, list) else sid
        parts.append(pd.DataFrame({"series_id": name.str.lstrip("/"), "timestamp": ts, "value": v,
                                   **{k: df[k] for k in keep}}))
    out = pd.concat(parts, ignore_index=True)
    return out.dropna(subset=["timestamp"])


def read_files(src, files) -> pd.DataFrame:
    parts = []
    for path, vars_ in files:
        for df in _frames(Path(path), src.read):
            parts.append(to_long(df, src, vars_))
    if not parts:
        return pd.DataFrame(columns=["series_id", "timestamp", "value"])
    return pd.concat(parts, ignore_index=True)


def read_index(src, paths) -> pd.DataFrame:
    """The index files -> DataFrame with column series (the id used in url templates) + strata columns."""
    ix = src.index
    parts = []
    for path, vars_ in paths:
        for df in _frames(Path(path), ix.get("read", {"format": "csv"})):
            for k, v in (vars_ or {}).items():
                df[f"_var_{k}"] = v
            parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    df = apply_filter(df, ix.get("filter", {}))
    df = derive(df, ix.get("derive", {}))
    if ix.get("where"):
        df = df.query(ix["where"], engine="python")
    df["series"] = df[ix["id"]].astype(str).str.strip()
    cols = ["series"] + [c for c in ix.get("strata", []) if c != "series"]
    return df[cols].drop_duplicates("series").reset_index(drop=True)

