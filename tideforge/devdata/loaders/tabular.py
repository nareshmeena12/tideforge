"""The generic `table` loader: csv / whitespace / fwf / parquet / json / xlsx files (plain, .gz, .zip or .tar[.gz]) -> long format, driven by the
read / time / id / value / events / filter / missing / scale keys of a sources.yaml entry."""
from __future__ import annotations

import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

READ_KEYS = {"format", "zip_member", "strip_columns", "drop_first_rows", "record_path", "normalize", "encoding", "melt",
             "fillna", "rename_regex", "empty_ok"}


def _frames(path: Path, read: dict):
    """One DataFrame per table in the file (a zip may hold several members)."""
    opts = {k: v for k, v in read.items() if k not in READ_KEYS}
    fmt = read.get("format", "csv")
    if fmt == "xlsx":                               # an xlsx is itself a zip: read it whole, never as zip members
        yield _parse(path, fmt, opts, read, None)
        return
    with open(path, "rb") as fh:                    # by content, not name: some endpoints answer .csv with a zip or gzip
        magic = fh.read(4)
    if path.suffix == ".zip" or magic[:4] == b"PK\x03\x04":
        pat = re.compile(read.get("zip_member", r".*\.(csv|txt|tsv|parquet|json)$"), re.I)
        with zipfile.ZipFile(path) as z:
            for name in sorted(z.namelist()):
                if pat.search(name) and not name.endswith("/") and "__MACOSX" not in name:
                    with z.open(name) as fh:
                        yield _parse(io.BytesIO(fh.read()), fmt, opts, read, None)
        return
    if tarfile.is_tarfile(path):                    # e.g. MIMIT's quarterly .tar.gz of CSVs: members like a zip's
        pat = re.compile(read.get("zip_member", r".*\.(csv|txt|tsv|parquet|json)$"), re.I)
        with tarfile.open(path) as t:
            for m in sorted(t.getmembers(), key=lambda m: m.name):
                if m.isfile() and pat.search(m.name):
                    yield _parse(io.BytesIO(t.extractfile(m).read()), fmt, opts, read, None)
        return
    comp = "gzip" if path.suffix == ".gz" or magic[:2] == b"\x1f\x8b" else None
    yield _parse(path, fmt, opts, read, comp)


def _parse(src, fmt, opts, read, comp):
    if fmt == "parquet":
        df = pd.read_parquet(src, **opts)
    elif fmt == "xlsx":                 # openpyxl: the `devdata` extra (owner 2026-10-31); sheet_name / header etc. in opts
        try:
            import openpyxl  # noqa: F401
        except ImportError as e:
            raise ImportError("xlsx sources need the devdata extra: pip install --user -e '.[devdata]'") from e
        df = pd.read_excel(src, engine="openpyxl", **opts)
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
    for pat, new in (read.get("rename_regex") or {}).items():      # e.g. USGS "<ts_id>_00060_00003" -> "flow"
        df = df.rename(columns={c: new for c in df.columns if re.search(pat, str(c))})
        if (df.columns == new).sum() > 1:         # e.g. two USGS sensors at one site: first non-empty value, left to right
            merged = df.loc[:, df.columns == new].bfill(axis=1).iloc[:, 0]
            df = df.loc[:, df.columns != new].assign(**{new: merged})
    if read.get("fillna") is not None:            # e.g. 0 where a missing cell means "no events"
        df = df.fillna(read["fillna"])
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


WHERE_OPS = {"eq", "ne", "lt", "le", "gt", "ge", "in", "not_in", "startswith", "not_startswith", "contains", "not_contains",
             "match", "not_match"}
DERIVE_OPS = {"prefix", "const", "copy", "strip"}


def where_mask(df: pd.DataFrame, spec: dict) -> pd.Series:
    """Rows to KEEP: {column: {op: value, ...}, ...}, every condition ANDed; plain vectorised pandas (no query / eval strings,
    which pandas 3 parses differently: '"slice" is not a supported function', Perlmutter 2026-10-31). Ops: eq ne lt le gt
    ge (a value), in / not_in (a list), startswith / not_startswith (a string or a list), contains / not_contains (a literal
    string), match / not_match (a regex at the start)."""
    keep = pd.Series(True, index=df.index)
    for col, conds in spec.items():
        c = df[col]
        for op, v in conds.items():
            s = c.astype(str)
            if op in ("eq", "ne", "lt", "le", "gt", "ge"):
                r = getattr(c, op)(v)
            elif op in ("in", "not_in"):
                r = c.isin(list(v))
            elif op in ("startswith", "not_startswith"):
                r = s.str.startswith(tuple(v) if isinstance(v, list) else v)
            elif op in ("contains", "not_contains"):
                r = s.str.contains(v, regex=False)
            elif op in ("match", "not_match"):
                r = s.str.match(v)
            else:
                raise KeyError(f"unknown where op {op!r} (known: {sorted(WHERE_OPS)})")
            r = r.fillna(False).astype(bool)
            keep &= ~r if op.startswith("not_") else r
    return keep


def derive(df: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """{new_col: {prefix: [column, n]}} (the first n characters), {const: value}, {copy: column}, {strip: column} (text with
    surrounding whitespace removed)."""
    for k, d in (spec or {}).items():
        (op, v), = d.items()
        if op == "prefix":
            df[k] = df[v[0]].astype(str).str[:int(v[1])]
        elif op == "const":
            df[k] = v
        elif op == "copy":
            df[k] = df[v]
        elif op == "strip":
            df[k] = df[v].astype(str).str.strip()
        else:
            raise KeyError(f"unknown derive op {op!r} (known: {sorted(DERIVE_OPS)})")
    return df


def parse_time(df: pd.DataFrame, t: dict, tz: str) -> pd.Series:
    """time spec -> UTC naive timestamps. {col}, {cols: [..]} (joined with a space) or {parts: {year: Y, month: M, ...}};
    format, unit (epoch numbers), tz (default the source's tz); strings with an offset are converted using it."""
    if "parts" in t:
        p = {k: pd.to_numeric(df[v], errors="coerce") for k, v in t["parts"].items()}
        ts = pd.to_datetime(pd.DataFrame(p), errors="coerce")
    else:
        if "expr" in t:                                       # e.g. "day * 86400 + time" (seconds), with unit
            s = df.eval(t["expr"], engine="python")
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
        df = df[where_mask(df, where)]
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
    empty_ok = bool(src.read.get("empty_ok"))           # a file with no table or no value column is skipped, not an error
    cols = src.value if isinstance(src.value, list) else [src.value]
    for path, vars_ in files:
        try:
            for df in _frames(Path(path), src.read):
                if empty_ok and (df.empty or any(c not in df.columns for c in cols if c)):
                    continue
                parts.append(to_long(df, src, vars_))
        except pd.errors.EmptyDataError:
            if not empty_ok:
                raise
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
        df = df[where_mask(df, ix["where"])]
    df["series"] = df[ix["id"]].astype(str).str.strip()
    cols = ["series"] + [c for c in ix.get("strata", []) if c != "series"]
    return df[cols].drop_duplicates("series").reset_index(drop=True)

