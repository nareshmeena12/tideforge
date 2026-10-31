"""Eurostat SDMX TSV (D297 demo_r_mwk3_ts and similar): first column "dim1,dim2,...,geo\\TIME_PERIOD", one column per period
("2000-W01", "2000-01", "2000"); values like "123", "123 p" (flag), ":" (missing). read options: keep (dimension filter, e.g.
{sex: T}); series = geo; stratum country = the first two letters of geo."""
from __future__ import annotations

import gzip
import io
from pathlib import Path

import pandas as pd


def _period(cols: list[str]) -> pd.DatetimeIndex:
    c = pd.Index(cols).str.strip()
    if c.str.contains("-W").all():
        return pd.to_datetime(c + "-1", format="%G-W%V-%u")
    if c.str.match(r"^\d{4}-\d{2}$").all():
        return pd.to_datetime(c, format="%Y-%m")
    return pd.to_datetime(c, format="%Y")


def parse(text: str, keep: dict | None = None) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str)
    head = df.columns[0]
    dims = head.split("\\")[0].split(",")
    key = df[head].str.split(",", expand=True)
    key.columns = dims
    for k, v in (keep or {}).items():
        ok = key[k].isin(v if isinstance(v, list) else [v])
        df, key = df[ok], key[ok]
    periods = _period(list(df.columns[1:]))
    vals = df.iloc[:, 1:].apply(lambda s: pd.to_numeric(s.str.strip().str.split(" ").str[0], errors="coerce"))
    out = pd.DataFrame({"series_id": key["geo"].to_numpy().repeat(len(periods)),
                        "timestamp": list(periods) * len(df), "value": vals.to_numpy().reshape(-1)})
    out["country"] = out["series_id"].str[:2]
    return out


def read_files(src, files):
    parts = []
    for p, _ in files:
        p = Path(p)
        with open(p, "rb") as f:                  # by content: compressed=true answers gzip whatever the name says
            gz = f.read(2) == b"\x1f\x8b"
        text = gzip.open(p, "rt", encoding="utf-8").read() if gz else p.read_text(encoding="utf-8")
        parts.append(parse(text, (src.read or {}).get("keep")))
    return pd.concat(parts, ignore_index=True)
