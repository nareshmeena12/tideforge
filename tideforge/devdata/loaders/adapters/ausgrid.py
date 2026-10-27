"""Ausgrid "Solar home electricity data" (D029): one CSV per year, one row per (customer, category, day) with 48 half-hour
columns ("0:30" ... "23:30", "0:00" = the period ending at midnight). Series: customer/category (GC general consumption,
CL controlled load, GG gross generation). Timestamps = period start, in src.tz (to be confirmed by the probe), then UTC.
read options: skiprows (title line above the header, default 1), date_format (default: day first)."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd

SLOT = re.compile(r"^\d{1,2}:\d{2}$")


def _tables(path: Path, skiprows: int):
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            for n in sorted(z.namelist()):
                if n.lower().endswith(".csv"):
                    with z.open(n) as fh:
                        yield pd.read_csv(fh, skiprows=skiprows, low_memory=False)
    else:
        yield pd.read_csv(path, skiprows=skiprows, low_memory=False)


def wide_to_long(df: pd.DataFrame, tz: str, date_format: str | None = None) -> pd.DataFrame:
    df.columns = [str(c).strip() for c in df.columns]
    slots = [c for c in df.columns if SLOT.match(c)]
    if len(slots) != 48:
        raise ValueError(f"expected 48 half-hour columns, found {len(slots)}")
    day = pd.to_datetime(df["date"], format=date_format, dayfirst=date_format is None, errors="coerce")
    sid = df["Customer"].astype(str) + "/" + df["Consumption Category"].astype(str).str.strip()
    vals = df[slots].apply(pd.to_numeric, errors="coerce").to_numpy()
    n = len(df)
    ts = pd.Series(day.to_numpy().repeat(48) + pd.to_timedelta([30 * k for k in range(48)] * n, unit="min").to_numpy())
    ts = ts.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT") if tz != "UTC" else ts.dt.tz_localize("UTC")
    out = pd.DataFrame({"series_id": sid.to_numpy().repeat(48), "timestamp": ts.dt.tz_convert("UTC").dt.tz_localize(None),
                        "value": vals.reshape(-1)})
    if "Postcode" in df:
        out["Postcode"] = df["Postcode"].to_numpy().repeat(48)
    return out.dropna(subset=["timestamp"])


def read_files(src, files):
    r = src.read or {}
    parts = [wide_to_long(df, src.tz, r.get("date_format"))
             for p, _ in files for df in _tables(Path(p), int(r.get("skiprows", 1)))]
    return pd.concat(parts, ignore_index=True)
