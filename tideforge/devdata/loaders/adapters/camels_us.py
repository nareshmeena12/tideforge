"""CAMELS-US (D071), the NCAR release basin_timeseries_v1p2_metForcing_obsFlow.zip: streamflow per gauge
(usgs_streamflow/<huc>/<gauge>_streamflow_qc.txt: gauge year month day Q flag; -999 = missing) and the basin-mean Daymet
forcing (basin_mean_forcing/daymet/<huc>/<gauge>_lump_cida_forcing_leap.txt: 3 header lines, then Year Mnth Day Hr dayl(s)
prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)). Series = gauge; stratum = huc (2-digit region). Daily, no time zone.
The member layout is from the release's documentation and is confirmed by the probe step."""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

FLOW = re.compile(r"usgs_streamflow/(\d{2})/(\d{8})_streamflow_qc\.txt$")
FORCING = re.compile(r"basin_mean_forcing/daymet/(\d{2})/(\d{8})_lump_cida_forcing_leap\.txt$")


def parse_flow(text: str) -> pd.DataFrame:
    df = pd.read_csv(io.StringIO(text), sep=r"\s+", header=None, names=["gauge", "y", "m", "d", "q", "flag"],
                     dtype={"gauge": str})
    q = df["q"].astype(float).mask(df["q"] < 0)
    ts = pd.to_datetime(dict(year=df["y"], month=df["m"], day=df["d"]))
    return pd.DataFrame({"timestamp": ts, "value": q})


def parse_forcing(text: str) -> pd.DataFrame:
    lines = text.splitlines()
    df = pd.read_csv(io.StringIO("\n".join(lines[3:])), sep=r"\s+")
    ts = pd.to_datetime(dict(year=df["Year"], month=df["Mnth"], day=df["Day"]))
    out = df.drop(columns=["Year", "Mnth", "Day", "Hr"])
    out.insert(0, "timestamp", ts)
    return out


def members(path: Path, pattern: re.Pattern):
    with zipfile.ZipFile(path) as z:
        for n in sorted(z.namelist()):
            m = pattern.search(n)
            if m:
                yield m.group(1), m.group(2), z.read(n).decode("utf-8", "replace")


def read_index(src, paths):
    rows = []
    for p, _ in paths:
        with zipfile.ZipFile(p) as z:
            rows += [(m.group(2), m.group(1)) for m in map(FLOW.search, z.namelist()) if m]
    return pd.DataFrame(rows, columns=["series", "huc"]).drop_duplicates("series")


def read_files(src, files):
    parts = []
    for p, _ in files:
        for huc, g, text in members(Path(p), FLOW):
            df = parse_flow(text)
            df.insert(0, "series_id", g)
            df["huc"] = huc
            parts.append(df)
    return pd.concat(parts, ignore_index=True)


def read_forcing(path: Path, gauges: set[str] | None = None) -> pd.DataFrame:
    """Long table (gauge, timestamp, prcp, tmax, tmin, srad, ...) for the covariate task."""
    parts = []
    for _, g, text in members(Path(path), FORCING):
        if gauges is None or g in gauges:
            df = parse_forcing(text)
            df.insert(0, "gauge", g)
            parts.append(df)
    out = pd.concat(parts, ignore_index=True)
    out.columns = [re.sub(r"\(.*\)$", "", c) for c in out.columns]
    return out.replace(-999.0, np.nan)
