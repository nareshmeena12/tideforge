"""Fast time parsing (owner 2026-11-13: per-element dateutil stalled canon for hours) and years before 1678."""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from tideforge.tfbench.layouts import numeric_time
from tideforge.tfbench.readers import open_parts
from tideforge.tfbench.timefreq import guess_format, parse_time


def test_formats_and_equivalence():
    cases = {
        "2021-03-04": "2021-03-04", "2021-03-04 05:06:07": "2021-03-04 05:06:07",
        "2021-03-04T05:06:07+02:00": "2021-03-04 03:06:07",
    }
    s = pd.Series(list(cases) + [None, "garbage"])
    out = parse_time(s)
    assert list(out[:3]) == [pd.Timestamp(v) for v in cases.values()]
    assert out[3:].isna().all()
    assert guess_format(pd.Series(["13/02/2020", "25/12/2021"], dtype="string")) == "%d/%m/%Y"
    assert parse_time(pd.Series(["13/02/2020", "01/02/2020"]))[0] == pd.Timestamp("2020-02-13")
    assert parse_time(pd.Series(["03/04/2020", "05/06/2021"]))[0] == pd.Timestamp("2020-03-04")   # month first, as before
    assert parse_time(pd.Series(["Mar 04, 2020", "Dec 31, 2021"]))[1] == pd.Timestamp("2021-12-31")


def test_large_dmy_column_is_fast_and_silent():
    days = pd.date_range("2001-01-01", periods=9000, freq="D")
    s = pd.Series(np.tile(days.strftime("%d/%m/%Y"), 200))            # 1.8M rows, 9k distinct values
    t0 = time.monotonic()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)                   # no "could not infer format" fallback
        out = parse_time(s)
    assert time.monotonic() - t0 < 20
    assert (out[:9000].to_numpy() == days.to_numpy()).all()


def test_numeric_years():
    out = parse_time(pd.Series([1990, 1991, None], dtype="float64"))
    assert list(out[:2]) == [pd.Timestamp("1990-01-01"), pd.Timestamp("1991-01-01")] and pd.isna(out[2])


def test_years_before_1678_get_a_synthetic_yearly_clock(tmp_path):
    f = tmp_path / "rings.parquet"
    pq.write_table(pa.table({"year": np.arange(1200, 2001, dtype="int64"), "value": np.arange(801, dtype="float64")}), f)
    parts = open_parts(f, tmp_path / "w", "rings.parquet")
    out, info = numeric_time(parts, "year")
    assert info["synthetic_time"] and info["freq"] == "YS" and info["year_shift"] == 260
    t = pd.concat([b.to_pandas() for b in out[0].batches(["_time", "year"])])
    assert t.loc[t.year == 2000, "_time"].iloc[0] == pd.Timestamp("2260-01-01")
    assert t.loc[t.year == 1418, "_time"].iloc[0] == pd.Timestamp("1678-01-01")
    assert t.loc[t.year < 1418, "_time"].isna().all()
