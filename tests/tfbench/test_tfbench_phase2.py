"""Phase 2 (owner 2026-11-13): calendar time (MMWR / ISO weeks, yyyymm, year + month, financial year, year-week text),
hour column + date in the path, numeric epoch time lists in nested rows (the two plan TypeErrors), csv headers with empty
or repeated names, netCDF in a tar, the files / table_cols overrides, zarr exclusion, 404 list in the report."""
from __future__ import annotations

import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tideforge.tfbench import readers
from tideforge.tfbench.canon import Fail, build_dataset, plan_dataset
from tideforge.tfbench.config import Settings
from tideforge.tfbench.layouts import mmwr_week_end, iso_week_start

S = Settings(min_context=20, max_series=1000, max_points=50_000)


def row(**kw):
    return {"dataset_id": "ds", "repo_id": "o/r", "config_or_subset": "", "catalog_row": 1, "status": "keep",
            "domain": "health", "licence_hub": "cc-by-4.0", "catalog_licence": "CC-BY", **kw}


def files_of(pairs):
    return [(Path(f), rel, Path(f).stat().st_size) for f, rel in pairs]


def build(tmp_path, pairs, ov=None, **kw):
    return build_dataset(row(**kw), files_of(pairs), ov or {}, S, tmp_path / "out" / "ds", tmp_path / "work")


def data(tmp_path):
    return pd.read_parquet(tmp_path / "out" / "ds" / "data.parquet")


def walk(n, seed=0):
    return np.cumsum(np.random.default_rng(seed).normal(size=n)) + 50


# ---- calendar ------------------------------------------------------------------------------------------------------------
def test_mmwr_week_ends_on_saturday_per_cdc():
    # CDC: MMWR week 1 of 2020 ended 2020-01-04, of 2021 ended 2021-01-09, of 2023 ended 2023-01-07; 2020 has week 53
    got = mmwr_week_end([2020, 2021, 2023, 2020], [1, 1, 1, 53])
    assert list(got.dt.strftime("%Y-%m-%d")) == ["2020-01-04", "2021-01-09", "2023-01-07", "2021-01-02"]
    assert (got.dt.dayofweek == 5).all()
    assert iso_week_start([2021], [1]).iloc[0] == pd.Timestamp("2021-01-04")


def test_nndss_like_mmwr_year_and_week_give_real_weeks(tmp_path):
    f = tmp_path / "nndss.csv"
    weeks = [(y, w) for y in (2021, 2022) for w in range(1, 53)]
    d = pd.DataFrame([{"Reporting Area": a, "Current MMWR Year": y, "MMWR WEEK": w, "Label": "Measles",
                       "Current week": float(abs(hash((a, y, w))) % 50), "Cumulative YTD Current MMWR Year": 1000.0}
                      for a in ("OHIO", "UTAH") for y, w in weeks])
    d.to_csv(f, index=False)
    m = build(tmp_path, [(f, "nndss.csv")], {"calendar": {"year": "Current MMWR Year", "week": "MMWR WEEK",
                                                         "week_rule": "mmwr"},
                                            "id_cols": ["Reporting Area", "Label"], "targets": ["Current week"],
                                            "covariates": []})
    assert m["synthetic_time"] is False and m["frequency"] == "W-SAT"
    x = data(tmp_path)
    assert x["timestamp"].min() == pd.Timestamp("2021-01-09") and (x["timestamp"].dt.dayofweek == 5).all()
    assert m["n_series"] == 2


def test_yyyymm_column_found_without_overrides(tmp_path):
    f = tmp_path / "bike.csv"
    months = pd.date_range("2015-01-01", periods=72, freq="MS")
    pd.DataFrame({"rent_month": np.tile(months.strftime("%Y%m").astype(int), 2), "station_name": np.repeat(["a", "b"], 72),
                  "use_count": np.r_[walk(72, 1), walk(72, 2)]}).to_csv(f, index=False)
    m = build(tmp_path, [(f, "bike.csv")])
    assert m["synthetic_time"] is False and m["frequency"] == "MS"
    assert data(tmp_path)["timestamp"].min() == pd.Timestamp("2015-01-01")
    assert m["targets"] == ["use_count"]
    assert any(t["step"] == "time from calendar columns" for t in m["transformations"])


def test_year_and_month_named_columns_and_fiscal_year(tmp_path):
    f = tmp_path / "fipe.parquet"
    ym = pd.date_range("2012-01-01", periods=60, freq="MS")
    pq.write_table(pa.table({"ano_referencia": np.tile(ym.year, 2), "mes_referencia": np.tile(ym.month, 2).astype(float),
                             "ano_modelo": np.repeat([2001, 2002], 60), "codigo_fipe": np.repeat(["x", "y"], 60),
                             "valor_centavos": np.r_[walk(60, 3), walk(60, 4)]}), f)
    m = build(tmp_path, [(f, "fipe.parquet")], {"calendar": {"year": "ano_referencia", "month": "mes_referencia"},
                                                 "id_cols": ["codigo_fipe", "ano_modelo"], "targets": ["valor_centavos"],
                                                 "covariates": []})
    assert m["frequency"] == "MS" and data(tmp_path)["timestamp"].max() == pd.Timestamp("2016-12-01")

    g = tmp_path / "fy.csv"
    fy = [f"{y}-{(y + 1) % 100:02d}" for y in range(1960, 2020)]
    pd.DataFrame({"financial_year": fy * 2, "station_name": ["A"] * 60 + ["B"] * 60,
                  "entries_exits": np.r_[walk(60, 5), walk(60, 6)]}).to_csv(g, index=False)
    m2 = build_dataset(row(), files_of([(g, "fy.csv")]), {"id_cols": ["station_name"], "targets": ["entries_exits"],
                                                          "calendar": {"fiscal_year": "financial_year"}},
                       Settings(min_context={"yearly": 10, **{k: 20 for k in ("subhourly", "hourly", "subdaily", "daily",
                                                                               "weekly", "monthly", "quarterly")}}),
                       tmp_path / "o2" / "ds", tmp_path / "w2")
    assert m2["frequency"] == "YS"


def test_year_week_text_as_epi_week(tmp_path):
    f = tmp_path / "clima.csv"
    yw = [f"{y}/{w:02d}" for y in (2014, 2015) for w in range(1, 53)]
    pd.DataFrame({"codigo_ibge": 1100015, "ano_semana": yw, "T2M": walk(len(yw), 7)}).to_csv(f, index=False)
    m = build(tmp_path, [(f, "c.csv")], {"calendar": {"yearweek": "ano_semana", "week_rule": "mmwr"},
                                          "id_cols": ["codigo_ibge"], "targets": ["T2M"]})
    assert m["frequency"] == "W-SAT"
    assert data(tmp_path)["timestamp"].min() == pd.Timestamp("2014-01-04")


def test_calendar_override_names_a_missing_column(tmp_path):
    f = tmp_path / "a.csv"
    pd.DataFrame({"year": [2000] * 3, "v": [1.0, 2.0, 3.0]}).to_csv(f, index=False)
    with pytest.raises(Fail, match="calendar: columns not in the table"):
        build(tmp_path, [(f, "a.csv")], {"calendar": {"year": "year", "month": "mes"}})


def test_hour_column_named_time_plus_date_in_path(tmp_path):
    pairs = []
    for k, day in enumerate(pd.date_range("2022-01-01", periods=8, freq="D")):
        rel = f"hourly/{day:%Y-%m-%d}.csv"
        f = tmp_path / "raw" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"cbg": np.repeat(["c1", "c2"], 24), "time": np.tile(np.arange(24), 2),
                      "population": walk(48, k)}).to_csv(f, index=False)
        pairs.append((f, rel))
    m = build(tmp_path, pairs, {"time_col": "time", "id_cols": ["cbg"], "targets": ["population"]})
    assert m["synthetic_time"] is False and m["frequency"] == "h"
    assert data(tmp_path)["timestamp"].min() == pd.Timestamp("2022-01-01 00:00")


def test_real_time_column_is_not_replaced_by_calendar_columns(tmp_path):
    f = tmp_path / "t.csv"
    ts = pd.date_range("2020-01-01", periods=200, freq="h")
    pd.DataFrame({"date": ts.astype(str), "year": ts.year, "month": ts.month, "v": walk(200)}).to_csv(f, index=False)
    m = build(tmp_path, [(f, "t.csv")])
    assert m["frequency"] == "h" and m["time_col"] == "date"


# ---- nested epoch lists (the plan TypeErrors) -----------------------------------------------------------------------------
@pytest.mark.parametrize("unit,mult", [("s", 1), ("ms", 1000)])
def test_nested_numeric_time_list(tmp_path, unit, mult):
    f = tmp_path / "g.parquet"
    rows = []
    for k in range(3):
        t0 = 1_703_808_000 + 86400 * k
        rows.append({"dataset": "AZT1D", "subject_id": f"Subject {k}",
                     "timestamp": [float((t0 + 300 * i) * mult) for i in range(300)], "BGvalue": list(walk(300, k))})
    pq.write_table(pa.Table.from_pylist(rows), f)
    p = plan_dataset(row(), files_of([(f, "data/train-0.parquet")]), {"id_cols": ["dataset", "subject_id"]}, S,
                     tmp_path / "wp")
    assert p["status"] == "ok", p["reason"]
    assert p["time_list"] == "timestamp"
    m = build(tmp_path, [(f, "data/train-0.parquet")], {"id_cols": ["dataset", "subject_id"]})
    assert m["frequency"] == "5min" and m["n_series"] == 3 and not m["synthetic_time"]
    assert data(tmp_path)["item_id"].str.startswith("AZT1D|").all()
    assert data(tmp_path)["timestamp"].min() == pd.Timestamp("2023-12-29")


# ---- readers -------------------------------------------------------------------------------------------------------------
def test_unique_names():
    assert readers.unique_names(["a", "b"]) is None
    assert readers.unique_names(["", "a", "a", "a.1"]) == ["Unnamed: 0", "a", "a.2", "a.1"]


def test_csv_with_empty_and_repeated_header(tmp_path):
    f = tmp_path / "x.csv"
    f.write_text(",v,v\n2021-01-01 00:00,1,2\n2021-01-01 01:00,3,4\n", encoding="utf-8")
    p = readers.open_parts(f, tmp_path / "w")[0]
    assert p.schema.names == ["Unnamed: 0", "v", "v.1"]
    t = p.read()
    assert t.num_rows == 2 and t.column("v.1").to_pylist() == [2.0, 4.0]
    assert p.read(["v"]).column_names == ["v"]


def test_wide_csv_with_unnamed_time_column(tmp_path):
    f = tmp_path / "SZH" / "volume.csv"
    f.parent.mkdir()
    ts = pd.date_range("2023-04-01", periods=300, freq="h")
    pd.DataFrame({"": ts.astype(str), "0": walk(300, 1), "1": walk(300, 2)}).to_csv(f, index=False)
    m = build(tmp_path, [(f, "SZH/volume.csv")], {"time_col": "Unnamed: 0", "layout": "wide"})
    assert m["n_series"] == 2 and m["frequency"] == "h"


def test_zarr_zip_unreadable(tmp_path):
    f = tmp_path / "icon.zarr.zip"
    f.write_bytes(b"PK")
    with pytest.raises(readers.Unreadable, match="zarr"):
        readers.open_parts(f, tmp_path / "w")


def _caravan_nc(path: Path, gauge: str, seed: int):
    xr = pytest.importorskip("xarray")
    pytest.importorskip("netCDF4")
    t = pd.date_range("2000-01-01", periods=400, freq="D")
    ds = xr.Dataset({"streamflow": ("date", walk(400, seed)), "total_precipitation_sum": ("date", walk(400, seed + 9)),
                     "grid": (("date", "x"), np.zeros((400, 2)))}, coords={"date": t})
    ds.attrs["gauge"] = gauge
    ds.to_netcdf(path, engine="netcdf4")


def test_netcdf_in_tar_one_series_per_catchment_and_files_override(tmp_path):
    src = tmp_path / "src"
    (src / "caravan").mkdir(parents=True)
    for k, g in enumerate(["camels_01", "camels_02"]):
        _caravan_nc(src / "caravan" / f"{g}.nc", g, k)
    tar_keep = tmp_path / "timeseries-camels.tar"
    with tarfile.open(tar_keep, "w") as tf:
        tf.add(src / "caravan", arcname="caravan")
    tar_skip = tmp_path / "timeseries-hysets-0.tar"
    tar_skip.write_bytes(b"not a tar")                      # would fail if opened: the files override keeps it closed
    p = readers.open_parts(tar_keep, tmp_path / "w0", "timeseries-camels.tar")
    assert [q.schema.names for q in p][0] == ["date", "streamflow", "total_precipitation_sum"]
    m = build(tmp_path, [(tar_keep, "timeseries-camels.tar"), (tar_skip, "timeseries-hysets-0.tar")],
              {"files": r"timeseries-camels(gb|aus|cl)?\.tar$", "targets": ["streamflow"],
               "covariates": ["total_precipitation_sum", "not_there"]})
    assert m["n_series"] == 2 and m["frequency"] == "D"
    assert list(m["covariates"]) == ["total_precipitation_sum"]
    assert any(t["step"] == "files (overrides.yaml)" and t["kept"] == 1 for t in m["transformations"])


def test_files_override_matching_nothing_fails(tmp_path):
    f = tmp_path / "a.csv"
    f.write_text("date,v\n2020-01-01,1\n", encoding="utf-8")
    with pytest.raises(Fail, match="files: no file matches"):
        build(tmp_path, [(f, "a.csv")], {"files": "nothing"})


def test_table_cols_override(tmp_path):
    a = tmp_path / "meta.csv"
    pd.DataFrame({"gauge_id": ["1"], "lat": [1.0]}).to_csv(a, index=False)
    b = tmp_path / "q.csv"
    pd.DataFrame({"datetime": pd.date_range("2020-01-01", periods=300, freq="h").astype(str),
                  "discharge_m3s": walk(300)}).to_csv(b, index=False)
    m = build(tmp_path, [(a, "d/a.csv"), (b, "d/q.csv")], {"table_cols": ["datetime", "discharge_m3s"],
                                                           "time_col": "datetime", "targets": ["discharge_m3s"]})
    assert m["n_series"] == 1


# ---- report --------------------------------------------------------------------------------------------------------------
def test_report_lists_404_files(tmp_path):
    from tideforge.tfbench.report import removed_files
    md = tmp_path / "manifest"
    md.mkdir()
    lines = [{"repo": "o/r", "path": "a.csv", "status": "error: RemoteEntryNotFoundError: 404 Client Error"},
             {"repo": "o/r", "path": "b.csv", "status": "error: HTTPError: 404"},
             {"repo": "o/r", "path": "b.csv", "status": "ok"},
             {"repo": "o/r", "path": "c.csv", "status": "error: Timeout"}]
    (md / "o_r.jsonl").write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    g = removed_files(md)
    assert list(g["path"]) == ["a.csv"]
