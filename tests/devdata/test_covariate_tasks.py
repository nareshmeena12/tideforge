"""Covariate-task helpers and two joins on fixtures: nearest stations, population weights, weighted mean with missing
cities, holiday flags on the local date, CAMELS forcing and calendar tasks end to end."""
import io
import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from tideforge.devdata import covariate_tasks as T
from tideforge.devdata.schema import load_sources


def test_nearest_and_weighted_mean():
    st = pd.DataFrame({"id": ["a", "b", "c"], "lat": [40.0, 40.1, 45.0], "lon": [-75.0, -75.0, -75.0]})
    n = T.nearest(st, 40.09, -75.0, 2, 50)
    assert n["id"].tolist() == ["b", "a"] and n["km"].iloc[0] < 2
    assert T.nearest(st, 0.0, 0.0, 1, 50).empty
    idx = pd.date_range("2024", periods=3, freq="h")
    f = pd.DataFrame({"x": [10.0, np.nan, np.nan], "y": [20.0, 20.0, np.nan]}, index=idx)
    m = T.weighted_mean(f, pd.Series({"x": 0.75, "y": 0.25}), 0.2)
    assert m.iloc[0] == pytest.approx(12.5) and m.iloc[1] == 20.0 and np.isnan(m.iloc[2])
    assert np.isnan(T.weighted_mean(f, pd.Series({"x": 0.75, "y": 0.25}), 0.5).iloc[1])


def test_holiday_flags_use_the_local_date():
    ts = pd.DatetimeIndex(["2024-07-13 21:00", "2024-07-13 23:00", "2024-07-14 22:30"])     # UTC
    assert T.holiday_flags(ts, "Europe/Paris", "FR").tolist() == [0.0, 1.0, 0.0]
    assert T.holiday_flags(pd.DatetimeIndex(["2024-07-04"]), None, "US", "IA").tolist() == [1.0]


def test_resolve_cities(tmp_path):
    gz = tmp_path / "gaz.txt"
    gz.write_text("USPS\tGEOID\tNAME\tINTPTLAT\tINTPTLONG\nPA\t1\tPhiladelphia city\t40.0\t-75.1\n"
                  "PA\t2\tPhiladelphia Heights CDP\t41\t-76\nTN\t3\tNashville-Davidson metropolitan government (balance)\t36.1\t-86.7\n")
    pe = tmp_path / "pop.csv"
    pe.write_text("SUMLEV,STNAME,NAME,POPESTIMATE2022,POPESTIMATE2023\n162,Pennsylvania,Philadelphia city,1,1550000\n"
                  "162,Tennessee,Nashville-Davidson metropolitan government (balance),1,690000\n050,Pennsylvania,Philadelphia County,1,1\n")
    c = pd.DataFrame({"ba": ["X", "Y"], "city": ["Philadelphia", "Nashville"], "state": ["PA", "TN"]})
    out = T.resolve_cities(c, gz, pe)
    assert out["pop"].tolist() == [1550000.0, 690000.0] and out["weight"].tolist() == [1.0, 1.0]
    with pytest.raises(KeyError):
        T.resolve_cities(pd.DataFrame({"ba": ["X"], "city": ["Nowhere"], "state": ["PA"]}), gz, pe)


def test_camels_and_calendar_tasks(tmp_path):
    srcs, _ = load_sources()
    from tideforge.devdata import fetch as F
    raw, staging = tmp_path / "raw", tmp_path / "staging"
    z = raw / "camels_us" / "files" / F.local_name(srcs["D071"].fetch["urls"][0])
    z.parent.mkdir(parents=True)
    head = "45.1\n250\n2252.7\nYear Mnth Day Hr dayl(s) prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)\n"
    with zipfile.ZipFile(z, "w") as zz:
        zz.writestr("x/basin_mean_forcing/daymet/01/01013500_lump_cida_forcing_leap.txt",
                    head + "1980 1 1 12 3 1.5 150 0 -6.5 -16.3 171\n1980 1 2 12 3 0.0 140 0 -5.0 -15.0 170\n")
    d = staging / "camels_us"
    d.mkdir(parents=True)
    pd.DataFrame({"series_id": "01013500", "timestamp": pd.to_datetime(["1980-01-01", "1980-01-02"]),
                  "value": [10.0, 11.0]}).to_parquet(d / "D.parquet")
    tasks = T.load_tasks()
    m = T.build_task("camels_flow_forcing", tasks["camels_flow_forcing"], raw, staging, srcs)
    out = pd.read_parquet(staging / "covtasks" / "camels_flow_forcing" / "D.parquet")
    assert out["prcp"].tolist() == [1.5, 0.0] and out["target"].tolist() == [10.0, 11.0]
    assert m["covariates"]["tmax"]["role"] == "known" and m["covariates"]["tmax"]["oracle"] is True
    assert m["split_unit"] == "caravan_camels"
    d = staging / "iowa_liquor"
    d.mkdir(parents=True)
    pd.DataFrame({"series_id": "POLK", "timestamp": pd.to_datetime(["2024-07-03", "2024-07-04"]), "value": [1.0, 2.0]}
                 ).to_parquet(d / "D.parquet")
    m = T.build_task("iowa_liquor_holidays", tasks["iowa_liquor_holidays"], raw, staging, srcs)
    out = pd.read_parquet(staging / "covtasks" / "iowa_liquor_holidays" / "D.parquet")
    assert out["holiday"].tolist() == [0.0, 1.0] and m["covariates"]["holiday"]["oracle"] is False


def test_same_source_task(tmp_path):
    srcs, _ = load_sources()
    from tideforge.devdata import fetch as F
    raw, staging = tmp_path / "raw", tmp_path / "staging"
    src = srcs["D295"]
    url = F.file_list(src.fetch, ["3304557"])[0][0]
    f = raw / src.slug / "files" / F.local_name(url)
    f.parent.mkdir(parents=True)
    f.write_text("data_iniSE,casos,tempmin,umidmax,umidmed,umidmin,tempmed,tempmax\n"
                 "2025-12-28,215,25.4,84.8,69.0,51.6,28.6,32.8\n2025-12-21,144,23.2,87.5,65.5,42.3,28.0,33.3\n")
    d = staging / src.slug
    d.mkdir(parents=True)
    w = pd.date_range("2025-12-21", periods=2, freq="W")
    pd.DataFrame({"series_id": "3304557", "timestamp": w, "value": [144.0, 215.0]}).to_parquet(d / "W.parquet")
    m = T.build_task("infodengue_weather", T.load_tasks()["infodengue_weather"], raw, staging, srcs)
    out = pd.read_parquet(staging / "covtasks" / "infodengue_weather" / "W.parquet")
    assert out["tempmin"].tolist() == [23.2, 25.4] and m["covariates"]["umidmed"]["oracle"] is True
    task = dict(T.load_tasks()["infodengue_weather"], covariates={"tempmin": {"role": "known", "oracle": True},
                                                                  "holiday": {"country": "BR", "role": "known", "oracle": False}})
    m2 = T.build_task("infodengue_holidays", task, raw, staging, srcs)
    out2 = pd.read_parquet(staging / "covtasks" / "infodengue_holidays" / "W.parquet")
    assert "holiday" in out2 and out2["tempmin"].tolist() == [23.2, 25.4] and set(m2["covariates"]) == {"tempmin", "holiday"}


def test_batch2_tasks_are_well_formed():
    srcs, b = load_sources()
    tasks = T.load_tasks()
    assert len(tasks) == 25
    for name, t in tasks.items():
        assert t["target"]["source"] in srcs, name
        for c, v in t["covariates"].items():
            assert v["role"] in ("known", "past"), (name, c)
            if c == "temperature":
                assert v["role"] == "known" and v["oracle"] is True, name        # weather: known + oracle
            if c == "holiday":
                assert v["oracle"] is False and v["country"], name
