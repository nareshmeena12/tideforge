"""The batch-2 table specs written from the 2026-10-31 probe (logs/perlmutter/batch2_probe.txt), run on the rows the probe
showed (or the documented layout, where the probe only reached a zip): the parsed series ids, timestamps (UTC) and values."""
import zipfile

import pandas as pd

from tideforge.devdata import fetch as F
from tideforge.devdata import loaders, schema

SRCS, _ = schema.load_sources()


def read(i, text: str | bytes, tmp_path, name=None, vars_=None):
    src = SRCS[i]
    url = F.file_list(src.fetch)[0][0]
    p = tmp_path / (name or F.local_name(url))
    p.write_bytes(text if isinstance(text, bytes) else text.encode("utf-8"))
    return loaders.read_files(src, [(p, vars_ or {})])


def test_dsny(tmp_path):
    df = read("D252", "MONTH,BOROUGH,COMMUNITYDISTRICT,REFUSETONSCOLLECTED,PAPERTONSCOLLECTED,MGPTONSCOLLECTED,RESORGANICSTONS\n"
                      "2026 / 09,Bronx,01,3517.0,261.4,140.8,25.8\n2026 / 08,Bronx,01,3600.0,250.0,150.0,20.0\n", tmp_path)
    r = df[df["series_id"] == "Bronx/01/REFUSETONSCOLLECTED"].sort_values("timestamp")
    assert r["value"].tolist() == [3600.0, 3517.0] and r["timestamp"].iloc[-1] == pd.Timestamp("2026-09-01")


def test_zillow_wide(tmp_path):
    df = read("D162", 'RegionID,SizeRank,RegionName,RegionType,StateName,2000-01-31,2000-02-29\n'
                      '102001,0,United States,country,,122625.7,122841.5\n394913,1,"New York, NY",msa,NY,217974.2,218900.5\n',
              tmp_path)
    r = df[df["series_id"] == "394913"].sort_values("timestamp")
    assert r["value"].tolist() == [217974.2, 218900.5] and r["timestamp"].tolist() == [pd.Timestamp("2000-01-31"),
                                                                                         pd.Timestamp("2000-02-29")]


def test_stats19_events(tmp_path):
    df = read("D345", "collision_index,collision_year,police_force,collision_severity,date,day_of_week\n"
                      "1979219300228,1979,21,3,31/05/1979,5\n1979219300229,1979,21,2,31/05/1979,5\n"
                      "1979440JH0331,1979,44,3,02/08/1979,5\n", tmp_path)
    assert sorted(df["series_id"].unique()) == ["21", "44"] and (df["value"] == 1.0).all()
    assert df["timestamp"].min() == pd.Timestamp("1979-05-31")


def test_melbourne_pedestrians_local_hours(tmp_path):
    df = read("D383", "id,location_id,sensing_date,hourday,direction_1,direction_2,pedestriancount,sensor_name\n"
                      "702320250829,70,2025-08-29,23,19,17,36,Errol20_T\n702320250830,70,2025-08-30,0,1,2,3,Errol20_T\n",
              tmp_path)
    r = df.sort_values("timestamp")
    assert r["series_id"].tolist() == ["70", "70"] and r["value"].tolist() == [36.0, 3.0]
    assert r["timestamp"].tolist() == [pd.Timestamp("2025-08-29 13:00"), pd.Timestamp("2025-08-29 14:00")]   # AEST = UTC+10


def test_nyc_311_and_sf_fire(tmp_path):
    df = read("D340", "day,complaint_type,n\n2021-09-21T00:00:00.000,Abandoned Bike,3\n2021-09-22T00:00:00.000,Abandoned Bike,3\n",
              tmp_path)
    assert df["series_id"].unique().tolist() == ["Abandoned Bike"] and df["timestamp"].iloc[0] == pd.Timestamp("2021-09-21 04:00")
    df = read("D343", "day,call_type,battalion,n\n2001-04-15T00:00:00.000,Administrative,B01,1\n", tmp_path)
    assert df["series_id"].tolist() == ["Administrative/B01"] and df["timestamp"].iloc[0] == pd.Timestamp("2001-04-15 07:00")


def test_seattle_library_parts(tmp_path):
    df = read("D334", "checkoutyear,checkoutmonth,materialtype,usageclass,n\n2005,4,ATLAS,Physical,26\n2005,5,ATLAS,Physical,21\n",
              tmp_path)
    assert df["series_id"].unique().tolist() == ["ATLAS/Physical"]
    assert df.sort_values("timestamp")["timestamp"].tolist() == [pd.Timestamp("2005-04-01"), pd.Timestamp("2005-05-01")]


def test_steel_zip(tmp_path):
    buf = tmp_path / "steel.zip"
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Steel_industry_data.csv", "date,Usage_kWh,CO2(tCO2),WeekStatus\n01/01/2018 00:15,3.17,0,Weekday\n"
                                              "01/01/2018 00:30,4.0,0,Weekday\n")
    df = read("D394", buf.read_bytes(), tmp_path)
    assert df["series_id"].unique().tolist() == ["daewoo"] and df["value"].tolist() == [3.17, 4.0]
    assert df["timestamp"].iloc[0] == pd.Timestamp("2017-12-31 15:15")                                    # KST = UTC+9


def test_santa_fe_index_clock(tmp_path):
    df = read("D279", "-0.5 1.2 95\n-0.4 1.1 96\n-0.3 1.0 97\n", tmp_path, name="b1.txt", vars_={"part": "b1"})
    r = df[df["series_id"] == "b1/heart_rate"]
    assert r["value"].tolist() == [-0.5, -0.4, -0.3]
    assert r["timestamp"].tolist() == list(pd.date_range("1970-01-01", periods=3, freq="s"))
