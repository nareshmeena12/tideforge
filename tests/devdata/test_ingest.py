"""Ingest pipeline on local fixtures (file:// URLs, no network): schema checks, fetch log and redaction, table loader options,
adapters, frozen stratified selection, regularise / clean rules, end-to-end output + meta.json."""
import gzip
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tideforge.devdata import fetch as F
from tideforge.devdata import freqs, ingest, schema
from tideforge.devdata.loaders import tabular
from tideforge.devdata.loaders.adapters import ausgrid, camels_us, eurostat_tsv


# ---------------------------------------------------------------------------------------------------- schema
def test_repo_sources_validate():
    assert schema.validate_all() == []


def _src(**kw):
    d = dict(slug="x", country="US", fetch={"urls": ["file:///x"]}, read={"format": "csv"}, time={"col": "t"}, id=["k"], value="v",
             tasks=[{"freq": "h"}])
    d.update(kw)
    return schema.from_dict(kw.pop("_id", "D055"), d)


def test_schema_catches_problems():
    assert any("mixed" in p for p in schema.validate(_src()))                       # D055 is a mixed catalog row
    assert schema.validate(_src(exclude={"rule": "drop TIME's sub-series"})) == []
    bad = _src(exclude={"rule": "r"}, fetch={"urls": ["u"], "headers": {"X-Api-Key": "abc123"}})
    assert any("looks like a secret" in p for p in schema.validate(bad))
    assert any("environment variable" in p for p in schema.validate(_src(exclude={"rule": "r"},
                                                                        fetch={"urls": ["u"], "auth": {"env": "lower"}})))
    agg = schema.from_dict("D002", dict(slug="e", fetch={"urls": ["u"]}, read={"format": "csv"}, time={"col": "t"}, id=["k"],
                                        value="v", tasks=[{"freq": "h"}]))
    assert any("aggregator rules" in p for p in schema.validate(agg))
    with pytest.raises(KeyError):
        schema.from_dict("D055", {"slug": "x", "nope": 1})
    assert any("country is required" in p for p in schema.validate(_src(exclude={"rule": "r"}, country="")))
    assert any("not an ISO" in p for p in schema.validate(_src(exclude={"rule": "r"}, country="usa")))


def test_series_country():
    from tideforge.devdata.country import series_country
    fips = {"prefix": 2, "codes": "fips"}
    assert series_country(fips, "UKM00003772") == "GB" and series_country(fips, "KSM00047108") == "KR"
    assert series_country(fips, "ASN00066062") == "AU" and series_country(fips, "XXM0001") == "multi"
    assert series_country({"prefix": 2, "codes": "iso2"}, "EL30") == "GR" and series_country("multi", "a") == "multi"


# ---------------------------------------------------------------------------------------------------- fetch
def test_expand_render_redact():
    v = F.expand_vars({"year": {"range": [2020, 2021]}, "s": "index", "m": {"months": ["2024-11", "2025-01"], "fmt": "%Y%m"}},
                      ["a b", "c"])
    assert len(v) == 2 * 2 * 3 and v[0] == {"m": "202411", "s": "a b", "year": 2020}
    assert F.render("https://x/{s}/{year}", {"s": "a/b c", "year": 1}) == "https://x/a%2Fb%20c/1"
    assert F.redact("https://x?a=1&api_key=SECRET&b=2", {"env": "K", "param": "api_key"}) == \
        "https://x?a=1&api_key=<redacted>&b=2"


def test_download_logs_sha_and_never_the_key(tmp_path, monkeypatch):
    src = tmp_path / "data.csv"
    src.write_bytes(b"t,k,v\n")
    log = tmp_path / "log.jsonl"
    url = src.as_uri()
    monkeypatch.setenv("TF_TEST_KEY", "SECRET123")
    row = F.download(url, tmp_path / "out" / "f.csv", log=log)
    assert row["status"] == "ok" and row["sha256"] == __import__("hashlib").sha256(b"t,k,v\n").hexdigest()
    assert F.download(url, tmp_path / "out" / "f.csv", log=log)["status"] == "exists"
    assert "SECRET123" not in F.redact(F._with_auth("https://h/x?a=1", {}, {"env": "TF_TEST_KEY", "param": "k"})[0],
                                       {"env": "TF_TEST_KEY", "param": "k"})
    assert len(log.read_text().splitlines()) == 1


# ---------------------------------------------------------------------------------------------------- table loader
def test_table_loader_options(tmp_path):
    csv = "station,date,element,value,qflag\nA1,20200101,TMAX,105,\nA1,20200102,TMAX,-9999,\nA1,20200103,TMAX,99,X\n" \
          "A1,20200101,PRCP,5,\nB2,20200101,TMAX,7,\n"
    p = tmp_path / "f.csv.gz"
    p.write_bytes(gzip.compress(csv.encode()))
    s = _src(read={"format": "csv", "dtype": {"qflag": str}}, time={"col": "date", "format": "%Y%m%d"}, id=["station", "element"],
             value="value", filter={"element": ["TMAX"], "qflag": [None]}, missing=[-9999], scale=0.1,
             exclude={"where": "station != 'B2'", "rule": "r"})
    out = tabular.read_files(s, [(p, {})])
    assert out["series_id"].unique().tolist() == ["A1/TMAX"]
    assert out["value"].tolist()[0] == pytest.approx(10.5) and np.isnan(out["value"].tolist()[1]) and len(out) == 2


def test_zip_members_parts_time_tz_and_file_vars(tmp_path):
    p = tmp_path / "f.zip"
    with zipfile.ZipFile(p, "w") as z:
        z.writestr("a.csv", "YY,MM,DD,hh,X,Y\n2024,1,1,0,1,2\n2024,7,1,12,3,4\n")
        z.writestr("readme.pdf", "x")
    s = _src(read={"format": "csv"}, time={"parts": {"year": "YY", "month": "MM", "day": "DD", "hour": "hh"}},
             id=["_var_series"], value=["X", "Y"], tz="America/New_York", exclude={"rule": "r"})
    out = tabular.read_files(s, [(p, {"series": "S9"})])
    assert sorted(out["series_id"].unique()) == ["S9/X", "S9/Y"]
    assert out["timestamp"].min() == pd.Timestamp("2024-01-01 05:00") and pd.Timestamp("2024-07-01 16:00") in set(out["timestamp"])


def test_json_and_melt(tmp_path):
    j = tmp_path / "f.json"
    j.write_text(json.dumps({"items": [{"article": "A", "timestamp": "2024010100", "views": 3}]}))
    s = _src(read={"format": "json", "record_path": ["items"]}, time={"col": "timestamp", "format": "%Y%m%d%H"},
             id=["article"], value="views", exclude={"rule": "r"})
    assert tabular.read_files(s, [(j, {})])["value"].tolist() == [3.0]
    w = tmp_path / "w.csv"
    w.write_text("time,f1,f2\n0,1,2\n60,3,4\n")
    s = _src(read={"format": "csv", "melt": {"id_vars": ["time"], "var_name": "function"}}, time={"col": "time", "unit": "s"},
             id=["function"], value="value", exclude={"rule": "r"})
    out = tabular.read_files(s, [(w, {})])
    assert sorted(out["series_id"].unique()) == ["f1", "f2"] and out["value"].sum() == 10


# ---------------------------------------------------------------------------------------------------- adapters
def test_ausgrid_wide_to_long():
    slots = [f"{h}:{m:02d}" for h in range(24) for m in (0, 30)][1:] + ["0:00"]
    row = {"Customer": 1, "Generator Capacity": 1.0, "Postcode": 2000, "Consumption Category": "GC", "date": "1/07/2012",
           **{s: i for i, s in enumerate(slots)}, "Row Quality": ""}
    out = ausgrid.wide_to_long(pd.DataFrame([row]), "UTC")
    assert len(out) == 48 and out["series_id"].iloc[0] == "1/GC"
    assert out["timestamp"].iloc[0] == pd.Timestamp("2012-07-01 00:00") and out["value"].iloc[47] == 47


def test_eurostat_tsv():
    text = "freq,sex,unit,geo\\TIME_PERIOD\t2024-W01 \t2024-W02 \nW,T,NR,AT111\t12 \t: \nW,F,NR,AT111\t5 \t6 p\n"
    out = eurostat_tsv.parse(text, {"sex": "T"})
    assert out["series_id"].tolist() == ["AT111", "AT111"] and out["country"].iloc[0] == "AT"
    assert out["timestamp"].iloc[0] == pd.Timestamp("2024-01-01") and out["value"].iloc[0] == 12 and np.isnan(out["value"].iloc[1])


def test_camels_parsers():
    f = camels_us.parse_flow("01013500 1980 01 01 655.00 A\n01013500 1980 01 02 -999.00 M\n")
    assert f["value"].iloc[0] == 655.0 and np.isnan(f["value"].iloc[1])
    text = "45.1\n250\n2252.7\nYear Mnth Day Hr dayl(s) prcp(mm/day) srad(W/m2) swe(mm) tmax(C) tmin(C) vp(Pa)\n" \
           "1980 1 1 12 30172.51 0.00 153.40 0.00 -6.54 -16.30 171.69\n"
    g = camels_us.parse_forcing(text)
    assert g["tmax(C)"].iloc[0] == -6.54 and g["timestamp"].iloc[0] == pd.Timestamp("1980-01-01")


# ---------------------------------------------------------------------------------------------------- selection and cleaning
def test_select_is_frozen_and_stratified():
    df = pd.DataFrame({"series": [f"s{i}" for i in range(100)], "c": ["a"] * 80 + ["b"] * 20})
    a = ingest.select(df, "D001", 10, ["c"])
    assert a == ingest.select(df.sample(frac=1, random_state=1), "D001", 10, ["c"])
    assert sum(x in set(df[df.c == "b"]["series"]) for x in a) == 2
    assert ingest.select(df, "D001", 10, ["c"]) != ingest.select(df, "D002", 10, ["c"])
    assert ingest.quotas({"a": 1, "b": 50}, 10) == {"a": 0, "b": 10}           # proportional, no minimum per stratum
    assert ingest.quotas({"a": 2, "b": 3}, 10) == {"a": 2, "b": 3}


def test_regularise_and_clean():
    t = pd.to_datetime(["2024-01-01 00:10", "2024-01-01 00:20", "2024-01-01 03:05"])
    g = pd.DataFrame({"timestamp": t, "value": [1.0, 3.0, 5.0]})
    m = ingest.regularise(g, "h", "mean", False)
    assert m.iloc[0] == 2.0 and np.isnan(m.iloc[1]) and m.iloc[3] == 5.0
    c = ingest.regularise(g, "h", "count", True)
    assert c.tolist() == [2.0, 0.0, 0.0, 1.0]
    s = pd.Series(np.r_[np.nan, np.arange(100.0), np.nan], index=pd.date_range("2024", periods=102, freq="h"))
    kept, why = ingest.clean(s, 50)
    assert why == "kept" and len(kept) == 100
    assert ingest.clean(s, 200)[1] == "short"
    assert ingest.clean(pd.Series(np.ones(100)), 10)[1] == "constant"
    assert ingest.clean(pd.Series(np.r_[1.0, np.full(60, np.nan), 2.0, 3.0]), 10)[1] == "missing"


# ---------------------------------------------------------------------------------------------------- end to end
def test_fetch_and_ingest_end_to_end(tmp_path, horizons):
    raw_src = tmp_path / "remote"
    raw_src.mkdir()
    idx = raw_src / "index.csv"
    idx.write_text("id,region\n" + "".join(f"S{i},{'N' if i % 2 else 'S'}\n" for i in range(12)))
    rng = np.random.default_rng(0)
    t = pd.date_range("2024-01-01", periods=24 * 30, freq="h")
    for i in range(12):
        n = len(t) if i != 3 else 40                                  # S3 is too short
        v = rng.normal(size=n).cumsum() if i != 5 else np.ones(n)     # S5 is constant
        pd.DataFrame({"t": t[:n].strftime("%Y-%m-%d %H:%M"), "v": v}).to_csv(raw_src / f"S{i}.csv", index=False)
    s = schema.from_dict("D057", dict(
        slug="fixture", index={"url": idx.as_uri(), "read": {"format": "csv"}, "id": "id", "strata": ["region"]},
        fetch={"url_template": (raw_src.as_uri() + "/{series}.csv"), "vars": {"series": "index"}, "rate_limit_s": 0},
        read={"format": "csv"}, time={"col": "t", "format": "%Y-%m-%d %H:%M"}, id=["_var_series"], value="v",
        tasks=[{"freq": "h", "how": "mean"}, {"freq": "D", "how": "mean"}], select={"max_series": 8, "oversample": 1.5}))
    run = ingest.Run(s, tmp_path / "raw", tmp_path / "out", say=lambda *a: None)
    files = run.fetch()
    assert len(files) == 12
    meta = run.ingest()
    h = pd.read_parquet(tmp_path / "out" / "fixture" / "h.parquet")
    assert h["series_id"].nunique() == 8 and not {"S3", "S5"} & set(h["series_id"])
    assert meta["tasks"]["h"]["dropped"]["short"] == 1 and meta["tasks"]["h"]["dropped"]["constant"] == 1
    assert meta["tasks"]["D"]["n_series"] == 0 and meta["tasks"]["D"]["dropped"]["short"] == 12    # 30 days < 3*30+64
    assert all(r["sha256"] for r in meta["raw_files"]) and meta["dataset_id"] == "D057"
