"""Generic layouts (owner 2026-11-12): time from paths, dates as column names, numeric time, index-only, event tables,
the metadata-path rule, the App_Flow zero-fill, the csv encoding fallback."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tideforge.tfbench import readers
from tideforge.tfbench.canon import Fail, build_dataset, plan_dataset
from tideforge.tfbench.config import Settings
from tideforge.tfbench.layouts import is_meta

S = Settings(min_context=20, max_series=1000, max_points=50_000)


def row(**kw):
    return {"dataset_id": "ds", "repo_id": "o/r", "config_or_subset": "", "catalog_row": 1, "status": "keep",
            "domain": "energy", "licence_hub": "cc-by-4.0", "catalog_licence": "CC-BY", **kw}


def build(tmp_path, files, ov=None, **kw):
    """files: (local path, repo-relative path)."""
    return build_dataset(row(**kw), [(Path(f), rel, Path(f).stat().st_size) for f, rel in files], ov or {}, S,
                         tmp_path / "out" / "ds", tmp_path / "work")


def data(tmp_path):
    return pd.read_parquet(tmp_path / "out" / "ds" / "data.parquet")


def walk(n, seed=0):
    return np.cumsum(np.random.default_rng(seed).normal(size=n)) + 50


def test_a_time_from_path_plus_hour_column(tmp_path):
    files = []
    for k, day in enumerate(pd.date_range("2021-03-01", periods=10, freq="D")):
        rel = f"data/{day:%Y/%m/%d}/counts.csv"
        f = tmp_path / "raw" / rel
        f.parent.mkdir(parents=True)
        pd.DataFrame({"region": np.repeat(["a", "b"], 24), "hour": np.tile(np.arange(24), 2),
                      "population": walk(48, k)}).to_csv(f, index=False)
        files.append((f, rel))
    m = build(tmp_path, files)
    d = data(tmp_path)
    assert m["frequency"] == "h" and m["n_series"] == 2
    assert d["timestamp"].min() == pd.Timestamp("2021-03-01") and d["timestamp"].max() == pd.Timestamp("2021-03-10 23:00")
    assert any(t["step"] == "time from the file path" for t in m["transformations"])


def test_b_years_as_column_names_are_melted(tmp_path):
    f = tmp_path / "hdr.csv"
    years = [str(y) for y in range(1960, 2020)]
    t = pd.DataFrame({"country": ["X", "Y", "Z"], "indicator": "hdi"})
    for k, y in enumerate(years):
        t[y] = [0.3 + 0.01 * k, 0.5 + 0.005 * k * (k % 3), 0.1 + 0.002 * k * k]
    t.to_csv(f, index=False)
    m = build(tmp_path, [(f, "hdr.csv")])
    d = data(tmp_path)
    assert m["frequency"] == "YS" and m["n_series"] == 3 and m["targets"] == ["value"]
    x = d[d["item_id"] == "X|hdi"].sort_values("timestamp")
    assert x["timestamp"].iloc[0] == pd.Timestamp("1960-01-01") and np.isclose(x["value"].iloc[1], 0.31)


def test_c_numeric_julian_time_gets_synthetic_calendar(tmp_path):
    f = tmp_path / "tess.csv"
    t = 1325.29 + np.arange(3000) * (2.0 / 1440)          # BTJD days, 2-minute cadence
    t = t + np.random.default_rng(0).normal(0, 1e-6, len(t))
    pd.DataFrame({"time": t, "flux": walk(3000)}).to_csv(f, index=False)
    m = build(tmp_path, [(f, "tess.csv")])
    d = data(tmp_path)
    assert m["frequency"] == "2min" and m["synthetic_time"] is True
    assert d["timestamp"].iloc[0] == pd.Timestamp("2000-01-01") and len(d) == 3000


def test_c_real_julian_dates_keep_the_calendar(tmp_path):
    f = tmp_path / "jd.csv"
    pd.DataFrame({"jd": 2459000.5 + np.arange(400), "mag": walk(400)}).to_csv(f, index=False)
    m = build(tmp_path, [(f, "jd.csv")])
    assert m["synthetic_time"] is False and m["frequency"] == "D"
    assert data(tmp_path)["timestamp"].iloc[0] == pd.Timestamp("2020-05-31")


def test_d_index_only_with_frequency_from_overrides_or_fail(tmp_path):
    f = tmp_path / "smap.csv"
    pd.DataFrame({"channel": np.repeat(["A-1", "B-2"], 300), "value": walk(600)}).to_csv(f, index=False)
    m = build(tmp_path, [(f, "smap.csv")], {"freq": "min"})
    d = data(tmp_path)
    assert m["synthetic_time"] is True and m["n_series"] == 2 and m["frequency"] == "min"
    a = d[d["item_id"] == "A-1"]
    assert a["timestamp"].iloc[0] == pd.Timestamp("2000-01-01") and len(a) == 300
    with pytest.raises(Fail, match="no time axis"):
        build(tmp_path, [(f, "smap.csv")], dataset_id="unknown_rate")
    p = plan_dataset(row(dataset_id="unknown_rate"), [(f, "smap.csv", 1)], {}, S, tmp_path / "w")
    assert p["status"] == "not built" and "no time axis" in p["reason"]


def test_d_frequency_from_the_dataset_name(tmp_path):
    f = tmp_path / "x.csv"
    pd.DataFrame({"value": walk(500)}).to_csv(f, index=False)
    m = build(tmp_path, [(f, "x.csv")], dataset_id="sensors_hourly")
    assert m["frequency"] == "h" and m["synthetic_time"] is True


def test_e_event_table_with_text_dates_counts_daily(tmp_path):
    f = tmp_path / "crimes.csv"
    rng = np.random.default_rng(0)
    days = pd.Timestamp("2022-01-01") + pd.to_timedelta(rng.integers(0, 400, 3000), unit="D")
    pd.DataFrame({"municipio": rng.choice(["Porto Alegre", "Canoas"], 3000), "data_fato": days.strftime("%d/%m/%Y"),
                  "tipo": "furto"}).to_csv(f, index=False)
    m = build(tmp_path, [(f, "crimes.csv")])                   # no id-like column name: one series, all events
    d = data(tmp_path)
    assert m["layout"] == "events" and m["frequency"] == "D" and m["n_series"] == 1 and d["count"].sum() == 3000
    m = build(tmp_path, [(f, "crimes.csv")], {"id_cols": ["municipio"]})   # an override names the series
    d = data(tmp_path)
    assert m["layout"] == "events" and m["n_series"] == 2
    assert d["count"].sum() == 3000 and (d["count"] == 0).any() and d["count"].notna().all()


def test_metadata_paths_never_chosen(tmp_path):
    assert is_meta("Dataset_V0.0/metadata/missing/NEEA.csv") and is_meta("docs/x.csv") and is_meta("README.csv")
    assert is_meta("a.zip!data_dictionary.csv") and not is_meta("Dataset_V0.0/Energy-Load-Profiles/15min/NEEA.csv")
    meta = tmp_path / "metadata" / "missing" / "NEEA.csv"
    meta.parent.mkdir(parents=True)
    pd.DataFrame({"building": [f"b{i}" for i in range(5000)], "missing_share": 0.1}).to_csv(meta, index=False)
    real = tmp_path / "lp.csv"
    pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=500, freq="15min"), "b1": walk(500)}).to_csv(real, index=False)
    m = build(tmp_path, [(meta, "Dataset_V0.0/metadata/missing/NEEA.csv"), (real, "Dataset_V0.0/Energy-Load-Profiles/15min/lp.csv")])
    assert m["frequency"] == "15min" and m["n_series"] == 1


def test_app_flow_zero_fill(tmp_path):
    f = tmp_path / "flow.csv"
    rng = np.random.default_rng(1)
    hours = pd.date_range("2020-08-01", periods=2000, freq="h")
    rows = []
    for app in ("wx", "qq"):
        on = rng.random(len(hours)) < 0.2                  # rows only where value > 0: 80% of hours absent
        for h in hours[on]:
            rows.append((len(rows), app, "z1", f"{h.year}/{h.month}/{h.day} {h.hour}:00", float(rng.integers(1, 50))))
    pd.DataFrame(rows, columns=["Unnamed: 0", "app_name", "zone", "time", "value"]).to_csv(f, index=False)
    ov = {"time_col": "time", "id_cols": ["app_name", "zone"], "targets": ["value"], "covariates": [], "freq": "h",
          "fill_zero": True}
    m = build(tmp_path, [(f, "flow.csv")], ov)
    d = data(tmp_path)
    assert m["n_series"] == 2 and d["value"].notna().all() and 0.7 < (d["value"] == 0).mean() < 0.9
    assert "Unnamed: 0" not in d.columns
    with pytest.raises(Fail, match="mostly_missing"):           # without the fill the gaps are NaN and the series drop
        build(tmp_path, [(f, "flow.csv")], {k: v for k, v in ov.items() if k != "fill_zero"})


def test_csv_encoding_fallback_and_skipped_rows(tmp_path):
    f = tmp_path / "w.csv"
    text = "name,value\n" + "".join(f"café ’{i},{i}\n" for i in range(200)) + "bad,row,extra\n"
    f.write_bytes(text.encode("cp1252"))
    p = readers.open_parts(f, tmp_path / "w")[0]
    t = p.read()
    assert p.stats["encoding"] == "cp1252" and t.num_rows == 200 and p.stats["skipped"] == 1
    assert t.column("name")[0].as_py() == "café ’0"
    g = tmp_path / "l.csv"
    g.write_bytes("name,value\né\x81,1\n".encode("latin-1"))     # 0x81 is undefined in cp1252
    q = readers.open_parts(g, tmp_path / "w")[0]
    assert q.stats["encoding"] == "latin-1" and q.read().num_rows == 1
    u = tmp_path / "u.csv"
    u.write_text("a,b\n1,2\n", encoding="utf-8")
    assert readers.open_parts(u, tmp_path / "w")[0].stats["encoding"] == "utf-8"


def test_canon_inspect_writes_blocks_and_summary(tmp_path, monkeypatch):
    import types

    from tideforge.tfbench import run as RUN
    from tideforge.tfbench.config import Paths
    paths = Paths(data=tmp_path)
    f = tmp_path / "raw" / "o_r" / "x.csv"
    f.parent.mkdir(parents=True)
    pd.DataFrame({"brand": ["a", "b"], "cnt": [1.0, 2.0]}).to_csv(f, index=False)
    pd.DataFrame({"dataset_id": ["bad", "good", "gated"], "status": ["not built", "ok", "not built"],
                  "reason": ["no time axis (x)", "", "skipped in overrides.yaml: gated"]}).to_csv(
        tmp_path / "canon_plan.csv", index=False)
    jobs = [(row(dataset_id="bad"), [(str(f), "x.csv", f.stat().st_size)], {})]
    monkeypatch.setattr(RUN, "canon_jobs", lambda a, p: jobs)
    RUN.step_canon_inspect(types.SimpleNamespace(jobs=1, only=None), paths, S)
    txt = (tmp_path / "canon_inspect.txt").read_text(encoding="utf-8")
    assert "=== bad" in txt and "brand" in txt and "would build: 1" in txt and "excluded (overrides skip): 1" in txt
    assert "no time axis" in txt


def test_canon_inspect_failed_lists_status_reason_and_traceback(tmp_path, monkeypatch):
    import json
    import types

    from tideforge.tfbench import run as RUN
    from tideforge.tfbench.config import Paths
    paths = Paths(data=tmp_path)
    f = tmp_path / "raw" / "o_r" / "x.csv"
    f.parent.mkdir(parents=True)
    pd.DataFrame({"day": ["13/02/2020", "14/02/2020"], "cnt": [1.0, 2.0]}).to_csv(f, index=False)
    for did, st in {"bad": {"status": "error", "reason": "TypeError: boom", "traceback": "Traceback\n  line 1\nTypeError: boom"},
                    "good": {"status": "ok"}, "skip": {"status": "not built", "reason": "skipped in overrides.yaml: x"}}.items():
        (paths.canon / did).mkdir(parents=True)
        (paths.canon / did / "status.json").write_text(json.dumps(st), encoding="utf-8")
    fl = [(str(f), "x.csv", f.stat().st_size)]
    jobs = [(row(dataset_id=d), fl, {"skip": "gone"} if d == "bad" else {}) for d in ("bad", "good", "skip")]
    monkeypatch.setattr(RUN, "canon_jobs", lambda a, p: jobs)
    RUN.step_canon_inspect(types.SimpleNamespace(jobs=1, only=None, inspect=True, failed=True), paths, S)
    txt = (tmp_path / "canon_inspect_failed.txt").read_text(encoding="utf-8")
    assert "=== bad" in txt and "=== good" not in txt and "=== skip" not in txt
    assert "TypeError: boom" in txt and "  line 1" in txt and "stale" in txt
    assert "time parse day: format %d/%m/%Y, unreadable 0%" in txt and "failed datasets: 1" in txt


def test_encoding_sampled_for_big_files_and_restart_when_sample_misses(tmp_path, monkeypatch):
    monkeypatch.setattr(readers, "SAMPLE_FULL", 1000)
    monkeypatch.setattr(readers, "SAMPLE_HEAD", 400)
    monkeypatch.setattr(readers, "SAMPLE_WINDOW", 200)
    monkeypatch.setattr(readers, "SAMPLE_WINDOWS", 2)
    monkeypatch.setattr(readers, "CSV_BLOCK", 4096)
    f = tmp_path / "big.csv"
    lines = [f"name{i},{i}" for i in range(3000)]
    lines[1500] = "café,1500"                       # one cp1252 byte, between the sampled windows
    f.write_bytes(("name,value\n" + "\n".join(lines) + "\n").encode("cp1252"))
    p = readers.open_parts(f, tmp_path / "w")[0]
    assert p.stats["encoding"] == "utf-8"              # the sample did not see it
    t = p.read()
    assert p.stats["encoding"] == "cp1252" and t.num_rows == 3000 and p.stats["rows"] == 3000
    assert t.column("name")[1500].as_py() == "café" and t.column("value")[2999].as_py() == 2999
