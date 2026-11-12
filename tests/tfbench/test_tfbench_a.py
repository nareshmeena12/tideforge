"""tfbench Milestone A on tiny synthetic fixtures (no network)."""
from __future__ import annotations

import gzip
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tideforge.tfbench import audit as TA
from tideforge.tfbench import download as DL
from tideforge.tfbench import readers
from tideforge.tfbench.canon import Fail, build_dataset, cov_roles, stratified_order
from tideforge.tfbench.config import Paths, Settings
from tideforge.tfbench.rules import band, min_length, slug
from tideforge.tfbench.select import select
from tideforge.tfbench.timefreq import infer_freq, parse_time, regularise

S = Settings(min_context=20, max_series=1000, max_points=50_000)
RNG = np.random.default_rng(0)


def row(did="ds", status="keep", **kw):
    return {"dataset_id": did, "repo_id": "o/r", "config_or_subset": "", "catalog_row": 1, "status": status,
            "domain": "energy", "licence_hub": "cc-by-4.0", "catalog_licence": "CC-BY", **kw}


def build(tmp_path, files, ov=None, s=S, **kw):
    return build_dataset(row(**kw), [(Path(f), Path(f).name, Path(f).stat().st_size) for f in files], ov or {}, s,
                         tmp_path / "out" / "ds", tmp_path / "work")


def walk(n, seed):
    return np.cumsum(np.random.default_rng(seed).normal(size=n)) + 50


# ---- frequency / regularisation ---------------------------------------------------------------------------------------
@pytest.mark.parametrize("freq,expect", [("h", "h"), ("15min", "15min"), ("D", "D"), ("MS", "MS"), ("ME", "ME"),
                                         ("W-SUN", "W-SUN"), ("YS", "YS"), ("QS", "QS")])
def test_infer_freq(freq, expect):
    ts = pd.Series(pd.date_range("2001-01-01", periods=60, freq=freq))
    alias, share, _ = infer_freq([ts, ts.iloc[::2]])
    assert alias == expect and share >= 0.5


def test_infer_freq_mode_with_gaps():
    ts = pd.Series(pd.date_range("2020-01-01", periods=200, freq="h")).drop(index=range(50, 60))
    assert infer_freq([ts])[0] == "h"


def test_regularise_gaps_stay_nan_and_snap():
    ts = pd.date_range("2020-01-01", periods=10, freq="h").delete([3, 4]) + pd.Timedelta(minutes=2)  # off-grid stamps
    df = pd.DataFrame({"timestamp": ts, "y": np.arange(8.0)})
    out, st = regularise(df, "h", ["y"])
    assert len(out) == 10 and out["y"].isna().sum() == 2 and st["snapped"] == 8
    assert out["timestamp"].iloc[0] == pd.Timestamp("2020-01-01")


def test_regularise_duplicates_mean_or_sum():
    df = pd.DataFrame({"timestamp": pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"]), "y": [1.0, 3.0, 5.0]})
    assert regularise(df, "D", ["y"])[0]["y"].tolist() == [2.0, 5.0]
    assert regularise(df, "D", ["y"], agg={"y": "sum"})[0]["y"].tolist() == [4.0, 5.0]


def test_regularise_keep_last():
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=100, freq="D"), "y": np.arange(100.0)})
    out, _ = regularise(df, "D", ["y"], keep_last=30)
    assert len(out) == 30 and out["y"].iloc[-1] == 99


def test_parse_time_years_and_epoch():
    assert parse_time(pd.Series([1990, 1991])).dt.year.tolist() == [1990, 1991]
    assert parse_time(pd.Series([1_600_000_000_000])).iloc[0].year == 2020
    assert parse_time(pd.Series(["2020-01-01T00:00:00+02:00"])).iloc[0] == pd.Timestamp("2019-12-31 22:00")


def test_bands_and_min_length():
    assert band("15min") == "subhourly" and band("MS") == "monthly" and band("W-SUN") == "weekly"
    s = Settings()
    assert min_length("h", s) == 256 + 48 and min_length("D", s) == 128 + 30 and min_length("W-SUN", s) == 52 + 8
    assert min_length("MS", s) == 36 + 12 and min_length("QS", s) == 16 + 8 and min_length("YS", s) == 12 + 6


# ---- layouts ------------------------------------------------------------------------------------------------------------
def long_table(n_items=6, n=300, freq="h"):
    ts = pd.date_range("2021-01-01", periods=n, freq=freq)
    return pd.concat([pd.DataFrame({"station_id": f"s{i}", "timestamp": ts, "value": walk(n, i),
                                    "temperature": walk(n, 100 + i), "hour": ts.hour})
                      for i in range(n_items)], ignore_index=True)


def test_long_parquet(tmp_path):
    f = tmp_path / "a.parquet"
    pq.write_table(pa.Table.from_pandas(long_table()), f)
    m = build(tmp_path, [f])
    d = pd.read_parquet(tmp_path / "out/ds/data.parquet")
    assert m["layout"] == "long" and m["frequency"] == "h" and m["n_series"] == 6 and m["targets"] == ["value"]
    assert set(d.columns) >= {"item_id", "timestamp", "value", "temperature", "hour"}
    assert m["covariates"] == {"temperature": "past", "hour": "known"}
    assert m["licence"] == {"hub": "cc-by-4.0", "catalog": "CC-BY"} and m["catalog_status"] == "keep"


def test_wide_csv(tmp_path):
    ts = pd.date_range("2021-01-01", periods=400, freq="D")
    df = pd.DataFrame({"date": ts.strftime("%Y-%m-%d"), **{f"sensor{i}": walk(400, i) for i in range(4)}})
    f = tmp_path / "w.csv"
    df.to_csv(f, index=False)
    m = build(tmp_path, [f])
    assert m["layout"] == "wide" and m["n_series"] == 4 and m["frequency"] == "D" and m["targets"] == ["target"]


def test_nested_parquet(tmp_path):
    t = pa.table({"item_id": ["a", "b"], "start": pd.to_datetime(["2020-01-01", "2020-03-01"]),
                  "freq": ["H", "H"], "target": [walk(300, 1).tolist(), walk(250, 2).tolist()]})
    f = tmp_path / "n.parquet"
    pq.write_table(t, f)
    m = build(tmp_path, [f])
    d = pd.read_parquet(tmp_path / "out/ds/data.parquet")
    assert m["layout"] == "nested" and m["n_series"] == 2 and m["frequency"] == "h"
    assert d.groupby("item_id").size().to_dict() == {"a": 300, "b": 250}


def test_nested_multichannel(tmp_path):
    t = pa.table({"item_id": ["a"], "start": pd.to_datetime(["2020-01-01"]), "freq": ["D"],
                  "target": [[walk(300, 1).tolist(), walk(300, 2).tolist()]]})
    f = tmp_path / "n.parquet"
    pq.write_table(t, f)
    assert build(tmp_path, [f])["targets"] == ["target_0", "target_1"]


def test_events_daily_counts(tmp_path):
    days = pd.date_range("2021-01-01", periods=400, freq="D")
    rows = []
    for k, d in enumerate(days):
        for j in range(k % 3):            # 0, 1 or 2 events per day; days with 0 events have no rows
            rows.append({"event_time": d + pd.Timedelta(hours=j + 1), "event_type": "trip"})
    f = tmp_path / "e.parquet"
    pq.write_table(pa.Table.from_pandas(pd.DataFrame(rows)), f)
    m = build(tmp_path, [f])
    d = pd.read_parquet(tmp_path / "out/ds/data.parquet")
    assert m["layout"] == "events" and m["frequency"] == "D" and m["event_rule"]["step"] == "D"
    assert d["count"].iloc[:6].tolist() == [0, 1, 2, 0, 1, 2] or d["count"].iloc[:5].tolist() == [1, 2, 0, 1, 2]


def test_no_time_column_fails(tmp_path):
    f = tmp_path / "x.csv"
    pd.DataFrame({"a": np.arange(10.0), "b": np.arange(10.0)}).to_csv(f, index=False)
    with pytest.raises(Fail, match="time column"):
        build(tmp_path, [f])


def test_unreadable_format_fails(tmp_path):
    f = tmp_path / "x.npy"
    np.save(f, np.zeros(3))
    with pytest.raises(Fail, match="no readable table"):
        build(tmp_path, [f])


# ---- caps and filters ---------------------------------------------------------------------------------------------------
def test_series_cap_is_frozen_and_stratified(tmp_path):
    ts = pd.date_range("2021-01-01", periods=200, freq="h")
    df = pd.concat([pd.DataFrame({"region": f"r{i % 2}", "site_id": f"s{i}", "timestamp": ts, "value": walk(200, i)})
                    for i in range(12)], ignore_index=True)
    f = tmp_path / "a.parquet"
    pq.write_table(pa.Table.from_pandas(df), f)
    s = Settings(min_context=20, max_series=4, max_points=150)
    m1 = build(tmp_path, [f], s=s, ov={"id_cols": ["site_id"], "strata": "region"})
    d1 = pd.read_parquet(tmp_path / "out/ds/data.parquet")
    m2 = build(tmp_path, [f], s=s, ov={"id_cols": ["site_id"], "strata": "region"})
    d2 = pd.read_parquet(tmp_path / "out/ds/data.parquet")
    assert m1["n_series"] == 4 and set(d1["item_id"]) == set(d2["item_id"])
    assert d1.groupby("item_id").size().max() == 150
    assert d1.groupby("item_id")["timestamp"].max().eq(ts[-1]).all()   # the last points are kept


def test_stratified_order_interleaves():
    items = pd.DataFrame({"item_id": [f"i{k}" for k in range(20)], "part": -1, "key": None,
                          "stratum": ["a"] * 10 + ["b"] * 10})
    o = stratified_order(items, 1)
    assert set(o["stratum"].iloc[:6]) == {"a", "b"}
    assert o["item_id"].tolist() == stratified_order(items.sample(frac=1, random_state=3), 1)["item_id"].tolist()


def test_filters_constant_missing_short_duplicate(tmp_path):
    ts = pd.date_range("2021-01-01", periods=300, freq="h")
    good = walk(300, 1)
    miss = walk(300, 2)
    miss[:200] = np.nan
    parts = [("good", ts, good), ("dup", ts, good), ("const", ts, np.full(300, 3.0)), ("miss", ts, miss),
             ("short", ts[:50], walk(50, 3))]
    df = pd.concat([pd.DataFrame({"site_id": n, "timestamp": t, "value": v}) for n, t, v in parts], ignore_index=True)
    f = tmp_path / "a.parquet"
    pq.write_table(pa.Table.from_pandas(df), f)
    m = build(tmp_path, [f])
    drops = next(t for t in m["transformations"] if t["step"] == "series filters")["dropped"]
    assert m["n_series"] == 1
    assert drops == {"duplicate": 1, "constant": 1, "mostly_missing": 1, "too_short": 1}


def test_keep_filtered_needs_rule_and_applies_it(tmp_path):
    f = tmp_path / "a.parquet"
    pq.write_table(pa.Table.from_pandas(long_table()), f)
    with pytest.raises(Fail, match="no drop rule"):
        build(tmp_path, [f], status="keep_filtered")
    m = build(tmp_path, [f], status="keep_filtered", ov={"drop_items": "^s[01]$"})
    assert m["n_series"] == 4


def test_cov_roles():
    r = cov_roles(["temp", "day_of_week", "is_holiday", "price", "load_forecast", "x"], ["x"])
    assert r == {"temp": "past", "day_of_week": "known", "is_holiday": "known", "price": "known_candidate",
                 "load_forecast": "known_candidate", "x": "known"}


# ---- readers ------------------------------------------------------------------------------------------------------------
def test_zip_and_gz_and_tar(tmp_path):
    df = long_table(2, 300)
    csv = tmp_path / "in.csv"
    df.to_csv(csv, index=False)
    z = tmp_path / "a.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.write(csv, "folder/data.csv")
        zf.writestr("README.txt", "hello")
    g = tmp_path / "b.csv.gz"
    g.write_bytes(gzip.compress(csv.read_bytes()))
    for f in (z, g):
        parts = readers.open_parts(f, tmp_path / "work")
        t = pa.concat_tables([p.read() for p in parts if "station_id" in p.schema.names])
        assert t.num_rows == len(df)
    assert build(tmp_path, [z])["n_series"] == 2


def test_csv_streamed_in_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(readers, "CSV_BLOCK", 4096)
    f = tmp_path / "big.csv"
    long_table(3, 300).to_csv(f, index=False)
    p = readers.open_parts(f, tmp_path)[0]
    chunks = list(p.batches(["value"]))
    assert len(chunks) > 3 and sum(c.num_rows for c in chunks) == 900
    assert pa.types.is_floating(p.schema.field("hour").type)   # integers read as float: later chunks cannot overflow


def test_unreadable_formats(tmp_path):
    for name in ("x.h5", "x.npy", "x.nc"):
        (tmp_path / name).write_bytes(b"\0" * 10)
        with pytest.raises(readers.Unreadable):
            readers.open_parts(tmp_path / name, tmp_path)


# ---- selection / download ---------------------------------------------------------------------------------------------
def test_select_rules():
    base = {"row_kind": "catalog", "profile_level": "viewer", "error": "", "layout": "long", "frequency": "",
            "ingest_difficulty": "easy", "domain": "x", "config_or_subset": ""}
    p = pd.DataFrame([
        {**base, "catalog_row": 1, "repo_id": "a/ok", "status": "keep", "scope_bytes": 1e9},
        {**base, "catalog_row": 2, "repo_id": "a/check", "status": "check", "scope_bytes": 1e9},
        {**base, "catalog_row": 3, "repo_id": "a/big", "status": "keep", "scope_bytes": 30e9},
        {**base, "catalog_row": 4, "repo_id": "a/allowed", "status": "keep_filtered", "scope_bytes": 30e9},
        {**base, "catalog_row": 5, "repo_id": "a/gated", "status": "keep", "scope_bytes": 1e9, "profile_level": "gated"},
        {**base, "catalog_row": 6, "repo_id": "a/x", "status": "keep", "scope_bytes": 1e9, "row_kind": "config"},
    ])
    s = select(p, Settings(allow_large=("a/allowed",)))
    assert s["keep"].tolist() == [True, False, False, True, False]
    assert "gated" in s["reason"].iloc[4] and "30.0 GB > 20 GB" in s["reason"].iloc[2]
    assert s["dataset_id"].iloc[0] == "a_ok"


def test_download_plan_dedupes_and_skips_formats():
    sel = pd.DataFrame({"catalog_row": [1, 2], "dataset_id": ["d1", "d2"], "keep": [True, True]})
    files = pd.DataFrame({"catalog_row": [1, 2, 2, 1], "repo_id": ["o/r"] * 4, "revision": ["main"] * 4,
                          "path": ["x.parquet", "x.parquet", "y.h5", "z.csv.gz"], "bytes": [10, 10, 5, 3]})
    p = DL.plan(sel, files)
    assert len(p) == 3
    assert p.set_index("path").loc["x.parquet", "dataset_ids"] == "d1;d2"
    assert not p.set_index("path").loc["y.h5", "fetch"]


def test_download_repo_resume_and_disk(tmp_path):
    paths = Paths(data=tmp_path)
    rows = pd.DataFrame({"repo_id": "o/r", "revision": "main", "path": ["a/x.csv", "y.csv"], "bytes": [3, 3],
                         "dataset_ids": "d"})
    calls = []

    def fetch(repo, path, rev, root):
        calls.append(path)
        p = Path(root) / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"abc")
        return p
    r = DL.download_repo("o/r", rows, paths, Settings(), fetch=fetch, disk_free=lambda p: 1e12)
    assert r["status"] == "ok" and len(calls) == 2
    DL.download_repo("o/r", rows, paths, Settings(), fetch=fetch, disk_free=lambda p: 1e12)
    assert len(calls) == 2                                        # resumed: nothing fetched again
    m = DL.combine(paths)
    assert m["sha256"].iloc[0] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    (tmp_path / "raw" / slug("o/r") / "y.csv").unlink()
    r = DL.download_repo("o/r", rows, paths, Settings(), fetch=fetch, disk_free=lambda p: 0)
    assert r["status"] == "skipped: disk" and len(calls) == 2


# ---- audit --------------------------------------------------------------------------------------------------------------
def _canon(tmp_path, series: dict, name="dev"):
    d = tmp_path / "canon" / name
    d.mkdir(parents=True)
    ts = pd.date_range("2020-01-01", periods=len(next(iter(series.values()))), freq="h")
    df = pd.concat([pd.DataFrame({"item_id": k, "timestamp": ts, "target": v}) for k, v in series.items()])
    df.to_parquet(d / "data.parquet")
    (d / "meta.json").write_text(json.dumps({"dataset_id": name, "frequency": "h", "targets": ["target"]}))
    return d


def test_audit_finds_planted_shifted_rescaled_copy(tmp_path):
    from tideforge.devdata.audit import DB
    ref = walk(2000, 7) + 5 * np.sin(np.arange(2000) / 3)
    db = DB()
    db.add(ref, bench="fev", dataset="task_a", item="0", freq="")
    db.build()
    clean = {f"c{i}": walk(1000, 50 + i) for i in range(40)}
    copy = 3.0 * ref[137:1137] + 11.0                      # shifted by 137 points, rescaled and offset
    d = _canon(tmp_path, {**clean, "planted": copy})
    a = TA.audit_dataset(d, db, Settings())
    assert [x["item_id"] for x in a["dropped_series"]] == ["planted"]
    assert a["dropped_series"][0]["matched_task"] == "fev:task_a"
    assert a["old_rule_flag"] and not a["flagged"]           # series dropped, the rest of the dataset is clean
    # a dataset that is mostly a copy at a low share per series: flagged at dataset level
    part = walk(1000, 9)
    part[:300] = ref[500:800]
    d2 = _canon(tmp_path, {f"p{i}": part + i for i in range(3)} | {f"c{i}": walk(1000, 80 + i) for i in range(2)}, "dev2")
    a2 = TA.audit_dataset(d2, db, Settings(audit_series_share=0.5))
    assert a2["series_dropped"] == 0 and a2["flagged"] and a2["share_after_drops"] > 0.01
    (tmp_path / "audit").mkdir()
    (tmp_path / "audit" / "dev.json").write_text(json.dumps(a, default=str))
    assert "planted" not in set(TA.load_clean(d, tmp_path / "audit")["item_id"])
    s = TA.summarise(tmp_path / "audit")
    assert {"old_rule_windows", "old_rule_flag", "flagged"} <= set(s.columns)


def test_audit_ref_series_from_nested_parquet(tmp_path):
    root = tmp_path / "refs" / "fev"
    root.mkdir(parents=True)
    pq.write_table(pa.table({"id": ["a"], "target": [walk(100, 1).tolist()]}), root / "t.parquet")
    got = list(TA.ref_series(root, "fev", tmp_path / "w"))
    assert len(got) == 1 and len(got[0][0]) == 100 and got[0][1]["bench"] == "fev"


def test_audit_stride16_shift7_rescaled(tmp_path):
    """A copy shifted by 7 points (not a multiple of 16), rescaled and offset, is caught and dropped."""
    from tideforge.devdata.audit import DB
    ref = walk(3000, 11) + 4 * np.sin(np.arange(3000) / 5)
    db = DB()
    db.add(ref, bench="GIFT-Eval", dataset="g", item="0", freq="")
    db.build()
    d = _canon(tmp_path, {"copy": 2.5 * ref[7:1207] + 100.0, **{f"c{i}": walk(1200, 200 + i) for i in range(5)}}, "s7")
    a = TA.audit_dataset(d, db, Settings())
    assert [x["item_id"] for x in a["dropped_series"]] == ["copy"]
    assert a["dropped_series"][0]["share"] > 0.9 and a["share_after_drops"] == 0


def test_ref_ids_drop(tmp_path):
    refs = tmp_path / "refs" / "autogluon_fev_datasets" / "boomlet_1"
    refs.mkdir(parents=True)
    pq.write_table(pa.table({"id": ["s1", "s3"], "target": [[1.0, 2.0], [3.0, 4.0]]}), refs / "t.parquet")
    f = tmp_path / "a.parquet"
    pq.write_table(pa.Table.from_pandas(long_table()), f)
    m = build_dataset(row(status="keep_filtered"), [(f, f.name, 1)], {"ref_ids": {"path": "(?i)fev_datasets/.*boom"}},
                      S, tmp_path / "out" / "ds", tmp_path / "work", tmp_path / "refs")
    assert m["n_series"] == 4
    assert next(x for x in m["transformations"] if x["step"].startswith("ref_ids"))["count"] == 2


def test_pick_smoke_covers_cases():
    from tideforge.tfbench.run import pick_smoke
    sel = pd.DataFrame({"catalog_row": range(1, 7), "dataset_id": [f"d{i}" for i in range(1, 7)], "keep": True,
                        "layout": ["long", "wide", "nested", "events", "long", "long"],
                        "status": ["keep"] * 5 + ["keep_filtered"], "scope_bytes": [1, 2, 3, 4, 5, 6]})
    prof = pd.DataFrame({"catalog_row": range(1, 7), "scope_formats": ["parquet: 1 B", "csv: 1 B", "", "", "zip: 5 B", ""],
                         "freq_band": ["monthly", "", "", "", "", ""], "has_covariates": ["yes", "", "", "", "", ""]})
    picks = dict(pick_smoke(sel, prof))
    assert set(picks) == {"d1", "d2", "d3", "d4", "d5", "d6"}
    assert "monthly or yearly" in picks["d1"] and "covariates" in picks["d1"]
