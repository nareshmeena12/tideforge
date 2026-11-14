"""Owner's BUILD_REPORT review (2026-11-14): target rule, calendars and masked eval NaN, model-output covariates, task
drops, source groups."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from tideforge.tfbench import canon as CN
from tideforge.tfbench import split as SP
from tideforge.tfbench import tasks as TK
from tideforge.tfbench import tiers as TI
from tideforge.tfbench.config import Settings

S = Settings()


def meta(did, freq, covs=None, targets=("y",)):
    return {"dataset_id": did, "frequency": freq, "targets": list(targets), "covariates": covs or {}, "static": [],
            "domain": "x", "layout": "long"}


def rec(tmp, did):
    return json.loads((tmp / did / "subset.json").read_text())


# ---- target rule ----------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["op_carrier_fl_num", "DEB_TIME_HOUR", "forecast_hour", "Months_Reported", "AgeGrpStart",
                                  "AgeGrpSpan", "SortOrder", "Temperature_Setpoint_0", "x_t-11_d0", "MO", "station_id", "row_no"])
def test_never_targets(name):
    assert CN.is_bad_target(name)


@pytest.mark.parametrize("name", ["PopTotal", "WAIT_TIME_MAX", "price_per_hour", "traffic_volume", "Total_Units", "close",
                                  "numinptbeds", "count_rate", "value", "no2", "NO2"])
def test_real_targets_kept(name):
    assert not CN.is_bad_target(name)


def schema(*names):
    return pa.schema([("station", pa.string()), ("date", pa.timestamp("ms"))] + [(n, pa.float64()) for n in names])


def test_first_value_rule_skips_clock_and_id_columns():
    r, _, _ = CN.roles_for(schema("DEB_TIME_HOUR", "WAIT_TIME_MAX", "NB_UNITS"), {})
    assert r["target_cols"] == ["WAIT_TIME_MAX"]
    r, _, _ = CN.roles_for(schema("op_carrier_fl_num", "dep_delay", "arr_delay"), {})
    assert r["target_cols"] == ["dep_delay"]


def test_targets_match_override():
    r, _, _ = CN.roles_for(schema("VAP_1", "corn_yield_bu_ac", "area"), {"targets_match": "(?i)yield"})
    assert r["target_cols"] == ["corn_yield_bu_ac"] and r["target_rule"] == "overrides.yaml"
    with pytest.raises(CN.Fail):
        CN.roles_for(schema("VAP_1"), {"targets_match": "(?i)yield"})


# ---- calendars, masked eval NaN -------------------------------------------------------------------------------------------
def trading(n_series=8, n=3000, seed=0):
    """exchange_rate as canon writes it on a daily grid: weekends NaN (markets closed)."""
    r = np.random.default_rng(seed)
    parts = []
    for k in range(n_series):
        ts = pd.date_range("2000-01-03", periods=n, freq="D")
        y = 1 + np.cumsum(r.normal(0, 0.005, n))
        y[ts.dayofweek >= 5] = np.nan
        parts.append(pd.DataFrame({"item_id": f"fx{k}", "timestamp": ts, "y": y}))
    return pd.concat(parts, ignore_index=True)


def test_weekend_closed_daily_becomes_business_days(tmp_path):
    t = TK.build(trading(), meta("xr", "D"), {}, S, tmp_path / "xr")
    r = rec(tmp_path, "xr")
    assert t and r["calendar"].startswith("business") and r["variants"][0]["freq"] == "B"
    assert r["variants"][0]["horizons"]["short"]["series"] == 8 and r["variants"][0]["horizons"]["short"]["nan_in_eval"] == 0
    assert {x["freq_config"] for x in t} == {"xr__b", "xr__wsun"}      # B -> weekly: 5 points per week cover the bucket


def test_night_closure_is_masked_not_dropped(tmp_path):
    """hzmetro-like: 15-min metro counts, closed 23:00-06:00 every night (NaN on the grid), plus rare dropouts."""
    r = np.random.default_rng(1)
    parts = []
    for k in range(6):
        ts = pd.date_range("2019-01-01", periods=96 * 40, freq="15min")
        y = 50 + 30 * np.sin(np.arange(len(ts)) * 2 * np.pi / 96) + r.normal(0, 3, len(ts))
        y[(ts.hour >= 23) | (ts.hour < 6)] = np.nan
        y[r.random(len(ts)) < 0.01] = np.nan
        parts.append(pd.DataFrame({"item_id": f"st{k}", "timestamp": ts, "y": y}))
    t = TK.build(pd.concat(parts, ignore_index=True), meta("hz", "15min"), {}, S, tmp_path / "hz")
    h = rec(tmp_path, "hz")["variants"][0]["horizons"]["short"]
    assert t and h["series"] == 6 and h["nan_in_eval"] == 0 and h["closed_eval_nan"] > 0


def test_model_output_covariates_dropped(tmp_path):
    r = np.random.default_rng(2)
    parts = []
    for k in range(4):
        n = 600
        d = pd.DataFrame({"item_id": f"c{k}", "timestamp": pd.date_range("2020-01-01", periods=n, freq="D"),
                          "y": np.cumsum(r.normal(size=n)) + 50})
        for c in ("persistence", "plain_seed42", "gbm", "chronos-bolt median", "lag_7", "rolling_mean_7", "temp",
                  "y_forecast", "wind_forecast"):
            d[c] = r.normal(size=n)
        parts.append(d)
    covs = {c: "past" for c in ("persistence", "plain_seed42", "gbm", "chronos-bolt median", "lag_7", "rolling_mean_7",
                                "temp", "y_forecast")} | {"wind_forecast": "known"}
    t = TK.build(pd.concat(parts, ignore_index=True), meta("aq", "D", covs), {}, S, tmp_path / "aq")
    assert set(rec(tmp_path, "aq")["dropped_covariates_model_output"]) == {
        "persistence", "plain_seed42", "gbm", "chronos-bolt median", "lag_7", "rolling_mean_7"}
    cov = next(x["task"] for x in t if x["variant"] == "cov")
    assert cov["past_dynamic_columns"] == ["temp", "y_forecast"]     # "y" is too short a token to call it a forecast of y
    assert cov["known_dynamic_columns"] == ["wind_forecast"]


def test_forecast_of_the_target_dropped():
    assert TK.model_output_cols(["load_forecast", "temp_forecast", "price"], ["load"]) == ["load_forecast"]


def test_small_tasks_and_near_constant_series_dropped(tmp_path):
    n = 400
    ts = pd.date_range("2020-01-01", periods=n, freq="D")
    steps = np.repeat(np.arange(n // 100) % 2, 100).astype(float)            # 2 distinct values: near constant
    d = pd.DataFrame({"item_id": "a", "timestamp": ts, "y": steps})
    assert TK.build(d, meta("k", "D"), {}, Settings(min_task_points=1), tmp_path / "k") == []
    assert rec(tmp_path, "k")["variants"][0]["horizons"]["short"]["zero_scale"] == 1
    r = np.random.default_rng(3)
    d = pd.DataFrame({"item_id": "b", "timestamp": ts, "y": np.cumsum(r.normal(size=n))})   # 1 x 1 x 30 < 100
    assert TK.build(d, meta("s", "D"), {}, S, tmp_path / "s") == []
    assert "task too small" in rec(tmp_path, "s")["variants"][0]["horizons"]["short"]["skipped"]


def test_intermittent_series_survive(tmp_path):
    """80% zeros, Poisson counts otherwise: not near constant, the series and its task are kept."""
    r = np.random.default_rng(5)
    parts = []
    for k in range(4):
        n = 600
        y = np.where(r.random(n) < 0.8, 0.0, r.poisson(4, n) + 1.0)
        parts.append(pd.DataFrame({"item_id": f"i{k}", "timestamp": pd.date_range("2020-01-01", periods=n, freq="D"),
                                   "y": y}))
    d = pd.concat(parts, ignore_index=True)
    assert (d["y"] == 0).mean() > 0.75
    t = TK.build(d, meta("ic", "D"), {}, S, tmp_path / "ic")
    h = rec(tmp_path, "ic")["variants"][0]["horizons"]["short"]
    assert t and h["series"] == 4 and h["zero_scale"] == 0


def test_two_value_series_dropped():
    assert TK.near_constant(np.tile([0.0, 1.0, 1.0], 100), 7)
    assert not TK.near_constant(np.tile([0.0, 1.0, 2.0], 100), 7)


def test_thuml_bundle_does_not_merge_groups():
    import yaml
    from tideforge.tfbench import config as C
    base = C.TFB
    gm = SP.group_map(yaml.safe_load((base / "source_groups.yaml").read_text(encoding="utf-8")))
    assert not any(d.startswith("thuml_utsd_") for d in gm)
    assert gm["salesforce_lotsa_data_lcl"] == "london_smart_meters"
    assert gm["salesforce_lotsa_data_sunspot_with_missing"] == "sunspot"
    ov = yaml.safe_load((base / "overrides.yaml").read_text(encoding="utf-8"))
    assert "already present" in ov["thuml_utsd_london_smart_meters_wind_farms_aus_electricity_demand_pedestrian_counts_"
                                   "web_traffic_sunspot"]["skip"]


def test_version_flag_with_from_split(monkeypatch, tmp_path):
    from tideforge.tfbench import run as RN
    seen = []
    for st in RN.STEPS:
        monkeypatch.setattr(RN, f"step_{st}", lambda a, p, s, st=st: seen.append((st, s.version)), raising=False)
    monkeypatch.setattr(RN, "setup_logging", lambda paths, step: tmp_path / f"{step}.log")
    assert RN.main(["all", "--from", "split", "--no-upload", "--redo-b", "--version", "v1.1.0"]) == 0
    assert seen[0][0] == "split" and "upload" not in [x for x, _ in seen] and {v for _, v in seen} == {"v1.1.0"}
    assert Settings().version == "v1.1.0"


def test_baseline_filter_drops_tiny_mase(tmp_path):
    """Large in-sample moves, an almost flat eval span: seasonal-naive MASE far below 0.01 -> task dropped."""
    r = np.random.default_rng(4)
    parts = []
    for k in range(5):
        y = np.r_[np.cumsum(r.normal(0, 100, 400)), np.zeros(30)]
        y[-30:] = y[-31] + r.normal(0, 1e-4, 30)
        parts.append(pd.DataFrame({"item_id": f"m{k}", "timestamp": pd.date_range("2020-01-01", periods=430, freq="D"),
                                   "y": y}))
    t = TK.build(pd.concat(parts, ignore_index=True), meta("mz", "D"), {}, S, tmp_path / "mz")
    dropped = rec(tmp_path, "mz")["dropped_tasks"]
    assert not any(x["config"] == "mz__d" for x in t) and any("MASE" in x["why"] for x in dropped)


# ---- source groups --------------------------------------------------------------------------------------------------------
def test_groups_merge_and_stay_whole():
    groups = {"lsm": ["a1", "a2", "multi"], "wind": ["w1", "multi"], "nn5": ["n1", "n2"]}
    gm = SP.group_map(groups)
    assert gm["a1"] == gm["w1"] == gm["multi"] == "lsm+wind" and gm["n1"] == "nn5"
    ds = {d: "energy" for d in ["a1", "a2", "multi", "w1", "n1", "n2"] + [f"x{i}" for i in range(30)]}
    out = SP.assign(ds, None, groups)
    by = pd.DataFrame(out["datasets"]).T
    assert (by.groupby("source_group")["split"].nunique() == 1).all()
    assert out["unit"] == "source_group" and by.loc["x3", "source_group"] == "x3"
    later = SP.assign({**ds, "n3": "energy"}, out, {**groups, "nn5": ["n1", "n2", "n3"]})
    assert later["datasets"]["n3"]["split"] == out["datasets"]["n1"]["split"]        # joins its group's split


def test_legacy_splits_regenerated_once(tmp_path):
    p = tmp_path / "splits.json"
    p.write_text(json.dumps({"rule": "old", "datasets": {"a": {"split": "locked", "domain": "x"}}}))
    out = SP.load_or_create(p, {"a": "x", "b": "x"}, {"g": ["a", "b"]})
    assert out["unit"] == "source_group" and (tmp_path / "splits_by_dataset.json").exists()
    assert out["datasets"]["a"]["split"] == out["datasets"]["b"]["split"]
    assert SP.load_or_create(p, {"a": "x", "b": "x", "c": "x"}, {"g": ["a", "b"]})["datasets"]["a"] == out["datasets"]["a"]


def test_weight_one_per_group():
    t = pd.DataFrame({"dataset_id": ["a", "a", "b", "c", "c", "c"], "source_group": ["g", "g", "g", "c", "c", "c"]})
    w = TI.weights(t, np.ones(len(t), bool))
    assert np.allclose(w.groupby(t["source_group"]).sum(), 1.0)
    assert np.isclose(w[t["dataset_id"] == "a"].sum(), 0.5) and np.isclose(w[t["dataset_id"] == "b"].sum(), 0.5)
