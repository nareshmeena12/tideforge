"""Sizing plan of the real half: proportional passes, 4-epoch caps, weather as one group."""
import json

import pytest

from tidemark.real import plan as P

U = {"dominick": 0.5e9, "exchange_rate": 0.1e9, "wind_farms_hourly": 1.4e9, "weatherbench_daily": 3.0e9,
     "weatherbench_hourly_geopotential": 8.0e9, "weatherbench_weekly": 1.0e9}      # 14.0B unique


def test_proportional_when_the_caps_do_not_bind():
    pl = P.build_plan(U, 10e9)
    ep = {d: r["epochs"] for d, r in pl["datasets"].items()}
    assert all(e == pytest.approx(10 / 14) for e in ep.values())                       # same number of passes everywhere
    assert sum(r["planned_points"] for r in pl["datasets"].values()) == pytest.approx(10e9)
    assert sum(r["share"] for r in pl["datasets"].values()) == pytest.approx(1.0)
    w = pl["units"]["weather"]
    assert w["unique_points"] == pytest.approx(12e9) and w["share"] == pytest.approx(12 / 14)       # the weather group dominates


def test_weather_is_one_group_split_by_unique_points():
    pl = P.build_plan(U, 10e9)
    d = pl["datasets"]
    assert d["weatherbench_hourly_geopotential"]["planned_points"] / d["weatherbench_daily"]["planned_points"] == pytest.approx(8 / 3)
    assert {r["unit"] for k, r in d.items() if k.startswith("weatherbench_")} == {"weather"}


def test_caps_bind_and_the_rest_goes_to_the_others():
    small = {"a": 1e9, "b": 1e9, "weatherbench_daily": 20e9}
    pl = P.build_plan(small, 30e9, max_epochs=4)               # proportional would give 1.36 epochs; the cap is 4
    assert all(r["epochs"] == pytest.approx(30 / 22) for r in pl["datasets"].values())
    pl = P.build_plan({"a": 1e9, "b": 5e9}, 12e9, max_epochs=4)         # proportional 2 epochs each -> fine
    assert pl["datasets"]["a"]["epochs"] == pytest.approx(2)
    pl = P.build_plan({"a": 1e9, "b": 5e9}, 22e9, max_epochs=4, unit_weights={"a": 10.0, "b": 1.0})   # a wants more than its cap
    assert pl["datasets"]["a"]["epochs"] == pytest.approx(4) and pl["datasets"]["b"]["planned_points"] == pytest.approx(18e9)
    assert pl["unplaced_points"] == 0


def test_infeasible_budget_is_reported():
    pl = P.build_plan({"a": 1e9, "weatherbench_daily": 1e9}, 10e9, max_epochs=4)           # 8B fit under the caps
    assert pl["unplaced_points"] == pytest.approx(2e9)
    assert "cannot be placed" in P.report(pl)


def test_weather_max_share_hands_the_rest_to_the_others():
    pl = P.build_plan(U, 10e9, weather_max_share=0.4)
    assert pl["units"]["weather"]["share"] == pytest.approx(0.4)
    assert pl["datasets"]["dominick"]["epochs"] == pytest.approx(pl["datasets"]["wind_farms_hourly"]["epochs"])
    assert pl["datasets"]["dominick"]["epochs"] > 10 / 14


def test_report_and_manifest(tmp_path, capsys):
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"datasets": {k: {"total_points": v} for k, v in U.items()}}))
    out = tmp_path / "plan.json"
    assert P.main(["--manifest", str(man), "--real-points", "1e10", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "weather group" in text and "TOTAL" in text and "dominick" in text and "ABOVE" not in text
    assert json.loads(out.read_text())["targets"]["dominick"] == pytest.approx(0.5 / 14)


def test_weather_split_hourly_daily_weekly_and_flags_above_cap():
    pl = P.build_plan(U, 10e9, weather_max_share=0.77, weather_split=P.WEATHER_SPLIT)
    w = pl["units"]["weather"]["planned_points"]
    d = pl["datasets"]
    assert w == pytest.approx(7.7e9)
    assert d["weatherbench_hourly_geopotential"]["planned_points"] == pytest.approx(0.5 * w)
    assert d["weatherbench_daily"]["planned_points"] == pytest.approx(0.4 * w)
    assert d["weatherbench_weekly"]["planned_points"] == pytest.approx(0.1 * w)
    assert sum(r["planned_points"] for r in d.values()) == pytest.approx(10e9)
    tight = P.build_plan({"a": 1e9, "weatherbench_daily": 1e9, "weatherbench_weekly": 0.05e9, "weatherbench_hourly_x": 5e9}, 10e9,
                         weather_max_share=0.77, weather_split=P.WEATHER_SPLIT)
    assert tight["datasets"]["weatherbench_weekly"]["epochs"] > 4 and "ABOVE" in P.report(tight)
    with pytest.raises(ValueError):
        P.build_plan(U, 10e9, weather_split={"hourly": 0.5, "daily": 0.4, "weekly": 0.2})


def test_match_plan_reuses_points_and_shares_and_changes_only_the_epochs(tmp_path, capsys):
    first = P.build_plan(U, 10e9, weather_max_share=0.78, weather_split=P.WEATHER_SPLIT)
    big = dict(U)
    big["weatherbench_hourly_geopotential"] = 40e9                                 # the bigger sampling pool
    m = P.match_plan(big, first)
    for d in U:
        assert m["datasets"][d]["planned_points"] == first["datasets"][d]["planned_points"]
        assert m["datasets"][d]["share"] == first["datasets"][d]["share"] and m["targets"][d] == first["targets"][d]
    assert m["datasets"]["weatherbench_hourly_geopotential"]["epochs"] == pytest.approx(
        first["datasets"]["weatherbench_hourly_geopotential"]["planned_points"] / 40e9)
    assert m["datasets"]["weatherbench_daily"]["epochs"] == pytest.approx(first["datasets"]["weatherbench_daily"]["epochs"])
    assert m["datasets"]["weatherbench_hourly_geopotential"]["epochs_first_plan"] == first["datasets"]["weatherbench_hourly_geopotential"]["epochs"]
    assert sum(r["planned_points"] for r in m["datasets"].values()) == pytest.approx(10e9)
    text = P.report(m)
    assert "REUSED" in text and "epochs 1st" in text
    with pytest.raises(SystemExit, match="datasets differ"):
        P.match_plan({k: v for k, v in big.items() if k != "dominick"}, first)
    # the command line
    (tmp_path / "plan.json").write_text(json.dumps(first))
    (tmp_path / "big.json").write_text(json.dumps({"datasets": {k: {"total_points": v} for k, v in big.items()}}))
    assert P.main(["--manifest", str(tmp_path / "big.json"), "--match-plan", str(tmp_path / "plan.json"), "--out", str(tmp_path / "full.json")]) == 0
    out = json.loads((tmp_path / "full.json").read_text())
    assert out["targets"] == first["targets"] and out["matched_from"].endswith("plan.json")
    assert "REUSED" in capsys.readouterr().out
