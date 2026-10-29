"""X1 / X2 / X4: unions in order, every kind of conflict refused, role groups cover every switch, X2 removes one group and
keeps the model, X4 sets the points, the X1 report compares predicted and observed gains."""
import json

import pandas as pd
import pytest

from tideforge import expcfg
from tideforge.campaign import stack as S


def test_roles_cover_every_switch():
    S.check_roles()
    r = S.roles()
    r["groups"]["counts"]["switches"].remove("demand")
    with pytest.raises(ValueError, match="without a group"):
        S.check_roles(r)


def test_x1_unions_in_order(tmp_path):
    ps = S.x1(["G4.3", "G1.8", "G6.1", "G1.13a"], tmp_path)
    assert [p.stem for p in ps] == ["X1.1", "X1.2", "X1.3", "X1.4"]
    e = expcfg.load_exp(ps[-1])
    assert {"sc_load", "sc_pv", "gaps_scattered"} <= set(dict(e.gen.switches))
    assert e.train["crop_breaks"] and e.model == {"known_attn": "mask"}
    assert expcfg.load_exp(ps[0]).train == expcfg.load_exp("G4.3").train
    with pytest.raises(FileExistsError):
        S.x1(["G4.3"], tmp_path)


@pytest.mark.parametrize("pair,match", [(["G2.1", "G2.3a"], "cal_engine"), (["G3.1a", "G3.1b"], "family_share"),
                                        (["G7.4a", "G7.4b"], "int_share")])
def test_conflicting_switches_are_refused(pair, match):
    with pytest.raises(S.Conflict, match=match):
        S.union(pair)


def test_gen_train_base_and_plan_conflicts(monkeypatch):
    real = S.raw
    fake = {"A": dict(id="A", base="b0", gen={"second_group_p": 0.5}, train={"long_row_frac": 0.5}),
            "B": dict(id="B", base="b0", gen={"second_group_p": 1.0}),
            "C": dict(id="C", base="b0", train={"long_row_frac": 0.3}),
            "D": dict(id="D", base="defaults")}
    monkeypatch.setattr(S, "raw", lambda i: fake.get(i) or real(i))
    for pair, match in ((["A", "B"], "second_group_p"), (["A", "C"], "long_row_frac"), (["A", "D"], "base")):
        with pytest.raises(S.Conflict, match=match):
            S.union(pair)
    monkeypatch.setattr(S.SW, "plan", lambda sw: [("layer", "x", "breaks", "f"), ("layer", "y", "breaks", "g")])
    with pytest.raises(S.Conflict, match="both change layer"):
        S.plan_conflicts({})


def test_x2_removes_one_group_and_keeps_the_model(tmp_path):
    S.x1(["G4.3", "G1.8", "G6.1", "G1.13a", "G1.1"], tmp_path)
    raw = expcfg.load_exp(tmp_path / "X1.5.yaml").raw
    d = {k: v for k, v in raw.items() if k in ("base", "gen", "switches", "train", "model")}
    assert S.groups_present(d) == ["covariates", "breaks_gaps"]
    cov = S.remove_group(d, "covariates")
    assert set(cov["switches"]) == {"gaps_scattered"} and "gen" not in cov and cov["model"] == {"known_attn": "mask"}
    bg = S.remove_group(d, "breaks_gaps")
    assert "gaps_scattered" not in bg["switches"] and "train" not in bg and bg["gen"]["second_group_p"] == 1.0


def test_x4_sets_the_points(tmp_path):
    e = expcfg.load_exp(S.x4("G1.13a", tmp_path, 2e11))
    assert e.id == "X4" and e.train["target_points"] == 2e11 and e.model == {"known_attn": "mask"}


def test_report_predicted_vs_observed(tmp_path):
    b = tmp_path / "board.csv"
    pd.DataFrame(dict(exp=["G4.3", "G1.8", "X1.1", "X1.2"], dev_gain=[0.03, 0.02, 0.031, 0.02],
                      verdict=["win", "win", "", ""])).to_csv(b, index=False)
    n = tmp_path / "noise.json"
    n.write_text(json.dumps({"dev": {"metrics": {"MASE": {"range": 0.01}}}}))
    r = S.report(["G4.3", "G1.8"], str(b), str(n))
    assert r["predicted"].round(6).tolist() == [0.03, 0.05] and r["check"].tolist() == ["adds up", "interaction"]
    assert S.winners_from_board(str(b)) == ["G4.3", "G1.8"]
