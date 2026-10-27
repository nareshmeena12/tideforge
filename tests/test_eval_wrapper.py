"""tideforge.eval.score (argument passing to score_groups, variant in the tags, LOOKS rows) and tideforge.eval.looks."""
from pathlib import Path

import pandas as pd
import pytest

import tidemark.real.score_groups as G
import tidemark.real.ft3_core as C
from tideforge.eval import looks, score

LOOKS_TEMPLATE = """# Looks at fev-bench and TIME

Every time a model is scored on fev-bench or TIME, add a row, newest first. Nothing is selected on TIME.

| Date | Experiment | Checkpoint | Benchmark | Why this look | Result |
|---|---|---|---|---|---|
| | | | | | |
"""


def test_append_looks(tmp_path):
    p = tmp_path / "LOOKS.md"
    p.write_text(LOOKS_TEMPLATE)
    r1 = looks.format_row(experiment="E0.2", checkpoint="a_s0", benchmark="TIME", why="w", result="MASE 1", date="2026-10-27")
    r2 = looks.format_row(experiment="E0.2", checkpoint="b|s0", benchmark="fev-bench", why="w", result="x", date="2026-10-28")
    assert looks.append_looks([r1], p) == [r1]
    assert looks.append_looks([r2, r1], p) == [r2]                   # r1 is already there: not a second look
    lines = p.read_text().splitlines()
    i = lines.index(looks.HEADER)
    assert lines[i + 2:] == [r2, r1]                                  # newest first, placeholder gone
    assert "b/s0" in r2


def test_score_passes_variant_tags_to_score_groups(monkeypatch):
    seen = {}

    def fake_main(argv):
        seen["argv"] = argv
        seen["forecast"] = C.forecast_left_padded                 # what the scorer would call inside the variant
        return 0

    monkeypatch.setattr(G, "main", fake_main)
    orig = C.forecast_left_padded
    assert score.main(["score", "--root", "R", "--ckpt", "ft3main=/x/ft3_main.pt", "--variant", "nofill_mask", "--smoke"]) == 0
    assert seen["argv"] == ["score", "--root", "R", "--modes", "on,off", "--ckpt", "ft3mainnfmask=/x/ft3_main.pt", "--smoke"]
    assert seen["forecast"] is not orig and C.forecast_left_padded is orig
    score.main(["score", "--root", "R", "--ckpt", "ft3main=/p", "--skip-fev", "--num-proc", "2"])
    assert seen["argv"] == ["score", "--root", "R", "--modes", "on,off", "--ckpt", "ft3mainbase=/p", "--skip-fev", "--num-proc", "2"]
    assert seen["forecast"] is orig


def test_look_rows_carry_variant_and_mode():
    df = pd.DataFrame([{"checkpoint": "ft3mainnfmask", "TIME MASE ON": 0.6812, "TIME CRPS ON": 0.5723, "fev skill ON": 43.5,
                        "fev win ON": 60.1, "TIME MASE OFF": 0.69, "TIME CRPS OFF": 0.58, "fev skill OFF": 42.0}])
    rows = score.look_rows(df, ["ft3mainnfmask"], "E0.3", "delete vs mask")
    assert len(rows) == 4
    assert all("variant nofill_mask" in r and "| E0.3 |" in r for r in rows)
    assert any("ft3mainnfmaskon_s0" in r and "| fev-bench |" in r and "skill 43.50, win 60.10" in r for r in rows)
    assert any("ft3mainnfmask_s0 (variant nofill_mask, groups OFF)" in r and "MASE 0.6900, CRPS 0.5800" in r for r in rows)


def test_gate_is_the_six_numbers():
    assert G.GATE == {"ON": {"TIME MASE": (0.680, 0.002), "TIME CRPS": (0.571, 0.002), "fev skill": (43.78, 0.2)},
                      "OFF": {"TIME MASE": (0.687, 0.002), "TIME CRPS": (0.575, 0.002), "fev skill": (42.09, 0.2)}}


def test_repo_looks_file_has_the_header():
    assert looks.HEADER in Path(looks.LOOKS).read_text(encoding="utf-8").splitlines()


def test_compare_board(tmp_path):
    cols = "model,arm,seed,benchmark,metric,task,window,err\n"
    (tmp_path / "ours.csv").write_text(cols + "ft3mainbaseon_s0,ft3mainbaseon,0,fev,test_error,t1,all,0.5\n"
                                       "ft3mainbase_s0,ft3mainbase,0,time,MASE,d/short,all,0.70\n")
    (tmp_path / "ref.csv").write_text(cols + "ft3mainon_s0,ft3mainon,0,fev,test_error,t1,all,0.5\n"
                                      "ft3main_s0,ft3main,0,time,MASE,d/short,all,0.71\nft3main_s0,ft3main,0,fev,test_error,t9,all,1\n")
    j = score.compare_board(tmp_path / "ours.csv", tmp_path / "ref.csv")
    assert len(j) == 2 and j.iloc[0]["task"] == "d/short" and abs(j.iloc[0]["abs_diff"] - 0.01) < 1e-12
    assert j.iloc[1]["abs_diff"] == 0


def test_board_summary():
    import pandas as pd
    j = pd.DataFrame(dict(arm="ft3main", benchmark="time", metric="MASE", task=["a", "b"], window="all",
                          err_ours=[0.9, 1.1], err_ref=[1.0, 1.0]))
    s = score.board_summary(j)
    r = s.iloc[0]
    assert r["tasks"] == 2 and abs(r["ratio"] - (0.9 * 1.1) ** 0.5) < 1e-12 and r["share_better"] == 0.5
    assert r["ci_lo"] <= r["ratio"] <= r["ci_hi"]
