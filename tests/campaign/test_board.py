"""Results board on a hand-checked fixture: 3 B0 seeds (E0.7) and two screened experiments, with run.json files and probe
CSVs as finish writes them.

Hand check (MASE; gain = 1 - A / B0; E0.7 ranges: dev 2%, dev_cov 3%):
  B0     dev 0.80 (all seeds), cov ON 0.90 (ON / hidden 0.95 -> gain +5%), domains Energy 0.80, Retail 0.80,
         P1 sql@on lagged_linear 1.0, nonlinear_u 2.0 (decoy control 5.0)
  G4.3   dev 0.76 -> +5.00% > 2%: PASS; Retail 0.81 / 0.80 = 1.0125 > 1%: FAIL  -> "no win"
         (default rule; P4 has no rows -> n/a)
  G1.13a dev 0.81 -> -1.25%, not worse than -2%: PASS; cov ON 0.855 -> +5.00% > 3%: PASS;
         P1 known cells 0.9, 1.8 -> mean 1.35 < 1.5: PASS -> "win, 1-seed screen: confirm with 3 seeds";
         P1 ratio 1.35 / 1.5 = 0.9000, worst domain Energy 0.812 / 0.80 = 1.0150
"""
import csv
import json

import pandas as pd

from tideforge.campaign import board as BD
from tideforge.train.finish import ROW

SHA, COV_SHA = "a" * 64, "b" * 64


def _run_json(path, MASE, CRPS, domains=None, on_hidden=None, cov=False):
    path.mkdir(parents=True, exist_ok=True)
    o = {"datasets": 3, "MASE": [MASE, 0, 0], "CRPS": [CRPS, 0, 0]}
    if cov:
        d = dict(version="dev-v1", set="cov", cov=True, set_sha256=COV_SHA,
                 scores={"on": {"overall": o, "slices": []}, "hidden": {"overall": o, "slices": []}},
                 on_vs_hidden={"ratio": on_hidden, "lo": 0, "hi": 0, "datasets": 3})
    else:
        sl = [dict(slice="domain", value=k, datasets=1, MASE=v, CRPS=v) for k, v in domains.items()]
        d = dict(version="dev-v1", set="fast", cov=False, set_sha256=SHA, scores={"main": {"overall": o, "slices": sl}})
    (path / "run.json").write_text(json.dumps(d))


def _probes(path, cells):
    path.mkdir(parents=True, exist_ok=True)
    rows = [dict(probe="P1", cell=c, control=ctl, mode="on", view="main", metric="sql", value=v) for c, ctl, v in cells]
    pd.DataFrame(rows).to_csv(path / "results.csv", index=False)


def fixture(tmp_path):
    rows = []

    def run(exp, seed, dev, cov_on, on_hidden, domains, p1, date="2026-11-01"):
        d = tmp_path / "runs" / f"{exp}-s{seed}"
        _run_json(d / "dev" / "dev-v1" / "tuning_fast" / "m5", dev, dev * 0.75, domains)
        _run_json(d / "dev" / "dev-v1" / "tuning_cov" / "m5", cov_on, cov_on * 0.75, on_hidden=on_hidden, cov=True)
        if p1:
            _probes(d / "probes", p1)
        rows.append(dict(date=date, run_id=f"{exp}-x-s{seed}-code", exp=exp, seed=seed, code="code", ckpt=str(d / "final.pt"),
                         dev_MASE=dev, run_dir=str(d)))

    b0p1 = [("lagged_linear@1024", False, 1.0), ("nonlinear_u@1024", False, 2.0), ("decoy@1024", True, 5.0)]
    for s in range(3):
        run("E0.7", s, 0.80, 0.90, 0.95, {"Energy": 0.80, "Retail": 0.80}, b0p1)
    run("G4.3", 0, 0.76, 0.90, 0.95, {"Energy": 0.70, "Retail": 0.81}, b0p1)
    run("G1.13a", 0, 0.81, 0.855, 0.90, {"Energy": 0.812, "Retail": 0.81},
        [("lagged_linear@1024", False, 0.9), ("nonlinear_u@1024", False, 1.8), ("decoy@1024", True, 5.0)])
    rows.append(dict(date="2026-11-01", run_id="G1.13a-x-s0-code", exp="G1.13a", seed=0, code="code",
                     ckpt=str(tmp_path / "runs" / "G1.13a-s0" / "ckpt_10B.pt"), dev_MASE=0.83))       # a milestone row
    res = tmp_path / "results.csv"
    with open(res, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ROW)
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in ROW} for r in rows)
    noise = tmp_path / "noise.json"
    noise.write_text(json.dumps({"dev": {"metrics": {"MASE": {"range": 0.02}}},
                                 "dev_cov": {"metrics": {"MASE": {"range": 0.03}}}}))
    return res, noise


EXPECTED = """# Results board (dev-v1, vs E0.7: 3 seeds; E0.7 range MASE: dev +2.00%, dev_cov +3.00%)

| Exp | Seeds | Dev MASE | Gain vs B0 | Dev-cov MASE | Gain vs B0 | ON vs hidden | Probe A/B0 | Worst domain | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| G1.13a | 1 | 0.8100 | -1.25% | 0.8550 | +5.00% | +10.00% | P1 0.9000 | Energy 1.0150 | win, 1-seed screen: confirm with 3 seeds |
| G4.3 | 1 | 0.7600 | +5.00% | 0.9000 | +0.00% | +5.00% | P4 n/a | Retail 1.0125 | no win |
"""


def test_board_reproduces_the_hand_checked_fixture(tmp_path):
    res, noise = fixture(tmp_path)
    b = BD.build(res, str(noise))
    assert BD.board_md(b) == EXPECTED
    g4 = next(r for r in b["rows"] if r["exp"] == "G4.3")
    assert g4["criteria"][0].startswith("PASS dev beats B0") and g4["criteria"][1].startswith("FAIL no domain slice")
    g1 = next(r for r in b["rows"] if r["exp"] == "G1.13a")
    assert [c.split()[0] for c in g1["criteria"]] == ["PASS", "PASS", "PASS"]
    assert "1.3500 vs B0 1.5000" in g1["criteria"][1]
    assert b["milestones"] == [dict(exp="G1.13a", seed="0", run_id="G1.13a-x-s0-code", points_B=10, dev_MASE="0.83",
                                    ckpt=str(tmp_path / "runs" / "G1.13a-s0" / "ckpt_10B.pt"))]


def test_missing_noise_is_unknown_and_other_version_not_comparable(tmp_path):
    res, _ = fixture(tmp_path)
    b = BD.build(res, None)
    assert all(r["verdict"] in ("unknown", "no win") for r in b["rows"])
    p = tmp_path / "runs" / "G4.3-s0" / "dev" / "dev-v1" / "tuning_fast" / "m5" / "run.json"
    d = json.loads(p.read_text())
    d["set_sha256"] = "c" * 64
    p.write_text(json.dumps(d))
    g4 = next(r for r in BD.build(res, str(tmp_path / "noise.json"))["rows"] if r["exp"] == "G4.3")
    assert g4["verdict"].startswith("not comparable")


def test_write_part_b_and_index_rows(tmp_path):
    res, noise = fixture(tmp_path)
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    tpl = (BD.REPO / "experiments" / "TEMPLATE.md").read_text(encoding="utf-8").replace("<ID>", "G1.13a")
    (repo / "experiments" / "G1.13a.md").write_text(tpl, encoding="utf-8")
    (repo / "EXPERIMENTS.md").write_text("## 9. Index\n\n" + BD.INDEX_HEAD + "\n|---|---|---|---|---|\n"
                                         "| G4.3 | 2026-10-28 | done | hand-written | [G4.3](experiments/G4.3.md) |\n", encoding="utf-8")
    argv = ["--results", str(res), "--noise", str(noise), "--out", str(tmp_path / "out"), "--write", "--repo", str(repo)]
    assert BD.main(argv) == 0
    log = (repo / "experiments" / "G1.13a.md").read_text(encoding="utf-8")
    assert log.count(BD.BEGIN) == 1 and "| Fast dev score (MASE) | 0.8000 | 0.8100 | -1.25% | +2.00% |" in log
    assert "| Metric | B0 | This run | Change | Noise floor |\n|---|---|---|---|---|\n| Fast dev score" in log
    assert "| Fast dev score | | | | |" not in log                   # the template's empty table is replaced
    assert "Rule verdict: **win, 1-seed screen: confirm with 3 seeds**" in log and "**Verdict:** win | no effect" in log
    idx = (repo / "EXPERIMENTS.md").read_text(encoding="utf-8")
    assert "| G1.13a | 2026-11-01 | screened | dev -1.25% vs B0, dev-cov +5.00%; win, 1-seed screen: confirm with 3 seeds | " \
           "[G1.13a](experiments/G1.13a.md) <!-- board -->|" in idx
    assert "| G4.3 | 2026-10-28 | done | hand-written |" in idx and idx.count("| G4.3 |") == 1     # hand row left alone
    assert BD.main(argv) == 0                                                                    # idempotent
    assert (repo / "EXPERIMENTS.md").read_text(encoding="utf-8") == idx
    assert (repo / "experiments" / "G1.13a.md").read_text(encoding="utf-8") == log
    assert (tmp_path / "out" / "board.md").read_text(encoding="utf-8") == EXPECTED
