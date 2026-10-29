"""Campaign runner: the Phase 2 screen expands in order, packs 4 runs per regular node, generation is deterministic, done runs
are skipped (code version ignored by default, required with --strict-code; milestone rows never count), QOS limits hold."""
import csv

import pytest

from tideforge import slurm
from tideforge.campaign import runner as R
from tideforge.train import recipe as RC

QUIET = dict(log=lambda *a: None)


def _results(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "exp", "seed", "code", "ckpt"])
        w.writeheader()
        w.writerows(rows)
    return path


def test_phase2_screen_size_and_order():
    p = R.plan("phase2_screen", **QUIET)
    assert len(p["runs"]) == 55 and len(p["todo"]) == 55 and len(p["jobs"]) == 14
    assert [r.exp for r in p["runs"][:3]] == ["G1.1", "G1.2", "G3.1a"] and p["runs"][-1].exp == "G5.4b"
    ex = [r.exp for r in p["runs"]]
    assert ex[ex.index("G3.2") + 1] == "G3.4"
    assert all(len(j["runs"]) == 4 for _, j in p["jobs"][:-1]) and len(p["jobs"][-1][1]["runs"]) == 3
    assert all(j["qos"] == "regular" and j["gpus"] == 4 for _, j in p["jobs"])
    assert not {"G4.3f", "G1.13b", "G1.13c", "G1.14", "E0.7", "E0.8"} & {r.exp for r in p["runs"]}


def test_generation_is_deterministic(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "GENERATED", tmp_path / "gen")
    a = R.plan("phase2_screen", write=True, **QUIET)
    d = tmp_path / "gen" / "phase2_screen"
    files = {p.name: p.read_bytes() for p in d.glob("*.sbatch")}
    b = R.plan("phase2_screen", write=True, **QUIET)
    assert a["texts"] == b["texts"] and len(files) == 14
    assert files == {p.name: p.read_bytes() for p in d.glob("*.sbatch")}
    t = a["texts"]["phase2_screen_01"]
    assert "--exp G1.1 --seed 0" in t and "--exp G3.1b --seed 0" in t and "#SBATCH -q regular" in t


def test_done_runs_are_skipped(tmp_path):
    key, key2 = R.run_key(RC.load("G1.1"), 0), R.run_key(RC.load("G1.2"), 0)
    res = _results(tmp_path / "results.csv", [
        dict(run_id=f"{key}-oldcode1234", exp="G1.1", seed=0, code="oldcode1234", ckpt="/r/x/final.pt"),
        dict(run_id=f"{key2}-oldcode1234", exp="G1.2", seed=0, code="oldcode1234", ckpt="/r/y/ckpt_10B.pt")])  # a milestone
    p = R.plan("phase2_screen", results=res, **QUIET)
    todo = {r.exp for r in p["todo"]}
    assert "G1.1" not in todo and "G1.2" in todo and len(p["todo"]) == 54
    assert len(R.plan("phase2_screen", results=res, strict_code=True, **QUIET)["todo"]) == 55
    assert R.is_done(R.Run("G1.1", 0, key, 1e11), R.done_rows(res), code="oldcode1234")


def test_qos_and_campaign_checks(tmp_path):
    with pytest.raises(ValueError):
        slurm.check(dict(qos="regular", gpus=5, cpus=128, time="02:00:00"))
    with pytest.raises(ValueError):
        slurm.check(dict(qos="regular", gpus=4, cpus=128, time="49:00:00"))
    c = tmp_path / "dup.yaml"
    c.write_text("name: dup\nruns: [G1.1, {exp: G1.1, seeds: [0]}]\n")
    with pytest.raises(ValueError, match="twice"):
        R.expand(R.load_campaign(c))
    bad = tmp_path / "x.yaml"
    bad.write_text("name: y\nruns: [G1.1]\n")
    with pytest.raises(ValueError, match="file name"):
        R.load_campaign(bad)
