"""CLI end to end on the fake release (CPU, tiny model): outputs and tags, groups both, naive skill 0, locked guard + ledger +
looks row, fev-bench mode (our skill = fev.leaderboard's), aggregation weights / bootstrap / compare."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml

from tideforge.score import aggregate as AG
from tideforge.score import release as RL
from tideforge.score import run as RN


def score(release, tmp, *extra):
    return RN.main(["score", "--release-root", str(release.parent), "--version", release.name, "--tier", "full",
                    "--workers", "1", "--device", "cpu", "--out", str(tmp / "res"), "--n-boot", "50", *extra])


def test_score_both_modes_outputs(release, tmp_path, tiny_ckpt, monkeypatch):
    monkeypatch.setenv("TF_ROOT", str(tmp_path / "tfroot"))
    assert score(release, tmp_path, "--model", str(tiny_ckpt), "--model-id", "tiny", "--groups", "both") == 0
    base = tmp_path / "res" / "tiny" / release.name / "full_tuning"
    for mode in ("on", "off"):
        d = base / f"groups_{mode}"
        df = pd.read_csv(d / "tasks.csv")
        assert len(df) == 5 and set(AG.TAGS) <= set(df.columns)
        assert {"MASE", "WQL", "SQL", "naive_MASE", "rel_MASE"} <= set(df.columns)
        assert set(df["source_group"]) == {"g1", "c"} and set(df["cov_split"]) == {"covariate", "non-covariate"}
        st = json.loads((d / "stats.json").read_text())
        assert st["windows"] > 0 and st["windows_per_s"] > 0 and st["cache_gib"] >= 0
        assert st["rescale"] is True and st["n_rel_MASE_gt20"] == len(st["tasks_rel_MASE_gt20"])
        md = (d / "summary.md").read_text()
        assert "By covariate vs non-covariate tasks" in md and "95% CI" in md
        assert pd.read_csv(d / "failure_map.csv")["task"].iloc[0] in set(df["task"])
    on, off = (pd.read_csv(base / f"groups_{m}" / "tasks.csv").set_index("task") for m in ("on", "off"))
    assert np.isclose(on.loc["a__d/short/target", "MASE"], off.loc["a__d/short/target", "MASE"], rtol=1e-4)
    assert not (tmp_path / "tfroot" / "looks_pending.md").exists()            # tuning runs are not looks


def test_naive_skill_is_zero(release, tmp_path):
    assert score(release, tmp_path, "--model", "seasonal_naive") == 0
    df = pd.read_csv(tmp_path / "res" / "seasonal_naive" / release.name / "full_tuning" / "groups_off" / "tasks.csv")
    assert np.allclose(df["rel_test_error"], 1.0) and AG.skill(df)[0] == 0.0


def test_locked_guard_ledger_and_looks(release, tmp_path, monkeypatch):
    monkeypatch.setenv("TF_ROOT", str(tmp_path / "tfroot"))
    monkeypatch.setattr(RL, "LEDGER", tmp_path / "ledger.csv")
    assert score(release, tmp_path, "--model", "seasonal_naive") == 0
    assert not (tmp_path / "ledger.csv").exists()
    out = tmp_path / "res" / "seasonal_naive" / release.name
    assert not (out / "full_locked").exists()
    assert score(release, tmp_path, "--model", "seasonal_naive", "--locked") == 0
    led = pd.read_csv(tmp_path / "ledger.csv")
    assert len(led) == 1 and led["tier"][0] == "full" and (out / "full_locked").exists()
    assert "tfbench v9.9.9 locked full" in (tmp_path / "tfroot" / "looks_pending.md").read_text()


def test_fevbench_mode_matches_fev_leaderboard(release, tmp_path, tiny_ckpt, monkeypatch):
    monkeypatch.setenv("TF_ROOT", str(tmp_path / "tfroot"))
    ev = tmp_path / "eval_root"
    specs = RL.Release(release).tasks("full", "tuning")
    (ev / "eval" / "data" / "fev").mkdir(parents=True)
    (ev / "eval" / "data" / "fev" / "tasks.yaml").write_text(yaml.safe_dump({"tasks": [s["kw"] for s in specs]}))
    assert RN.main(["score", "--model", str(tiny_ckpt), "--model-id", "tiny", "--fev-bench", "--eval-root", str(ev),
                    "--workers", "1", "--device", "cpu", "--out", str(tmp_path / "res"), "--n-boot", "20"]) == 0
    d = tmp_path / "res" / "tiny" / "fev-bench" / "fev-bench_fev-bench" / "groups_on"
    st = json.loads((d / "stats.json").read_text())
    df = pd.read_csv(d / "tasks.csv")
    assert np.isclose(st["skill (model + naive)"], AG.skill(df)[0], atol=1e-9)
    assert "fev-bench" in (tmp_path / "tfroot" / "looks_pending.md").read_text()


def test_weights_bootstrap_compare(tmp_path):
    df = pd.DataFrame(dict(task=list("abcd"), source_group=["g", "g", "h", "k"], weight=[0.5, 0.5, 1, 1],
                           rel_test_error=[0.5, 2.0, 0.8, 0.9], rel_MASE=1.0, rel_WQL=1.0, rel_SQL=1.0))
    s, used, _ = AG.skill(df)
    assert np.isclose(s, 1 - np.exp((0.5 * np.log(0.5) + 0.5 * np.log(2) + np.log(0.8) + np.log(0.9)) / 3))
    assert AG.bootstrap(df, n=200, seed=1) == AG.bootstrap(df, n=200, seed=1)
    lo, hi = AG.bootstrap(df, n=200, seed=1)
    assert lo <= s <= hi
    for name, f in (("a", 1.0), ("b", 0.9)):
        d = tmp_path / name
        d.mkdir()
        x = df.assign(test_error=[1.0 * f, 2.0 * f, 3.0 * f, 4.0 * f], naive_test_error=1.0, band="daily", domain="x",
                      horizon_class="short", covariates="none", cov_split="non-covariate", multivariate=False)
        x["rel_test_error"] = x["test_error"]
        x.to_csv(d / "tasks.csv", index=False)
    md = AG.compare([tmp_path / "a", tmp_path / "b"], n_boot=50)
    assert "0.9" in md.split("against the first")[1]
