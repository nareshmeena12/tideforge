"""The window cache reproduces fev's windows exactly; the release guards locked tasks."""
from pathlib import Path

import numpy as np
import pytest

from tideforge.score import release as RL
from tideforge.score import windows as W


def test_cache_equals_fev_windows(release, tmp_path):
    import fev
    specs = RL.Release(release).tasks("full", "tuning")
    caches = W.build(specs, tmp_path / "cache", workers=1)
    assert [c.name for c in caches] == [s["kw"]["task_name"] for s in specs]
    for s, c in zip(specs, caches):
        t = fev.Task(**s["kw"])
        r = 0
        for wi, w in enumerate(t.iter_windows(num_proc=0)):
            past, fut = w.get_input_data()
            gt = w.get_ground_truth()
            for k in range(len(past)):
                assert c.r_win[r] == wi
                for d, col in enumerate(c.info["target_columns"]):
                    assert np.array_equal(c.past(r, d), np.asarray(past[col][k], float), equal_nan=True)
                    tab = gt.data.table.column(col)[k].as_py()          # metrics read fev's table, not python access
                    assert np.array_equal(c.y_true[r, :, d], np.asarray(tab, float), equal_nan=True)
                for col in c.info["known"]:
                    if col in c.cov:
                        assert np.array_equal(c.cov_future(r, col), np.asarray(fut[col][k], np.float32))
                r += 1
        assert r == c.info["n_rows"]
    by = {c.name: c for c in caches}
    assert by["a__d/short/target"].info["n_items"] == 4 and np.isnan(by["a__d/short/target"].y_true).any()
    assert by["c__d/short/joint"].info["target_columns"] == ["x", "y"] and by["c__d/short/joint"].has_groups
    assert by["c__d/short/target"].info["target_columns"] == ["target"]
    assert by["b__h/short/cov"].info["str_cov"] == ["hol"] and by["b__h/short/cov"].has_groups
    assert not by["b__h/short/target"].has_groups


def test_cache_resumes(release, tmp_path):
    specs = RL.Release(release).tasks("fast", "tuning")
    W.build(specs, tmp_path / "c", workers=1)
    mt = {p: p.stat().st_mtime_ns for p in (tmp_path / "c").rglob("info.json")}
    W.build(specs, tmp_path / "c", workers=1)
    assert mt == {p: p.stat().st_mtime_ns for p in (tmp_path / "c").rglob("info.json")}


def test_locked_needs_flag(release):
    r = RL.Release(release)
    with pytest.raises(RL.LockedError):
        r.tasks("full", "locked")
    assert [s["row"]["split"] for s in r.tasks("full", "locked", locked=True)] == ["locked"]
    assert all(not Path(s["kw"]["dataset_path"]).relative_to(release).parts[0] == "locked" for s in r.tasks("full", "tuning"))


def test_ledger(tmp_path):
    p = tmp_path / "l.csv"
    RL.ledger_append(dict(model_id="m", tier="full", version="v1"), p)
    RL.ledger_append(dict(model_id="n", tier="core", version="v1"), p)
    lines = p.read_text().splitlines()
    assert lines[0].startswith("date,model_id") and len(lines) == 3
