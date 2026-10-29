"""Dev scorer end to end on a tiny frozen version: identical rebuilds, the naive baseline scores exactly 1, the locked role
needs the flag (and adds a LOOKS row), compare refuses different versions and reports the unknown noise floor."""
import json

import pandas as pd
import pytest

from tideforge.devdata import ingest
from tideforge.eval import devscore as D
from tideforge.eval import public_models

from .conftest import build_version

LOOKS = "# Looks\n\n| Date | Experiment | Checkpoint | Benchmark | Why this look | Result |\n|---|---|---|---|---|---|\n| | | | | | |\n"


def _score(devroot, out, *extra, model="naive_time"):
    return D.main(["score", "--dev", str(devroot), "--version", "toy-v1", "--model", model, "--out", str(out), *extra])


def test_windows_rebuild_identically(tmp_path, monkeypatch, horizons):
    a = build_version(tmp_path, monkeypatch, sub="a")
    b = build_version(tmp_path, monkeypatch, sub="b")
    for f in ("tuning_fast.parquet", "tuning_score.parquet", "locked_fast.parquet", "covtasks_windows.parquet"):
        assert ingest.sha256_file(a / "toy-v1" / f) == ingest.sha256_file(b / "toy-v1" / f), f
    ma, mb = (json.loads((r / "toy-v1" / "manifest.json").read_text()) for r in (a, b))
    assert ma["sets"] == mb["sets"]


def test_naive_scores_exactly_one(devroot, tmp_path, tb):
    assert _score(devroot, tmp_path / "runs") == 0
    run, win = D.load_run(tmp_path / "runs" / "toy-v1" / "tuning_fast" / "naive_time")
    o = run["scores"]["main"]["overall"]
    assert o["MASE"] == [1.0, 1.0, 1.0] and o["CRPS"] == [1.0, 1.0, 1.0] and o["datasets"] == 3
    assert all(s["MASE"] == 1.0 for s in run["scores"]["main"]["slices"])
    assert set(win["hist"]) <= {"1-64", "65-512", "513-2048", "2049+"}


def test_cov_set_on_and_hidden(devroot, tmp_path, tb):
    assert _score(devroot, tmp_path / "runs", "--cov", model="naive") == 0
    run, win = D.load_run(tmp_path / "runs" / "toy-v1" / "tuning_cov" / "naive")
    assert set(win["mode"]) == {"on", "hidden"} and set(win["cov"]) == {"yes"}
    assert run["on_vs_hidden"]["ratio"] == pytest.approx(1.0)        # naive ignores covariates


def test_combined_weights(devroot, tmp_path, tb):
    _score(devroot, tmp_path / "runs")
    _score(devroot, tmp_path / "runs", "--cov")
    r = tmp_path / "runs" / "toy-v1"
    mr, mw = D.load_run(r / "tuning_fast" / "naive_time")
    cr, cw = D.load_run(r / "tuning_cov" / "naive_time")
    c = D.combined(mr, mw, cr, cw)
    assert c["datasets"] == 4 and c["MASE"][0] == pytest.approx(1.0)


def test_locked_refuses_without_the_flag(devroot, tmp_path):
    with pytest.raises(SystemExit, match="--locked"):
        _score(devroot, tmp_path / "runs", "--role", "locked")
    with pytest.raises(SystemExit, match="--why"):
        _score(devroot, tmp_path / "runs", "--role", "locked", "--locked")


def test_locked_adds_a_looks_row(devroot, tmp_path, tb):
    lf = tmp_path / "LOOKS.md"
    lf.write_text(LOOKS)
    assert _score(devroot, tmp_path / "runs", "--role", "locked", "--locked", "--experiment", "E0.5", "--why", "w",
                  "--looks-file", str(lf)) == 0
    rows = [ln for ln in lf.read_text().splitlines() if "dev locked toy-v1" in ln]
    assert len(rows) == 1 and "naive_time (main)" in rows[0]


def test_compare_and_rule5(devroot, tmp_path, tb, capsys):
    _score(devroot, tmp_path / "runs")
    _score(devroot, tmp_path / "runs", model="naive")
    r = tmp_path / "runs" / "toy-v1" / "tuning_fast"
    assert D.main(["compare", "--a", str(r / "naive"), "--b", str(r / "naive_time"), "--noise", str(tmp_path / "none.json"),
                   "--slice", "domain=Hydrology"]) == 0
    out = capsys.readouterr().out
    assert D.NOISE_UNKNOWN in out and "(2) own slice domain=Hydrology" in out and "(3)" in out
    noise = tmp_path / "e07.json"
    noise.write_text(json.dumps({"stat": "range", "dev": {"metrics": {"MASE": {"range": 0.5, "sd": 0.1}}}}))
    D.main(["compare", "--a", str(r / "naive"), "--b", str(r / "naive_time"), "--noise", str(noise)])
    assert "vs seed noise 0.5000 (range" in capsys.readouterr().out
    run = json.loads((r / "naive" / "run.json").read_text())
    run["version"] = "other"
    (r / "naive" / "run.json").write_text(json.dumps(run))
    with pytest.raises(SystemExit, match="different dev versions"):
        D.main(["compare", "--a", str(r / "naive"), "--b", str(r / "naive_time")])


def test_set_sha_is_checked(devroot, tmp_path):
    p = devroot / "toy-v1" / "tuning_fast.parquet"
    pd.read_parquet(p).head(1).to_parquet(p, index=False)
    with pytest.raises(RuntimeError, match="sha256"):
        _score(devroot, tmp_path / "runs")


def test_rank_corr():
    dev = {f"m{i}": float(i) for i in range(8)}
    c = D.rank_corr(dev, dict(dev))
    assert c["models"] == 8 and c["spearman"][0] == pytest.approx(1.0) and c["kendall"][0] == pytest.approx(1.0)
    c = D.rank_corr(dev, {k: -v for k, v in dev.items()})
    assert c["spearman"][0] == pytest.approx(-1.0)


def test_public_models_are_registered():
    assert {"chronos2", "timesfm25", "toto2_4m", "timestral5m"} <= set(public_models.REGISTRY)
    m = D.make_model("chronos2", device="cpu")              # lazy: no package is imported until the first forecast
    assert D.max_context(m) == 8192 and m.covariates == "past+known"


def test_save_forecasts_matches_the_scores(devroot, tmp_path, tb):
    import numpy as np
    from tideforge.eval import devmetrics as M
    assert _score(devroot, tmp_path / "runs", "--save-forecasts", model="naive") == 0
    d = tmp_path / "runs" / "toy-v1" / "tuning_fast" / "naive"
    run, win = D.load_run(d)
    f = D.load_forecasts(d / "forecasts.npz")
    assert len(f["q"]) == len(win)
    key = win.set_index(["dataset_id", "series_id", "window", "mode"])
    for i in range(0, len(win), max(1, len(win) // 7)):
        r = key.loc[(f["dataset_id"][i], f["series_id"][i], int(f["window"][i]), f["mode"][i])]
        assert f["q"][i].shape == (9, f["H"][i]) and len(f["y"][i]) == f["H"][i]
        assert np.all(np.diff(f["q"][i], axis=0) >= -1e-6)
    assert not (tmp_path / "runs" / "toy-v1" / "tuning_fast" / "naive_time" / "forecasts.npz").exists()


def test_forecasts_roundtrip(tmp_path):
    import numpy as np
    rng = np.random.default_rng(0)
    keep = []
    for i, H in enumerate((3, 7, 1)):
        it = dict(dataset_id=f"d{i % 2}", series_id=f"s{i}", window=i, S=7, y=rng.normal(size=H))
        keep.append((it, "on" if i else "main", np.sort(rng.normal(size=(9, H)), 0)))
    D.save_forecasts(tmp_path / "f.npz", keep)
    f = D.load_forecasts(tmp_path / "f.npz")
    for i, (it, m, q) in enumerate(keep):
        assert f["dataset_id"][i] == it["dataset_id"] and f["mode"][i] == m and f["S"][i] == 7
        assert np.allclose(f["q"][i], q, atol=1e-6) and np.allclose(f["y"][i], it["y"], atol=1e-6)


def test_noise_file_range_is_the_floor(devroot, tmp_path, tb, capsys):
    from tideforge.eval import noise as NZ
    _score(devroot, tmp_path / "runs")
    _score(devroot, tmp_path / "runs", model="naive")
    r = tmp_path / "runs" / "toy-v1" / "tuning_fast"
    probes = []
    for i, v in enumerate((0.30, 0.34, 0.31)):
        f = tmp_path / f"p{i}.csv"
        pd.DataFrame([dict(probe="P1", cell="c", mode="hidden", view="all", metric="MASE", value=v)]).to_csv(f, index=False)
        probes.append(str(f))
    out = tmp_path / "noise" / "e07.json"
    d = NZ.write(out, probe_csvs=probes)                        # probes first (dev set not built yet) ...
    assert "dev" not in d and NZ.floor(d, "MASE") is None
    assert d["probes"]["metrics"]["P1/c/hidden/all/MASE"]["range"] == pytest.approx(0.04)
    d = NZ.write(out, dev_runs=[str(r / "naive"), str(r / "naive_time")])   # ... dev later; probes kept
    m = d["dev"]["metrics"]["MASE"]
    assert d["stat"] == "range" and "P1/c/hidden/all/MASE" in d["probes"]["metrics"]
    assert m["range"] == pytest.approx(max(m["values"]) / min(m["values"]) - 1) and NZ.floor(d, "MASE") == m["range"]
    D.main(["compare", "--a", str(r / "naive"), "--b", str(r / "naive_time"), "--noise", str(out)])
    assert f"vs seed noise {m['range']:.4f} (range over seeds)" in capsys.readouterr().out


def test_noise_probe_part_and_floor(tmp_path):
    from tideforge.eval import noise as NZ
    probes = []
    for i, v in enumerate((0.30, 0.34, 0.31)):
        f = tmp_path / f"p{i}.csv"
        pd.DataFrame([dict(probe="P1", cell="c", mode="hidden", view="all", metric="MASE", value=v)]).to_csv(f, index=False)
        probes.append(str(f))
    d = NZ.write(tmp_path / "e07.json", probe_csvs=probes)
    s = d["probes"]["metrics"]["P1/c/hidden/all/MASE"]
    assert s["range"] == pytest.approx(0.04) and s["sd"] > 0 and NZ.floor(d, "MASE") is None
    assert NZ.floor({"dev": {"metrics": {"MASE": {"range": 0.02, "sd": 0.9}}}}, "MASE") == 0.02   # the range, never the sd
    with pytest.raises(SystemExit, match="at least 2"):
        NZ.write(tmp_path / "x.json", probe_csvs=probes[:1])
