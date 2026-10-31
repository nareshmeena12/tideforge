"""Probe core: deterministic files that are never overwritten, registry check, fingerprint, metrics, bootstrap / paired
comparison, the zero-mass estimate, the predict interface, the runner and report --compare (oracle and naive, CPU)."""
import json

import numpy as np
import pandas as pd
import pytest

import tideforge.probes.p01_covariates as P1
from tideforge.probes import common, metrics as M, run as R
from tideforge.probes.models import Naive, Oracle, predict


def test_save_load_and_never_overwrite(tmp_path, monkeypatch):
    ps = P1.build(4)
    meta = common.save(ps, tmp_path)
    back = common.load(tmp_path, "P1", 1)
    assert [c.name for c in back.cells] == [c.name for c in ps.cells]
    assert np.array_equal(back.cells[0].arrays["targets"], ps.cells[0].arrays["targets"])
    with pytest.raises(FileExistsError):
        common.save(ps, tmp_path)
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"P1": {"1": {"content_sha256": "0" * 64}}}))
    monkeypatch.setattr(common, "REGISTRY", reg)
    with pytest.raises(RuntimeError, match="registry"):
        common.load(tmp_path, "P1", 1)
    assert meta["fingerprint"] == common.fingerprint(ps)


def test_fingerprint_ignores_last_bits_but_not_values():
    x = np.array([1.2345678, -3.0, np.nan, 0.0])
    assert common.sig6(x) == common.sig6(x * (1 + 1e-12))
    assert common.sig6(x) != common.sig6(x * 1.001)


def test_metrics_and_bootstrap():
    rng = np.random.default_rng(0)
    y = rng.normal(size=(200, 1, 16))
    q = common.normal_q(np.zeros((200, 1, 16)), np.ones((200, 1, 16)))
    cov = M.Metric("cov80", M.coverage(q, y))
    v, lo, hi = M.bootstrap(cov)
    assert lo < 0.8 < hi and lo < v < hi
    c = M.pit_counts(q, y)
    assert M.pit_tv(c) < 0.05 and M.pit_tv(M.pit_counts(q * 0.2, y)) > 0.2
    a = M.Metric("sql", M.sql(q, y, np.ones((200, 1))))
    b = M.Metric("sql", M.sql(q * 3, y, np.ones((200, 1))))
    d, lo, hi = M.paired(a, b)
    assert d < 0 and hi < 0                                  # A clearly better, the paired CI excludes 0
    assert M.paired(a, a)[1:] == (0.0, 0.0)


def test_p_zero_from_quantiles():
    q = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2], float)[None, None, :, None]       # Poisson(1)-like: true P(0) = 0.37
    assert M.p_zero(q)[0, 0, 0] == pytest.approx(0.35)
    assert M.p_zero(np.zeros((1, 1, 9, 1)))[0, 0, 0] == 0.9                       # capped
    assert M.p_zero(np.ones((1, 1, 9, 1)) * 3)[0, 0, 0] == 0.0


def test_predict_interface():
    t = np.sin(np.arange(240) * 2 * np.pi / 24)
    q = predict(Naive(), t, None, 24, S=24)
    assert q.shape == (1, 9, 24) and np.allclose(q[0, 4], t[-24:], atol=1e-6)
    q2 = predict(Naive(), t, {"known": [(t, t[:24])]}, 24, S=24)
    assert q2.shape == (1, 9, 24)


def test_run_report_compare(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "probe_ids", lambda spec: ["P1"])
    monkeypatch.setattr(P1, "N_PER_CELL", 8)
    root, out = tmp_path / "sets", tmp_path / "out"
    assert R.main(["build", "--root", str(root), "--n", "8"]) == 0
    R.run(Oracle(), ["P1"], root, out, say=lambda *a: None)
    R.run(Naive(), ["P1"], root, out, say=lambda *a: None)
    df = pd.read_csv(out / "results.csv")
    assert set(df["model"]) == {"oracle", "naive"} and (df["ci_lo"] <= df["value"]).all() and (df["value"] <= df["ci_hi"]).all()
    assert "lagged_linear@1024" in R.table(df[df["model"] == "oracle"])
    cmp = R.compare(out, "oracle", "naive")
    row = cmp[(cmp["cell"] == "lagged_linear@1024") & (cmp["metric"] == "sql") & (cmp["mode"] == "on")].iloc[0]
    assert row["diff"] < 0 and row["ci_hi"] < 0


def test_probe_set_checks_content_not_file_bytes(tmp_path):
    ps = P1.build(4)
    meta = common.save(ps, tmp_path)
    d = tmp_path / "P1" / "v1"
    with np.load(d / "set.npz") as z:
        arrays = {k: z[k] for k in z.files}
    np.savez(d / "set.npz", **arrays)                                        # other bytes (uncompressed), same content
    assert common.load(tmp_path, "P1", 1).cells[0].n == ps.cells[0].n
    arrays[sorted(arrays)[0]].flat[0] += 1
    np.savez(d / "set.npz", **arrays)
    with pytest.raises(RuntimeError, match="content sha256"):
        common.load(tmp_path, "P1", 1)
    assert meta["content_sha256"] != meta["file_sha256"]


def test_registry_holds_perlmutters_build_and_local_builds_match_it():
    """Perlmutter's build (2026-10-31) is the reference: every registry fingerprint equals a fresh local build's."""
    import json as _json
    from tideforge.probes.run import module, probe_ids
    from conftest import REAL_REGISTRY
    reg = _json.loads(REAL_REGISTRY.read_text(encoding="utf-8"))
    assert sorted(reg, key=lambda k: int(k[1:])) == probe_ids("all")
    for p in probe_ids("all"):
        mod = module(p)
        r = reg[p][str(mod.VERSION)]
        ps = mod.build(mod.N_PER_CELL, mod.SEED)
        assert (r["seed"], r["n_per_cell"], r["cells"]) == (mod.SEED, ps.n_per_cell, len(ps.cells)), p
        assert common.fingerprint(ps) == r["fingerprint"], p


def test_migrate_old_meta(tmp_path, monkeypatch):
    import hashlib
    ps = P1.build(4)
    meta = common.save(ps, tmp_path)
    d = tmp_path / "P1" / "v1"
    old = {k: v for k, v in meta.items() if k not in ("content_sha256", "file_sha256")}
    old["sha256"] = hashlib.sha256((d / "set.npz").read_bytes()).hexdigest()       # the pre-2026-10-31 layout
    (d / "meta.json").write_text(json.dumps(old))
    monkeypatch.setattr(common, "REGISTRY", tmp_path / "none.json")
    with pytest.raises(RuntimeError, match="migrate"):
        common.load(tmp_path, "P1", 1)
    m = common.migrate(tmp_path, "P1", 1)
    assert m["content_sha256"] == meta["content_sha256"] and common.load(tmp_path, "P1", 1).cells[0].n == 4
    reg = tmp_path / "reg.json"
    reg.write_text(json.dumps({"P1": {"1": {"fingerprint": "0" * 64}}}))
    monkeypatch.setattr(common, "REGISTRY", reg)
    with pytest.raises(RuntimeError, match="fingerprint"):
        common.load(tmp_path, "P1", 1)
