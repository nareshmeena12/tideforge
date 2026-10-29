"""Public-model adapters (E0.5) without any model package: the registry imports clean, every adapter's input / output shapes
and values on its fake backend (public_models.fake, which has each library's call signature), covariates reach the models
that take them and are refused by the others, no quantile is interpolated, and the subprocess runner gives exactly what
in-process scoring gives (same quantiles, same windows.parquet)."""
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from tideforge.eval import devscore as D
from tideforge.eval import public_models as P
from tideforge.eval.public_models import base, fake, fevcheck
from tideforge.eval.public_models.runner import Remote
from tideforge.probes.common import Cell

PACKAGES = ("chronos", "timesfm", "toto2", "tirex", "tirex2", "timestral")


def test_registry_imports_without_extras():
    code = ("import sys; import tideforge.eval.public_models as P; [P.adapter_class(n) for n in P.SPECS]; "
            "import tideforge.eval.public_models.runner, tideforge.eval.public_models.fevcheck; "
            f"bad = [m for m in {PACKAGES!r} if m in sys.modules]; assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], check=True)
    assert set(P.SPECS) == {"chronos2", "timesfm25", "toto2_4m", "toto2_22m", "toto2_2p5b", "tirex", "tirex2", "timestral5m",
                            "timestral28m"}
    for s in P.SPECS.values():
        assert len(s.revision) == 40 and s.covariates in ("past+known", "none") and s.max_context > 0
    assert {n for n, s in P.SPECS.items() if s.venv} == {"toto2_4m", "toto2_22m", "toto2_2p5b", "tirex2"}
    assert {n for n, s in P.SPECS.items() if s.covariates == "none"} == {"timesfm25", "toto2_4m", "toto2_22m", "toto2_2p5b",
                                                                          "tirex"}


def _cell(n=3, k=2, L=40, H=7, P_=1, F=2, seed=0, nan=True):
    rng = np.random.default_rng(seed)
    a = {"targets": rng.normal(size=(n, k, L)).astype(np.float32), "future": np.zeros((n, k, H)),
         "past": rng.normal(size=(n, P_, L)).astype(np.float32),
         "known_ctx": rng.normal(size=(n, F, L)).astype(np.float32), "known_fut": rng.normal(size=(n, F, H)).astype(np.float32)}
    if nan:
        a["targets"][0, 0, 5:9] = np.nan
        a["targets"][1, -1, -1] = np.nan                    # the last point missing: the fakes use the last finite one
    return Cell("t", a, 7, modes=("on", "hidden"))


@pytest.mark.parametrize("name", sorted(P.SPECS))
def test_adapter_shapes_and_values_on_a_fake(name):
    m = P.make(name, "cpu", backend=fake.backend_for(name))
    c = _cell()
    q = m.predict_cell(c, "hidden")
    n, k, L = c.arrays["targets"].shape
    assert q.shape == (n, k, 9, c.H) and np.isfinite(q).all()
    want = np.stack([fake.expected(t, c.H) for t in c.arrays["targets"]])
    np.testing.assert_allclose(q, want, rtol=0, atol=1e-5)
    if m.covariates == "none":
        with pytest.raises(ValueError, match="hidden mode only"):
            m.predict_cell(c, "on")
    else:
        q_on = m.predict_cell(c, "on")
        want_on = np.stack([fake.expected(t, c.H, f) for t, f in zip(c.arrays["targets"], c.arrays["known_fut"])])
        np.testing.assert_allclose(q_on, want_on, rtol=0, atol=1e-5)
        assert not np.allclose(q_on, q)                    # the known covariates reached the model


def test_univariate_k1_and_varied_horizon():
    for name in ("timesfm25", "tirex", "chronos2"):
        m = P.make(name, "cpu", backend=fake.backend_for(name))
        for H in (1, 130):                                  # 130: TimesFM recompiles above its 128 max_horizon
            c = _cell(n=2, k=1, H=H, nan=False)
            assert m.predict_cell(c, "hidden").shape == (2, 1, 9, H)


def test_select_q9_refuses_to_interpolate():
    lv = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    q = np.arange(len(lv), dtype=float)[None, :, None]
    assert base.select_q9(q, lv).ravel().tolist() == list(range(1, 10))
    with pytest.raises(ValueError, match="refusing to interpolate"):
        base.select_q9(q[:, :5], (0.1, 0.25, 0.5, 0.75, 0.9))


def test_remote_runner_equals_in_process():
    name = "toto2_4m"                                       # a venv model; here the "venv" is this interpreter
    local = P.make(name, "cpu", backend=fake.backend_for(name))
    r = Remote(name, "cpu", python=sys.executable, fake=True)
    try:
        assert r.max_context == local.max_context and r.covariates == local.covariates
        for seed in (0, 1):
            c = _cell(seed=seed)
            assert np.array_equal(r.predict_cell(c, "hidden"), local.predict_cell(c, "hidden"))
        with pytest.raises(RuntimeError, match="hidden mode only"):        # the worker's error comes back
            r.predict_cell(_cell(), "on")
        assert r.predict_cell(_cell(seed=2), "hidden").shape == (3, 2, 9, 7)  # and the worker lives on
    finally:
        r.close()


def test_runner_writes_the_same_windows_parquet(tmp_path, monkeypatch, devroot, tb):
    def score(model, out):
        monkeypatch.setattr(D, "make_model", lambda *a, **k: model)
        assert D.main(["score", "--dev", str(devroot), "--version", "toy-v1", "--model", "tirex2", "--out", str(out)]) == 0
        return pd.read_parquet(out / "toy-v1" / "tuning_fast" / "tirex2" / "windows.parquet")
    a = score(P.make("tirex2", "cpu", backend=fake.backend_for("tirex2")), tmp_path / "local")
    r = Remote("tirex2", "cpu", python=sys.executable, fake=True)
    try:
        b = score(r, tmp_path / "remote")
    finally:
        r.close()
    pd.testing.assert_frame_equal(a, b)
    assert len(a) > 0


def test_devscore_cov_hidden_only_for_models_without_covariates(tmp_path, monkeypatch, devroot, tb):
    import json
    monkeypatch.setattr(D, "make_model", lambda *a, **k: P.make("tirex", "cpu", backend=fake.backend_for("tirex")))
    assert D.main(["score", "--dev", str(devroot), "--version", "toy-v1", "--model", "tirex", "--cov", "--out", str(tmp_path)]) == 0
    run = json.loads((tmp_path / "toy-v1" / "tuning_cov" / "tirex" / "run.json").read_text())
    assert list(run["scores"]) == ["hidden"] and run["covariates"] == "none (hidden only)" and "on_vs_hidden" not in run


class _Window:
    """A fev window stand-in (get_input_data + column lists), for the fev-check plumbing."""
    def __init__(self, n=4, L=30, H=5, cov=True):
        rng = np.random.default_rng(0)
        self.target_columns = ["a", "b"]
        self.past_dynamic_columns = ["p"] if cov else []
        self.known_dynamic_columns = ["k"] if cov else []
        lens = [L, L, L - 3, L]
        self.past = {c: [rng.normal(size=lens[i]) for i in range(n)] for c in ("a", "b", "p", "k")}
        self.past["k"] = [np.array(["x", "y"] * L)[:lens[i]] for i in range(n)]            # a categorical known column
        self.future = {"k": [np.array(["y", "x"] * H)[:H] for _ in range(n)]}

    def get_input_data(self):
        return self.past, self.future


class _Task:
    def __init__(self, cov=True):
        self.quantile_levels, self.horizon, self.seasonality, self.task_name = list(fake.Q9), 5, 1, "toy"
        self.cov = cov

    def iter_windows(self):
        return [_Window(cov=self.cov), _Window(cov=self.cov)]


@pytest.mark.parametrize("name", ["timestral5m", "timesfm25"])
def test_fevcheck_predictions_format(name):
    m = P.make(name, "cpu", backend=fake.backend_for(name))
    pr = fevcheck.task_predictions(m, _Task())
    assert len(pr) == 2 and set(pr[0]) == {"a", "b"} and len(pr[0]["a"]) == 4
    f = pr[0]["a"][2]
    assert set(f) == {"predictions", *map(str, fake.Q9)} and len(f["predictions"]) == 5
    assert f["predictions"] == f["0.5"] and f["0.1"][0] < f["0.9"][0]


def test_fevcheck_compare_rule():
    pub = pd.DataFrame({"test_error": [1.0] * 5, "MASE": [1.0] * 5}, index=list(fevcheck.TASKS))
    ok = {t: {"test_error": 1.019, "MASE": 1.0} for t in fevcheck.TASKS}
    assert fevcheck.compare(ok, pub)["passed"]
    bad = dict(ok, ETT_1W={"test_error": 1.021, "MASE": 1.0})
    r = fevcheck.compare(bad, pub)
    assert not r["passed"] and [x["task"] for x in r["rows"] if not x["pass_"]] == ["ETT_1W"]
    assert not fevcheck.compare({t: ok[t] for t in fevcheck.TASKS[:4]}, pub)["passed"]     # a missing task never passes
