"""D2 fixable error on a fixture, D3 repairs exact on constructed cases, the shared features (synthetic fixtures)."""
import json

import numpy as np
import pandas as pd
import pytest

from tideforge.diagnose import _boot, features as FE, fixable as FX, repair as RP

S, H = 12, 48
t = np.arange(H, dtype=np.float64)
OFF = np.linspace(-1.0, 1.0, 9)[:, None]                      # quantile offsets around the median


def truth():
    c = np.array([3, 1, -2, -4, -1, 0, 2, 5, 1, -1, -3, -1], np.float64)
    c = c - c.mean()
    return 10 + 0.2 * t + c[np.arange(H) % S]


def good(y):
    return y[None, :] + OFF


@pytest.mark.parametrize("name,bad", [
    ("level", lambda y: good(y) + 4.0),
    ("trend", lambda y: good(y) + 0.3 * (t - t.mean())),
    ("amplitude", lambda y: good(y) + 1.5 * RP.seasonal(y, S)),
    ("phase", lambda y: good(y) - RP.seasonal(y, S) + np.roll(RP.seasonal(y, S), 3)),
])
def test_each_repair_is_exact_on_its_own_error(name, bad):
    y = truth()
    q = bad(y)
    e_good = RP.crps(good(y), y)
    r = RP.repairs(q, y, S)
    assert np.allclose(r[name], good(y), atol=1e-9), name
    assert abs(RP.crps(r[name], y) - e_good) < 1e-9
    assert RP.crps(q, y) > e_good + 1e-6
    others = [k for k in RP.REPAIRS if k != name and k != "width"]
    assert any(RP.crps(r[k], y) > e_good + 1e-6 for k in others)       # not every repair fixes every error


def test_width_repair():
    """The truth scatters around the median as the right forecast says (a stratified sample of N(0, 1)); the forecast's spread
    is half of it. The CRPS-best scale is about 2, and the grid point 2 is hit exactly."""
    from scipy.stats import norm
    m = 10 + 0.2 * t
    z = norm.ppf((np.arange(H) + 0.5) / H)
    y = m + np.random.default_rng(3).permutation(z)
    q_true = m[None, :] + norm.ppf(RP.QL)[:, None]
    q_bad = m[None, :] + norm.ppf(RP.QL)[:, None] / 2
    r = RP.repairs(q_bad, y, S)
    assert np.allclose(r["width"], q_true, atol=1e-9)
    assert RP.crps(r["width"], y) < RP.crps(q_bad, y)


def test_spikes_repair():
    y = truth()
    ys = y.copy()
    ys[[7, 30]] += [40.0, -35.0]
    q = good(y)                                                    # right everywhere except the two spikes it missed
    r = RP.repairs(q, ys, S)
    assert np.allclose(r["spikes"], good(ys), atol=1e-9)
    assert RP.spikes_of(ys).sum() == 2


def test_seasonal_projection_is_exact_and_kills_lines():
    y = truth()
    s = RP.seasonal(y, S)
    assert np.allclose(RP.seasonal(s, S), s) and np.allclose(RP.seasonal(3 + 0.5 * t, S), 0, atol=1e-9)


def win_table(rng, n_ds=4, n_w=30):
    rows = []
    for d in range(n_ds):
        for w in range(n_w * (d + 1)):                            # datasets of different sizes: equal weight must hold
            rows.append(dict(dataset_id=f"d{d}", series_id=f"s{w}", window=w, mode="main", domain="E" if d < 2 else "F",
                             freq="H", term="short", hist="65-512", cov="no", ctx_len=200, CRPS=1.0, CRPS_naive=1.0,
                             MASE=1.0, MASE_naive=1.0))
    return pd.DataFrame(rows)


def test_fixable_shares_equal_dataset_weight():
    rng = np.random.default_rng(0)
    ours = win_table(rng)
    ref_a, ref_b = ours.copy(), ours.copy()
    ours.loc[ours.dataset_id == "d0", "CRPS"] = 2.0               # d0: a reference is 1.0 better on every window
    ours.loc[ours.dataset_id == "d3", "CRPS"] = 2.0               # d3 has 4x the windows: must weigh the same as d0
    ref_b["CRPS"] = 1.5
    df = FX.fixable(ours, {"a": ref_a, "b": ref_b}, "CRPS")
    ds = FX.by_dataset(df).set_index("dataset_id")
    assert ds.loc["d0", "share"] == pytest.approx(0.5) and ds.loc["d3", "share"] == pytest.approx(0.5)
    assert ds.loc["d1", "share"] == 0 and set(df["best_ref"]) == {"a"}
    assert df["share"].sum() == pytest.approx(1.0)


def test_fixable_refuses_locked(tmp_path):
    d = tmp_path / "r"
    d.mkdir()
    (d / "run.json").write_text(json.dumps(dict(role="locked")))
    with pytest.raises(ValueError, match="tuning"):
        FX.load_dev_run(d)


def test_fixable_fev_table(tmp_path):
    t = pd.DataFrame({"ours": [1.2, 0.9, 1.0], "A": [1.0, 1.0, 0.5], "B": [1.1, 0.8, 0.9]}, index=["x|t1", "x|t2", "x|t3"])
    t.to_csv(tmp_path / "t.csv")
    df = FX.from_fev_table(tmp_path / "t.csv", "ours", ["A", "B"]).set_index("task")
    assert df.loc["x|t3", "fix"] == pytest.approx(0.5) and df.loc["x|t1", "best_ref"] == "A"
    assert df["share"].sum() == pytest.approx(1.0)


def test_attribute_on_constructed_windows():
    y = truth()
    fc = dict(dataset_id=np.array(["d0", "d0"]), series_id=np.array(["a", "b"]), window=np.array([0, 0]),
              mode=np.array(["main", "main"]), S=np.array([S, S]), H=np.array([H, H]), q=[good(y) + 4.0, good(y) + 4.0],
              y=[y, y])
    win = pd.DataFrame(dict(dataset_id=["d0", "d0"], series_id=["a", "b"], window=[0, 0], mode=["main", "main"],
                            domain=["E", "E"], CRPS_naive=[1.0, 1.0]))
    df, sh = RP.attribute(fc, win, worst=1.0)
    lv = sh[(sh.scope == "all") & (sh.repair == "level")].iloc[0]
    e_good = RP.crps(good(y), y)
    assert lv.share_removed == pytest.approx(1 - e_good / RP.crps(good(y) + 4.0, y))


def test_window_features_on_known_shapes():
    rng = np.random.default_rng(1)
    L = 480
    tt = np.arange(L)
    seas = np.sin(2 * np.pi * tt / 24)
    f = FE.window_features(10 + seas + 0.05 * rng.normal(size=L), 24)
    assert f["seas_strength"] > 0.9 and f["period_ratio"] == 24 / L and f["zero_share"] == 0
    f = FE.window_features(0.05 * tt + 0.1 * rng.normal(size=L), 24)
    assert f["trend_strength"] > 0.9 and f["seas_strength"] < 0.2
    x = np.zeros(L)
    x[::8] = 3
    f = FE.window_features(x, 1)
    assert f["zero_share"] == pytest.approx(7 / 8) and f["adi"] == pytest.approx(8.0)
    x = rng.normal(size=L)
    x[-96:] = 5 + 3 * x[-96:]
    f = FE.window_features(x, 1)
    assert f["late_level"] > 1.5 and f["late_scale"] > 0.8
    x = rng.normal(size=L)
    x[[50, 200, 300]] += 30
    assert FE.window_features(x, 1)["spikiness"] == pytest.approx(3 / L)
    assert set(FE.SYNTH_COMPARABLE) <= set(FE.window_features(rng.normal(size=L), 7))


def test_gen_audit_uses_the_shared_features():
    from pathlib import Path
    src = (Path(__file__).resolve().parents[2] / "tools" / "gen_audit.py").read_text(encoding="utf-8")
    assert "feats = shape_features" in src and "def feats(" not in src
