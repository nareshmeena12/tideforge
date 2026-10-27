"""Groups-ON scoring: covariate encoding, the group built from a fev window, TIME multivariate groups, the two-mode table, the gate
and the comparison with the Colab log (fakes and a tiny model on CPU)."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from tidemark.real import ft3_core as C
from tidemark.real import score_groups as G
from test_real_eval import tiny_cfg

LOG_SAMPLE = """\
[  1/100] proenfo_gfc12                  h=168    ON: ft2_main_season  0.527   ft3_main_season  0.500* ft3_year_season  0.539  | HIDDEN: ft2_main_season  0.635   ft3_main_season  0.631* ft3_year_season  0.633  | ft2_main(old rule) 0.527/0.635  chronos2 0.518/0.568  (0s)
[  2/100] restaurant                     h=14     ON: ft2_main_season  0.700   ft3_main_season  0.710  | HIDDEN: ft2_main_season  0.700   ft3_main_season  0.710  | chronos2 0.5/0.5
      ---- SKILL so far,  10 tasks (higher is better) ----    ON: ft3_main_season  47.41  | HIDDEN: ft3_main_season  40.59
"""


def test_encode_column_numbers_and_categories():
    num = G.encode_column([np.array([1, 2, 3]), np.array([4.5, np.nan])])
    assert num[0].dtype == np.float32 and np.isnan(num[1][1])
    cat = G.encode_column([np.array(["0", "a", "b"], dtype=object), np.array(["a", None, ""], dtype=object)])
    assert list(cat[0]) == [0.0, 1.0, 2.0]
    assert cat[0][1] == cat[1][0] and np.isnan(cat[1][1]) and np.isnan(cat[1][2])          # same code for the same value, missing -> NaN
    assert G.encode_column([np.array([], dtype=float)])[0].shape == (0,)


class FakeWindow:
    def __init__(self, n=3, T=200, H=16, targets=("t1", "t2"), known=("k",), past=("p",), cat=False):
        rng = np.random.default_rng(0)
        self.target_columns, self.known_dynamic_columns, self.past_dynamic_columns = list(targets), list(known), list(past)
        self.horizon, self.n, self.T = H, n, T
        self.past = {c: [np.cumsum(rng.normal(size=T)) for _ in range(n)] for c in list(targets) + list(known) + list(past)}
        self.past["t1"][0][50:60] = np.nan
        if cat:
            self.past["k"] = [np.array(["a", "b"] * (T // 2), dtype=object) for _ in range(n)]
        self.future = {c: ([rng.normal(size=H) for _ in range(n)] if not cat else [np.array(["a", "b"] * (H // 2), dtype=object) for _ in range(n)])
                       for c in known}

    def get_input_data(self):
        return self.past, self.future


def fill_stub(a, S):
    a = np.asarray(a, dtype=np.float64)
    idx = np.where(np.isfinite(a), np.arange(len(a)), 0)
    np.maximum.accumulate(idx, out=idx)
    return a[idx].astype(np.float32)


def test_window_groups_structure():
    w = FakeWindow()
    gs = G.window_groups(w, 8192, 1, fill_stub)
    assert len(gs) == 3 and len(gs[0]["targets"]) == 2 and len(gs[0]["past"]) == 1 and len(gs[0]["known"]) == 1
    ctx, fut = gs[0]["known"][0]
    assert len(ctx) == 200 and len(fut) == 16 and np.isfinite(gs[0]["targets"][0]).all()          # the gap was filled by the stub
    assert len(G.window_groups(w, 100, 1, fill_stub)[0]["targets"][0]) == 100                    # context limited to max_ctx
    assert G.has_groups(w) and not G.has_groups(FakeWindow(targets=("t1",), known=(), past=()))
    cat = G.window_groups(FakeWindow(cat=True), 8192, 1, fill_stub)
    assert set(np.unique(cat[0]["known"][0][0])) == {0.0, 1.0} and set(np.unique(cat[0]["known"][0][1])) <= {0.0, 1.0}


def fake_task(window):
    return SimpleNamespace(quantile_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], seasonality=1, horizon=window.horizon,
                           iter_windows=lambda: iter([window]))


def test_fev_groups_predictions_with_a_tiny_model_and_the_plain_path():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = C.build_model(dict(cfg, dropout=0.0)).eval()
    w = FakeWindow(n=3, T=96, H=16)
    out = G.fev_groups_predictions(model, fake_task(w), torch.device("cpu"), C.forecast_groups, C.forecast_left_padded, fill_stub, 256, 8)
    assert len(out) == 1 and set(out[0]) == {"t1", "t2"} and len(out[0]["t1"]) == 3
    e = out[0]["t1"][0]
    assert len(e["predictions"]) == 16 and e["predictions"] == e["0.5"] and np.isfinite(e["0.1"]).all()
    assert all(a <= b for a, b in zip(e["0.1"], e["0.9"]))
    # a task without groups goes through the univariate path: the same numbers as forecast_left_padded
    w1 = FakeWindow(n=2, T=96, H=16, targets=("t1",), known=(), past=())
    on = G.fev_groups_predictions(model, fake_task(w1), torch.device("cpu"), C.forecast_groups, C.forecast_left_padded, fill_stub, 256, 8)
    ser = [fill_stub(y, 1) for y in w1.past["t1"]]
    _, q = C.forecast_left_padded(model, ser, 16, torch.device("cpu"), max_ctx=256, batch_size=8)
    q = np.sort(q.astype(np.float64), axis=1)
    assert on[0]["t1"][1]["0.9"] == q[1, 8].tolist()


def test_group_covariates_change_the_forecast_of_a_group_model():
    cfg = tiny_cfg()
    torch.manual_seed(1)
    model = C.build_model(dict(cfg, dropout=0.0)).eval()
    for p in model.parameters():                              # a fresh model has a zero-initialised group pathway: nudge every matrix
        if p.dim() > 1:
            p.data += 0.05 * torch.randn_like(p)
    w = FakeWindow(n=2, T=96, H=16, targets=("t1",))
    with_cov = G.fev_groups_predictions(model, fake_task(w), torch.device("cpu"), C.forecast_groups, C.forecast_left_padded, fill_stub, 256, 8)
    w2 = FakeWindow(n=2, T=96, H=16, targets=("t1",), known=(), past=())
    w2.past["t1"] = w.past["t1"]
    without = G.fev_groups_predictions(model, fake_task(w2), torch.device("cpu"), C.forecast_groups, C.forecast_left_padded, fill_stub, 256, 8)
    assert with_cov[0]["t1"][0]["0.5"] != without[0]["t1"][0]["0.5"]


def test_time_quantiles_shapes_and_modes():
    cfg = tiny_cfg()
    torch.manual_seed(0)
    model = C.build_model(dict(cfg, dropout=0.0)).eval()
    for p in model.parameters():
        if p.dim() > 1:
            p.data += 0.05 * torch.randn_like(p)
    rng = np.random.default_rng(0)
    inputs = [{"target": np.cumsum(rng.normal(size=(3, 120)), axis=1)} for _ in range(2)]
    args = (model, inputs, 16, 1, 3, torch.device("cpu"))
    kw = dict(fill=fill_stub, forecast_groups=C.forecast_groups, forecast_plain=C.forecast_left_padded, max_ctx=256, batch=8)
    on = G.time_quantiles(*args, "on", **kw)
    off = G.time_quantiles(*args, "off", **kw)
    assert on.shape == off.shape == (2, 9, 3, 16) and (np.diff(on, axis=1) >= -1e-9).all()
    assert not np.allclose(on, off)                                              # groups ON sees the sibling variates
    uni = [{"target": rng.normal(size=100).cumsum()} for _ in range(2)]
    a = G.time_quantiles(model, uni, 16, 1, 1, torch.device("cpu"), "on", **kw)
    b = G.time_quantiles(model, uni, 16, 1, 1, torch.device("cpu"), "off", **kw)
    assert a.shape == (2, 9, 16) and np.array_equal(a, b)                         # univariate datasets: the same forecast in both modes


def test_log_parsing_and_comparison(tmp_path):
    log = tmp_path / "log.txt"
    log.write_text(LOG_SAMPLE)
    ref = G.parse_fev_log(log)
    assert ref == {"proenfo_gfc12": (0.500, 0.631), "restaurant": (0.710, 0.710)}
    root = tmp_path / "R"
    for tag, vals in (("ft3mainon_s0", {"proenfo_gfc12": 0.508, "restaurant": 0.7101}), ("ft3main_s0", {"proenfo_gfc12": 0.631, "restaurant": 0.7099})):
        d = root / "eval" / "fev" / tag
        d.mkdir(parents=True)
        for t, r in vals.items():
            (d / f"{t}.json").write_text(json.dumps({"task_name": t, "test_error": r}))
    pd.DataFrame({"model_name": "seasonal_naive", "task_name": ["proenfo_gfc12", "restaurant"], "test_error": [1.0, 1.0]}).to_csv(
        root / "eval" / "fev_naive.csv", index=False)
    lines = G.compare_with_log(root, "ft3main", log)
    assert "ON: 2 tasks compared" in lines[0] and "1 differ by more than 0.005" in lines[0]
    assert any("proenfo_gfc12" in l and "+0.0080" in l for l in lines)
    assert any("OFF (HIDDEN): 2 tasks compared" in l and "0 differ" in l for l in lines)


def test_two_mode_table_and_gate(tmp_path):
    board = tmp_path / "eval" / "board"
    board.mkdir(parents=True)
    idx = ["ft3mainon_s0", "ft3main_s0", "p10bon_s0", "p10b_s0"]
    pd.DataFrame({"MASE (norm.)": [0.680, 0.687, 0.670, 0.678], "CRPS (norm.)": [0.571, 0.575, 0.565, 0.570]}, index=idx).to_csv(board / "time_board.csv")
    pd.DataFrame({"skill_score": [0.4378, 0.4209, 0.4405, 0.4230], "win_rate": [0.6, 0.58, 0.61, 0.59]}, index=idx).to_csv(board / "fev_board.csv")
    df = G.mode_table(tmp_path, ["ft3main", "p10b"], "ft3main")
    r = df[df.checkpoint == "p10b"].iloc[0]
    assert r["fev skill ON"] == pytest.approx(44.05) and r["fev skill ON chg"] == pytest.approx(0.27)
    assert r["TIME MASE ON chg"] == pytest.approx(100 * (0.670 / 0.680 - 1)) and r["fev win OFF"] == pytest.approx(59.0)
    lines = G.gate_lines(df, "ft3main")
    assert lines[-1] == "GATE: PASS" and sum("PASS" in l for l in lines[:-1]) == 6
    df.loc[0, "fev skill ON"] = 43.2
    assert G.gate_lines(df, "ft3main")[-1].startswith("GATE: FAIL")
