"""E0.3 inference variants: baseline == the unswapped scorers, originals restored even on an exception, what each variant feeds
the model, and the NaN audit counts (tiny model / fakes on CPU)."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tidemark.eval.score_twins as S
import tidemark.real.ft3_core as C
import tidemark.real.score_groups as G
from tideforge.eval import nan_audit
from tideforge.eval.nan_variants import (forecast_left_padded_masked, lane_array_masked, strip_leading_nan, use_variant,
                                         variant_of)

DEV = torch.device("cpu")


def tiny_model(seed=0):
    cfg = dict(C.CONFIG, d_model=64, d_kv=16, n_heads=4, d_ff=128, n_layers=2, max_ctx=256, group_lanes=8, dropout=0.0)
    torch.manual_seed(seed)
    m = C.build_model(cfg).eval()
    for p in m.parameters():            # the group pathway starts at zero: nudge every matrix so groups change the forecast
        if p.dim() > 1:
            p.data += 0.05 * torch.randn_like(p)
    return m


def gappy(n, T, seed=0, D=None):
    rng = np.random.default_rng(seed)
    shape = (n, T) if D is None else (n, D, T)
    y = np.cumsum(rng.normal(size=shape), axis=-1) + 10 * np.sin(np.arange(T) * 2 * np.pi / 12)
    y[..., :5] = np.nan                                     # leading gap
    y[..., 40:47] = np.nan                                  # interior block gap
    y[..., 70] = np.nan                                     # single missing point
    return y


def originals():
    return S.fill_season, C.forecast_left_padded, C._lane_array


# ---------------------------------------------------------------------------------------------------- swapping
def test_baseline_equals_the_unswapped_functions():
    m = tiny_model()
    ys = [S.fill_season(y, 12) for y in gappy(5, 150)]
    _, q0 = C.forecast_left_padded(m, ys, 20, DEV, max_ctx=256, batch_size=4)
    inputs = [{"target": t} for t in gappy(3, 150, 1, D=3)]
    kw = dict(max_ctx=256, batch=8)
    t0 = {mode: G.time_quantiles(m, inputs, 20, 12, 3, DEV, mode, S.fill_season, C.forecast_groups, C.forecast_left_padded, **kw)
          for mode in ("on", "off")}
    before = originals()
    with use_variant("baseline"):
        assert originals() == before
        ys1 = [S.fill_season(y, 12) for y in gappy(5, 150)]
        _, q1 = C.forecast_left_padded(m, ys1, 20, DEV, max_ctx=256, batch_size=4)
        t1 = {mode: G.time_quantiles(m, inputs, 20, 12, 3, DEV, mode, S.fill_season, C.forecast_groups, C.forecast_left_padded,
                                     **kw) for mode in ("on", "off")}
    assert np.array_equal(q0, q1)
    for mode in t0:
        assert np.array_equal(t0[mode], t1[mode]), mode


@pytest.mark.parametrize("variant", ["baseline", "nofill_delete", "nofill_mask"])
def test_originals_restored_after_an_exception(variant):
    before = originals()
    with pytest.raises(RuntimeError, match="scoring failed"):
        with use_variant(variant):
            if variant != "baseline":
                assert originals() != before
            raise RuntimeError("scoring failed")
    assert originals() == before


def test_unknown_variant():
    with pytest.raises(KeyError):
        with use_variant("nope"):
            pass
    assert variant_of("ft3mainnfmask") == "nofill_mask" and variant_of("ft3mainbaseon_s0") == "baseline"
    assert variant_of("ft3mainnfdel_s0") == "nofill_delete" and variant_of("ft3main") is None


def test_variants_swap_what_they_say():
    with use_variant("nofill_delete"):
        assert S.fill_season is strip_leading_nan
        assert C.forecast_left_padded is originals()[1] and C._lane_array is originals()[2]
    with use_variant("nofill_mask"):
        assert (S.fill_season, C.forecast_left_padded, C._lane_array) == (strip_leading_nan, forecast_left_padded_masked,
                                                                         lane_array_masked)


# ---------------------------------------------------------------------------------------------------- what reaches the model
class Spy:
    """Stands in for the decoder: records (ctx, mask) and returns zeros."""
    patch_len, n_q, is_decoder, max_pass_patches = 32, 9, True, 32

    def __init__(self):
        self.calls = []

    def eval(self):
        return self

    def predict_horizon(self, ctx, mask, horizon):
        self.calls.append((ctx.clone(), mask.clone()))
        return torch.zeros(ctx.shape[0], horizon, self.n_q)


def test_strip_leading_nan():
    y = np.array([np.nan, np.nan, 1.0, np.nan, 3.0])
    assert np.array_equal(strip_leading_nan(y, 2), np.array([1.0, np.nan, 3.0], np.float32), equal_nan=True)
    assert np.isnan(strip_leading_nan([np.nan, np.nan], 2)).all()
    assert np.array_equal(strip_leading_nan([1.0, 2.0], 2), np.array([1.0, 2.0], np.float32))


def test_delete_vs_mask_inputs():
    y = np.arange(1, 41, dtype=np.float32)
    y[[10, 11, 30]] = np.nan
    s_del, s_msk = Spy(), Spy()
    C.forecast_left_padded(s_del, [y], 8, DEV, max_ctx=256)
    forecast_left_padded_masked(s_msk, [y], 8, DEV, max_ctx=256)
    (cd, md), (cm, mm) = s_del.calls[0], s_msk.calls[0]
    assert int(md.sum()) == 37                                     # deleted: 37 points, the later ones moved 3 steps earlier
    assert int(mm.sum()) == 37 and mm.shape[1] == 64               # masked: 40 positions kept, 3 unobserved
    obs = mm[0, -40:].numpy()
    assert (~obs).nonzero()[0].tolist() == [10, 11, 30]
    assert np.array_equal(cm[0, -40:].numpy(), np.nan_to_num(y, nan=0.0))
    assert cd[0, -1].item() == 40.0 and cm[0, -1].item() == 40.0


def test_masked_equals_original_without_nan():
    m = tiny_model()
    rng = np.random.default_rng(3)
    ys = [rng.normal(size=n).cumsum().astype(np.float32) for n in (20, 64, 130)]
    a = C.forecast_left_padded(m, ys, 40, DEV, max_ctx=256, batch_size=2)
    b = forecast_left_padded_masked(m, ys, 40, DEV, max_ctx=256, batch_size=2)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    a = C.forecast_left_padded(m, [[np.nan, np.nan]], 8, DEV)
    b = forecast_left_padded_masked(m, [[np.nan, np.nan]], 8, DEV)
    assert np.array_equal(a[1], b[1])                              # an all-NaN series: zeros(1) in both


def test_lane_array_masked():
    a = np.array([np.nan, 1.0, 2.0, np.nan, 4.0], np.float32)
    v0, o0 = C._lane_array(a, 7)
    v1, o1 = lane_array_masked(a, 7)
    assert o0.tolist() == [False, False, False, True, True, True, True] and v0[5] == 2.0         # forward-filled
    assert o1.tolist() == [False, False, False, True, True, False, True] and v1[5] == 0.0         # unobserved
    b = np.array([3.0, 1.0, 2.0], np.float32)
    assert all(np.array_equal(x, y) for x, y in zip(C._lane_array(b, 5), lane_array_masked(b, 5)))


def test_nofill_mask_changes_the_forecast_only_where_there_are_gaps():
    m = tiny_model()
    clean = np.cumsum(np.random.default_rng(5).normal(size=150)).astype(np.float32)
    gap = gappy(1, 150, 2)[0]

    def run():
        ys = [S.fill_season(y, 12) for y in (clean, gap)]
        return C.forecast_left_padded(m, ys, 20, DEV, max_ctx=256)[1]

    base = run()
    with use_variant("nofill_delete"):
        dele = run()
    with use_variant("nofill_mask"):
        mask = run()
    assert np.array_equal(base[0], dele[0]) and np.array_equal(base[0], mask[0])
    assert not np.array_equal(base[1], dele[1]) and not np.array_equal(base[1], mask[1]) and not np.array_equal(dele[1], mask[1])


# ---------------------------------------------------------------------------------------------------- audit
def test_target_counts():
    ys = [np.array([np.nan, 1, 2, np.nan, 4, 5, np.nan, 7.0]), np.array([np.nan, np.nan]), np.arange(5.0)]
    c = nan_audit.target_counts(ys, 3, 8192, S.fill_season)
    assert c == dict(n_series=3, n_values=15, nan_leading=3, nan_interior=2, series_interior_nan=1, nan_after_fill=2,
                     all_nan_series=1)
    assert nan_audit.target_counts([np.r_[np.nan, np.arange(10.0)]], 1, 4, S.fill_season)["n_values"] == 4   # last max_ctx


class FakeWindow:
    def __init__(self):
        self.target_columns = ["t"]
        self.past_dynamic_columns = ["p"]
        self.known_dynamic_columns = ["k"]
        t = [np.array([1.0, np.nan, 3.0, 4.0]), np.array([np.nan, 2.0, 3.0, 4.0])]
        self._past = {"t": t, "p": [np.array([1.0, np.nan, 2.0, 2.0]), np.array([1.0, 1.0, 1.0, 1.0])],
                      "k": [np.array(["a", "b", None, "a"], object), np.array(["a", "a", "a", "a"], object)]}
        self._fut = {"k": [np.array(["b", None], object), np.array(["a", "a"], object)]}

    def get_input_data(self):
        return self._past, self._fut


def test_fev_task_counts():
    task = SimpleNamespace(seasonality=2, target_columns=["t"], iter_windows=lambda: iter([FakeWindow(), FakeWindow()]))
    c = nan_audit.fev_task_counts(task, 8192, S.fill_season, G.encode_column)
    assert c["n_windows"] == 2 and c["n_series"] == 4 and c["n_values"] == 16
    assert (c["nan_leading"], c["nan_interior"], c["nan_after_fill"]) == (2, 2, 0)
    assert (c["past_cov_values"], c["past_cov_nan"]) == (16, 2)
    assert (c["known_ctx_values"], c["known_ctx_nan"], c["known_fut_values"], c["known_fut_nan"]) == (16, 2, 8, 2)


def test_time_term_counts():
    inputs = [{"target": gappy(1, 100)[0]}, {"target": gappy(1, 100, D=2)[0]}]
    c = nan_audit.time_term_counts(inputs, 12, 8192, S.fill_season)
    assert c["n_windows"] == 2 and c["n_series"] == 3 and c["nan_leading"] == 15 and c["nan_interior"] == 24
    assert c["nan_after_fill"] == 0 and c["past_cov_nan"] == 0


def test_fev_cov_detail_and_report():
    task = SimpleNamespace(seasonality=2, target_columns=["t"], iter_windows=lambda: iter([FakeWindow()]))
    c = nan_audit.fev_task_counts(task, 8192, S.fill_season, G.encode_column)
    assert (c["past_cov_lead"], c["past_cov_inner"], c["past_cov_allnan_series"]) == (0, 1, 0)
    assert (c["known_ctx_lead"], c["known_ctx_inner"]) == (0, 1) and c["known_fut_allnan_series"] == 0
    assert c["cov_nan_columns"] == "p(past): 1; k(known ctx): 1; k(known future): 1"
    import pandas as pd
    df = pd.DataFrame([dict(benchmark="fev", task="gvar", **{k: v for k, v in c.items()}),
                       dict(benchmark="fev", task="x", **dict(c, nan_after_fill=0, all_nan_series=2))])
    df.loc[0, ["nan_after_fill", "all_nan_series"]] = [30820, 230]
    r = nan_audit.report(df)
    assert "gvar" in r and "all-NaN series in tasks with no NaN after fill: x (2)" in r and "totals: past 2 of 16" in r


# ---------------------------------------------------------------------------------------------------- cov_mask
def _cov_groups(n=3, T=150, H=20, gaps=False, seed=4):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        y = np.cumsum(rng.normal(size=T)).astype(np.float64)
        k = rng.normal(size=T + H)
        p = rng.normal(size=T)
        if gaps:
            k[30:90] = np.nan                     # a long missing stretch in the known history (as in favorita)
            k[T + 5:T + 9] = np.nan               # and in the known future
            p[100:110] = np.nan
        out.append(dict(targets=[y], past=[p], known=[(k[:T], k[T:])]))
    return out


def test_covmask_copy_equals_forecast_groups_without_nan():
    from tideforge.eval.nan_variants import forecast_groups_covmask
    m = tiny_model()
    g = _cov_groups()
    a = C.forecast_groups(m, g, 20, DEV, max_ctx=256)
    b = forecast_groups_covmask(m, g, 20, DEV, max_ctx=256)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(a, b))


def test_cov_mask_changes_only_covariate_gaps():
    m = tiny_model()
    clean, gappy_ = _cov_groups(), _cov_groups(gaps=True)
    base_clean = C.forecast_groups(m, clean, 20, DEV, max_ctx=256)
    base_gap = C.forecast_groups(m, gappy_, 20, DEV, max_ctx=256)
    fill_before = S.fill_season
    with use_variant("cov_mask"):
        assert S.fill_season is fill_before                                 # targets: baseline fill
        cm_clean = C.forecast_groups(m, clean, 20, DEV, max_ctx=256)
        cm_gap = C.forecast_groups(m, gappy_, 20, DEV, max_ctx=256)
    assert all(np.array_equal(x[1], y[1]) for x, y in zip(base_clean, cm_clean))
    assert not all(np.array_equal(x[1], y[1]) for x, y in zip(base_gap, cm_gap))
    assert variant_of("ft3maincmask") == "cov_mask" and variant_of("ft3maincmaskon_s0") == "cov_mask"


def test_cov_mask_restores_on_exception():
    before = (C._lane_array, C.forecast_groups, S.fill_season)
    with pytest.raises(RuntimeError):
        with use_variant("cov_mask"):
            assert C.forecast_groups is not before[1]
            raise RuntimeError("x")
    assert (C._lane_array, C.forecast_groups, S.fill_season) == before
