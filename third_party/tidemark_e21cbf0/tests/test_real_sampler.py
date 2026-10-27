"""Mixed real + synthetic batches: shapes, the loss mask, shares, exact resume (tiny model and data on CPU)."""
import numpy as np
import pytest
import torch

from tidemark.real import ft3_core as C
from tidemark.real import model as M
from tidemark.real import sampler as SM
from tidemark.real import store as ST
from test_real_eval import tiny_cfg
from test_real_store import write_arrow

nan = float("nan")
DEV = torch.device("cpu")


def make_store(tmp_path, spec):
    """spec: {name: (n_series, length, nan_frac)} -> RealStore (series are random walks with NaN holes)."""
    rng = np.random.default_rng(0)
    for name, (ns, ln, nf) in spec.items():
        rows = []
        for _ in range(ns):
            v = np.cumsum(rng.normal(size=ln)).tolist()
            for i in np.flatnonzero(rng.random(ln) < nf):
                v[i] = nan
            v[0], v[-1] = 1.0, 2.0
            rows.append(v)
        f = tmp_path / f"{name}.arrow"
        write_arrow(f, rows)
        ST.convert_dataset(name, [f], tmp_path / "shards" / name)
    return ST.RealStore(tmp_path / "shards", sorted(spec))


def cfg3():
    return dict(tiny_cfg(), pack_rows=24)


def mixed(tmp_path, spec=None, targets=None, seed=5, **mix):
    spec = spec or {"a": (30, 600, 0.0), "b": (30, 300, 0.2)}
    store = make_store(tmp_path, spec)
    C.set_group_options(cfg3())
    mc = SM.MixConfig(rows=24, real_fraction=1 / 3, **mix)
    return SM.MixedBatcher(cfg3(), DEV, store, targets or {"a": 0.5, "b": 0.5}, mc, seed=seed), store


def test_split_rows_and_shapes(tmp_path):
    mb, _ = mixed(tmp_path)
    assert (mb.rows_syn, mb.rows_real) == (16, 8) and SM.MixConfig(rows=288).split_rows(8) == (192, 96)
    b = mb.next_batch()
    R, W = 24, 256
    assert b["x"].shape == (R, W) and b["mask"].shape == (R, W) and b["loss_ok"].shape == (R, W) and b["blank"].shape == (R, W)
    s1, s2 = b["parts"]
    assert s1 == 16 * 8 and s2 == 8 * 8 and b["slot_valid"].numel() == s1 + s2
    assert b["seg_patch"][:16].max() < s1 <= b["seg_patch"][16:].min() and b["gid"][16:].min() >= s1       # real slots shifted
    assert b["points"] == b["points_syn"] + b["points_real"] and b["points_real"] > 0
    assert b["loss_ok"][:16].all()                                                # synthetic: everything is scored


def test_the_masked_model_trains_on_a_mixed_batch(tmp_path):
    mb, _ = mixed(tmp_path)
    b = mb.next_batch()
    model = M.build_masked_model(dict(cfg3(), dropout=0.0))
    out = M.packed_forward_masked(model, b)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)


def test_missing_points_are_observed_but_not_scored_and_not_counted(tmp_path):
    mb, _ = mixed(tmp_path, {"b": (40, 400, 0.3)}, {"b": 1.0})
    tot_missing = tot = 0
    for _ in range(4):
        b = mb.next_batch()
        r_mask, r_loss = b["mask"][16:], b["loss_ok"][16:]
        assert (r_loss <= r_mask).all()                                           # a scored point is always an observed one
        tot_missing += int((r_mask & ~r_loss).sum())
        tot += int(r_mask.sum())
        assert b["points_real"] == int(r_loss.sum())                              # VALID points only
    assert 0.1 < tot_missing / tot < 0.6                                          # roughly the 30% of the data (horizon rule pushes it down)
    assert mb.real.realised()["b"]["missing_share"] > 0.1


def test_horizon_rule_skips_windows_with_a_missing_tail(tmp_path):
    n = 50
    rows = []
    for i in range(n):
        v = np.cumsum(np.random.default_rng(i).normal(size=500)).tolist()
        if i % 2:                                                                 # odd series: the last 200 points are missing
            v[250:480] = [nan] * 230
        v[-1] = 1.0
        rows.append(v)
    write_arrow(tmp_path / "h.arrow", rows)
    ST.convert_dataset("h", [tmp_path / "h.arrow"], tmp_path / "shards" / "h")
    store = ST.RealStore(tmp_path / "shards", ["h"])
    C.set_group_options(cfg3())
    rb = SM.RealBatcher(cfg3(), DEV, store, {"h": 1.0}, SM.MixConfig(rows=24, horizon_points=64), rows=8, seed=1)
    for _ in range(3):
        rb.next_batch()
    assert rb.rejected > 0 and rb.forced == 0                                     # bad windows were redrawn, none forced


def test_short_series_only_serve_short_contexts_and_the_controller_tracks_targets(tmp_path):
    spec = {"short": (60, 80, 0.0), "long": (60, 900, 0.0)}
    mb, store = mixed(tmp_path, spec, {"short": 0.3, "long": 0.7})
    for _ in range(40):
        mb.next_batch()
    r = mb.real.realised()
    assert mb.real.no_dataset == 0
    # series of 80 points can only serve contexts up to 80, which carry few of the points: the target 0.3 is not reachable, the
    # controller gets as close as the data allows (it is a best effort, reported in the realised shares)
    assert r["long"]["realised"] > 0.6 and 0.1 < r["short"]["realised"] < 0.3
    assert mb.real.a[mb.real.names.index("short")] > 0                                                         # the controller pushes the under-served dataset up


def test_exact_resume_of_the_data_stream(tmp_path):
    mb, store = mixed(tmp_path)
    for _ in range(3):
        mb.next_batch()
    sd = mb.state_dict()
    rng_state = torch.get_rng_state()
    ref = [mb.next_batch() for _ in range(3)]
    store2 = ST.RealStore(tmp_path / "shards", store.names)
    C.set_group_options(cfg3())
    mb2 = SM.MixedBatcher(cfg3(), DEV, store2, {"a": 0.5, "b": 0.5}, SM.MixConfig(rows=24, real_fraction=1 / 3), seed=5)
    mb2.load_state_dict(sd)
    torch.set_rng_state(rng_state)
    for want in ref:
        got = mb2.next_batch()
        for k in ("x", "mask", "loss_ok", "blank", "seg_patch", "gid", "role", "slot_valid"):
            assert torch.equal(want[k], got[k]), k
        assert want["points"] == got["points"]
    assert mb2.real.realised() == mb.real.realised()


def test_synthetic_pools_are_keyed(tmp_path):
    cfg = cfg3()
    a = SM.SyntheticBatcher(dict(cfg, pack_rows=16), DEV, seed=9, stream=1)
    b = SM.SyntheticBatcher(dict(cfg, pack_rows=16), DEV, seed=9, stream=1)
    c = SM.SyntheticBatcher(dict(cfg, pack_rows=16), DEV, seed=9, stream=2)
    C.set_group_options(cfg)
    torch.manual_seed(0)
    xa = a.next_batch()["x"]
    torch.manual_seed(0)
    xb = b.next_batch()["x"]
    torch.manual_seed(0)
    xc = c.next_batch()["x"]
    assert torch.equal(xa, xb) and not torch.equal(xa, xc)                           # a different rank sees different data
