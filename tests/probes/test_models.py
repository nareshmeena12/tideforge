"""The Ft3 adapter end to end on CPU with a tiny random ft3 model saved as a checkpoint: groups on (covariates) and hidden,
multi-target items (P8), gaps under every NaN variant (P6); shapes and sorted quantiles."""
import numpy as np
import pytest
import torch

import tidemark.real.ft3_core as C
import tideforge.probes.p01_covariates as P1
import tideforge.probes.p06_gaps as P6
import tideforge.probes.p08_hierarchy as P8
from tideforge.probes.models import Ft3, M5, make


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory):
    cfg = dict(C.CONFIG, d_model=64, d_kv=16, n_heads=4, d_ff=128, n_layers=2, max_ctx=512, group_lanes=8, dropout=0.0)
    torch.manual_seed(0)
    m = C.build_model(cfg)
    p = tmp_path_factory.mktemp("ck") / "tiny.pt"
    torch.save({"cfg": cfg, "model": m.state_dict()}, p)
    return p


def _check(q, cell):
    assert q.shape == (cell.n,) + cell.arrays["oracle_q"].shape[1:]
    assert np.isfinite(q).all() and (np.diff(q, axis=2) >= 0).all()


def test_ft3_modes_and_targets(ckpt):
    m = Ft3(ckpt, "cpu")
    cell = P1.build(3).cell("lagged_linear@1024")
    for mode in ("on", "hidden"):
        _check(m.predict_cell(cell, mode), cell)
    on, hid = m.predict_cell(cell, "on"), m.predict_cell(cell, "hidden")
    assert not np.array_equal(on, hid)                     # the covariate lane reaches the model in groups ON
    h = P8.build(2).cell("similar")
    _check(m.predict_cell(h, "on"), h)


@pytest.mark.parametrize("variant", ["baseline", "nofill_delete", "nofill_mask"])
def test_ft3_gaps_under_nan_variants(ckpt, variant):
    cell = P6.build(3).cell("random")
    _check(Ft3(ckpt, "cpu", variant).predict_cell(cell, "main", "main"), cell)


def test_make_and_m5_stub(ckpt):
    assert make("oracle").name == "oracle" and make("naive").name == "naive"
    with pytest.raises(NotImplementedError):
        M5("x").predict_cell(None)
    with pytest.raises(ValueError):
        make("ft3main")
