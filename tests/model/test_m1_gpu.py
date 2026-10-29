"""M1 on a GPU (Perlmutter: pytest -m gpu tests/model/test_m1_gpu.py): the firewall and the past-covariate leak test hold
exactly under CUDA SDPA kernels and bf16 autocast (a masked key contributes exactly 0 there too), and forecast_groups (which
runs under autocast on CUDA) equals the training forward pass under the same autocast."""
import numpy as np
import pytest
import torch

import tidemark.real.ft3_core as C
from model import test_m1 as T          # pytest puts tests/ on sys.path (tests/model is a package)

pytestmark = pytest.mark.gpu


def _on(dev, m, b):
    return m.to(dev), {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


@pytest.mark.parametrize("bf16", [False, True])
def test_firewall_and_past_leak_on_cuda(bf16):
    m, b = _on("cuda", T.make(), T.batch())
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        assert T.changed(m, b, T.at([T.A, T.B], 0, T.P), [T.PAST, T.KNOWN], list(range(T.P))) == 0.0
        assert T.changed(m, b, T.at([T.PAST], T.T + 1), T.ALL_ROWS, T.EARLY) == 0.0
        assert T.changed(m, b, T.at([T.KNOWN], T.T + 2), [T.A, T.B], T.EARLY) > 0          # the mechanism works


def test_forecast_groups_on_cuda_matches_cpu_order_of_magnitude():
    m = T.make().cuda()
    gr, H = T._group()
    q = np.asarray(C.forecast_groups(m, [gr], H, torch.device("cuda"))[0][1])
    q_cpu = np.asarray(C.forecast_groups(T.make(), [gr], H, torch.device("cpu"))[0][1])
    assert np.isfinite(q).all() and np.abs(q - q_cpu).max() < 0.05 * (np.abs(q_cpu).max() + 1)   # bf16 vs fp32
