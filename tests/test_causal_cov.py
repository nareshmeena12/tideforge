"""causal_cov: zero effect means no dependence, targets never reach covariates, at least 60% targets, determinism, effect labels,
throughput; P11's oracle and the adapter job file."""
import time

import numpy as np
import pytest
import torch

from tideforge.gen.ext import causal_cov as CC
from tideforge.gen.groups import ROLE_TARGET
from tideforge.gen.switches import REGISTRY

P = dict(REGISTRY["causal_cov"].params)
N, V, L = 200, 8, 256


def run(x, seed=3, **kw):
    g = torch.Generator().manual_seed(seed)
    role = torch.zeros(x.shape[:2], dtype=torch.long)
    lgid = torch.arange(x.shape[1]).expand(x.shape[:2]).clone()
    return CC.causal_blocks(x, role, lgid, g, dict(P, **kw))


def base():
    return torch.randn(N, V, L, generator=torch.Generator().manual_seed(1)).cumsum(-1) * 0.1 + 5


def test_roles_share():
    _, role, lgid, _ = run(base())
    t = (role == ROLE_TARGET).float().mean(1)
    assert (t >= 0.6).all() and ((role != ROLE_TARGET).sum(1) >= 1).all()
    assert (lgid[role != ROLE_TARGET] == 0).all()


def test_shares_of_strength():
    _, role, _, eff = run(base(), seed=4)
    e = eff[role != ROLE_TARGET]
    assert abs(float((e == 0).float().mean()) - 0.3) < 0.1 and abs(float((e >= 0.5).float().mean()) - 0.4) < 0.1


def test_zero_effect_means_no_dependence():
    """All covariates zero: no target depends on them. A covariate is changed, every target stays identical, and untouched lanes
    keep their exact input values (no clip)."""
    x = base()
    y, role, _, eff = run(x, zero=1.0, weak=0.0, strong=0.0, target_edge_p=0.0)
    assert torch.equal(y, x) and (eff == 0).all()
    x2 = x.clone()
    x2[role != ROLE_TARGET] += 7.0
    y2, *_ = run(x2, zero=1.0, weak=0.0, strong=0.0, target_edge_p=0.0)
    assert torch.equal(y2[role == ROLE_TARGET], y[role == ROLE_TARGET])


def test_zero_covariate_has_no_path_through_targets():
    x = base()
    y, role, _, eff = run(x, zero=0.5, weak=0.25, strong=0.25, target_edge_p=0.5)
    x2 = x.clone()
    y0, r0, _, e0 = run(x2)
    assert (eff[role == ROLE_TARGET] >= 0).all()
    untouched = (role == ROLE_TARGET) & (eff == 0)
    y2, *_ = run(x, zero=0.5, weak=0.25, strong=0.25, target_edge_p=0.0)
    # with target links on, a target with effect 0 may only have read other targets with effect 0
    x3 = x.clone()
    cov0 = (role != ROLE_TARGET) & (eff == 0)
    x3[cov0] += 7.0
    y3, *_ = run(x3, zero=0.5, weak=0.25, strong=0.25, target_edge_p=0.5)
    assert torch.equal(y3[untouched], y[untouched])
    assert untouched.sum() > 0 and cov0.sum() > 0


def test_never_the_reverse():
    x = base()
    y, role, _, _ = run(x)
    x2 = x.clone()
    x2[role == ROLE_TARGET] += torch.randn(int((role == ROLE_TARGET).sum()), L)
    y2, *_ = run(x2)
    assert torch.equal(y2[role != ROLE_TARGET], y[role != ROLE_TARGET])


def test_strong_effect_acts_and_clip_holds():
    x = base()
    y, role, _, eff = run(x, zero=0.0, weak=0.0, strong=1.0)
    t = (role == ROLE_TARGET) & (eff > 0)
    assert t.sum() > 0 and not torch.equal(y[t], x[t]) and bool(torch.isfinite(y).all())
    z = (y[t] - x[t].mean(-1, keepdim=True)) / x[t].std(-1, keepdim=True)
    assert float(z.abs().max()) <= 5 + 1e-3


def test_deterministic():
    a, b = run(base()), run(base())
    assert all(torch.equal(p, q) for p, q in zip(a, b))
    assert not torch.equal(run(base(), seed=9)[0], a[0])


@pytest.mark.parametrize("op", range(len(CC.OPS)))
def test_each_operator_is_causal(op):
    """Changing x after t leaves y up to t unchanged, except coint and trend-free ops that are pointwise, which also hold."""
    g = lambda: torch.Generator().manual_seed(5)   # noqa: E731
    y, x = torch.randn(4, 128, generator=g()), torch.randn(4, 128, generator=torch.Generator().manual_seed(6))
    a = torch.ones(4, 1)
    r, _ = CC.apply_op(op, y, x, a, g(), P)
    x2 = x.clone()
    x2[:, 100:] += 3
    r2, _ = CC.apply_op(op, y, x2, a, g(), P)
    assert torch.equal(r[:, :100], r2[:, :100])


def test_switch_runs_through_the_generator_with_effect():
    import tideforge.gen as G
    from tideforge.expcfg import apply_changes, base_config
    gc = apply_changes(base_config("b0"), None, {"causal_cov": True})
    prof = G.configure(gc, seed=0)
    CC.pop_effect()
    torch.manual_seed(0)
    t0 = time.time()
    x, role, lgid = G.generate_blocks(16, 8, 512, torch.device("cpu"), None, profile=prof, group_prob=gc.group_prob)
    rate = 16 * 8 * 512 / (time.time() - t0)
    eff = CC.pop_effect()
    print(f"causal_cov CPU {rate / 1e6:.2f}M pts/s")
    assert eff.shape == (16, 8) and ((role != ROLE_TARGET).sum(1) >= 1).all() and rate > 0


def test_p11_oracle():
    from tideforge.probes import metrics as M, p11_overtrust as P11
    ps = P11.build(64)
    assert [c.name for c in ps.cells] == list(P11.CELLS)
    ratio = {}
    for c in ps.cells:
        y, sc = c.arrays["future"], M.scale_seasonal(c.arrays["targets"], c.S)
        on, hid = M.sql(c.arrays["oracle_q"], y, sc), M.sql(c.arrays["oracle_hidden_q"], y, sc)
        ratio[c.name] = float(on.sum() / hid.sum())
    assert ratio["control"] < 0.9 and max(v for k, v in ratio.items() if k != "control") < 1.02
    assert P11.harm_summary(dict(a=1.2, b=0.9, control=0.5)) == pytest.approx(0.1)


def test_job_files():
    from tideforge.train import adjob as J
    for s, lr in (("pilot", None), ("ad12", "3e-4"), ("ad34", "1e-3")):
        t = J.render(s, "06:00:00", lr, "m0000_g")
        assert "srun" not in t and "LOOKS.md" not in t and "CUDA_VISIBLE_DEVICES=1" in t
    t = J.render("ad12", "06:00:00", "3e-4", "x")
    assert t.count("check-identical") == 4 and "R1.2-*" in t and "--seed 1" in t and t.count("looks_pending.md") == 4
    with pytest.raises(ValueError):
        J.render("ad12", "06:00:00", None, "x")


def test_pilot_score(tmp_path):
    import pandas as pd
    from tideforge.train import adjob as J
    rows = [dict(probe="P1", cell="lagged@1", mode="on", metric="sql_on_vs_hidden", value=0.5),
            dict(probe="P1", cell="decoy@1", mode="on", metric="sql_on_vs_hidden", value=9.0),
            dict(probe="P11", cell="weak", mode="on", metric="harm", value=1.2),
            dict(probe="P11", cell="control", mode="on", metric="harm", value=0.5)]
    pd.DataFrame(rows).to_csv(tmp_path / "r.csv", index=False)
    assert J.pilot_score(tmp_path / "r.csv") == pytest.approx(0.7)
