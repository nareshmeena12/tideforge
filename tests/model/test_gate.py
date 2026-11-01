"""attn_gate: none changes nothing; sigmoid keeps causality and M1's firewall; g = 1 is the ungated model."""
import sys
from pathlib import Path

import pytest
import torch

from tideforge import model as TM
from tideforge.expcfg import load_exp, model_fragment
from tideforge.model import gated as GT
from tideforge.model import m1 as M1
from tideforge.train import recipe as RC

sys.path.insert(0, str(Path(__file__).parent))
import test_m1 as T  # noqa: E402


def test_none_is_the_same_model_and_run_id():
    cfg = RC.load("E0.7").ft3_cfg()
    assert cfg["attn_gate"] == "none" and not [n for n, _ in TM.build(cfg).named_parameters() if "w_g" in n]
    torch.manual_seed(0)
    a = TM.build(cfg).state_dict()
    torch.manual_seed(0)
    b = TM.build({k: v for k, v in cfg.items() if k != "attn_gate"}).state_dict()
    assert list(a) == list(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert model_fragment(load_exp("E0.7").model) == "" and TM.n_params(TM.build(cfg)) == 5_447_968


def test_param_match_counts():
    assert TM.n_params(TM.build(RC.load("R1.5").ft3_cfg())) == 5_351_136 == RC.load("R1.6").sections["model"]["params"]
    assert RC.load("R1.5").exp.run_id().endswith("-agsigmoid")


@pytest.fixture
def gated(monkeypatch):
    plain = M1.build_m1
    monkeypatch.setattr(M1, "build_m1", lambda *a, **k: GT.install(plain(*a, **k)))
    made = TM.build(dict(T.CFG, known_attn="mask", attn_gate="sigmoid"))
    assert isinstance(made.blocks[0].time, GT.GatedTime) and isinstance(made.blocks[0].group, GT.GatedGroup)


@pytest.mark.parametrize("fn,arg", [(T.test_a_target_future_never_changes_earlier_outputs, ((), False)),
                                    (T.test_a_target_future_never_changes_earlier_outputs, (("target_time_full",), True)),
                                    (T.test_b_known_future_reaches_earlier_targets, ("mask", True)),
                                    (T.test_c_firewall_covariates_never_read_targets, ((), False)),
                                    (T.test_c_firewall_covariates_never_read_targets, (("symmetric_group",), True)),
                                    (T.test_d_past_future_never_changes_anything_earlier, ((), False)),
                                    (T.test_e_packed_series_stay_isolated, ((), False)),
                                    (T.test_g_second_target_never_opens_a_path_to_the_future, ((), False))])
def test_m1_guarantees_hold_with_the_gate(gated, fn, arg):
    fn(*arg)


@pytest.mark.parametrize("known_attn", ["mask", "off"])
def test_forecast_groups_equals_training_forward_with_the_gate(gated, known_attn):
    T.test_f_forecast_groups_equals_the_training_forward(known_attn)


def test_lookahead_bound_with_the_gate(gated):
    T.test_lookahead_bound_is_exact(2)


@pytest.mark.parametrize("known_attn", ["off", "mask"])
def test_gate_of_one_equals_ungated(known_attn):
    cfg = dict(T.CFG, known_attn=known_attn, attn_gate="none", dtype=None)
    torch.manual_seed(0)
    plain = TM.build(cfg).eval()
    torch.manual_seed(0)
    gate = TM.build(dict(cfg, attn_gate="sigmoid")).eval()
    for blk in gate.blocks:
        for a in (blk.time, blk.group):
            a.w_g.register_forward_hook(lambda m, i, o: torch.full_like(o, 50.0))
    b = T.batch()
    assert torch.equal(T.trace(plain, b)[-1], T.trace(gate, b)[-1])
    h = [T.trace(plain, b)[i] for i in (0, 1, 2)]
    assert all(torch.equal(x, y) for x, y in zip(h, T.trace(gate, b)))
