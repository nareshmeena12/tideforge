"""The adapter trainer on CPU with a tiny model: covariate-only batches, the backbone stays frozen, the checkpoint loads as a model."""
import torch

from tideforge import model as TM
from tideforge.model import adapter as AD
from tideforge.train import adapter_train as AT
from tideforge.train import batcher as TB
from tideforge.train import finish as FN  # noqa: F401
from tideforge.train import trainer as TR

from .test_trainer import tiny


def backbone(tmp_path):
    r = tiny("R1.2")
    run = TR.Run(r, 0, tmp_path / "bb", "cpu", "code", log=lambda *a, **k: None).build(check_params=False)
    p = tmp_path / "bb.pt"
    torch.save(TR.slim(run.model, run.cfg, 0, 0.0, "bb", "test"), p)
    return p, run.model.state_dict()


def test_has_covariate():
    role = torch.tensor([[0, 0, 1], [0, 0, 0], [2, 0, 0], [0, 0, 1]])
    gid = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 1, 2], [0, 1, 2]])
    assert TB.has_covariate(role, gid).tolist() == [True, False, False, False]


def test_adapter_run(tmp_path):
    p, before = backbone(tmp_path)
    run = AT.AdapterRun(tiny("AD.2"), 0, tmp_path, "cpu", "code", p, log=lambda *a, **k: None).build()
    assert run.n_adapter == sum(q.numel() for q in AD.adapter_parameters(run.model))
    b = run.batcher.next_batch()
    assert "effect" in b and b["effect"].shape == b["slot_valid"].shape and (b["role"] != 0).any()
    run.opt_cfg["target_points"] = 1e12
    assert run.train(max_steps=3) == TR.EXIT_DONE
    sd = run.model.state_dict()
    assert all(torch.equal(sd[k], v) for k, v in before.items())
    assert any(v.abs().sum() > 0 for k, v in sd.items() if k.endswith("adapter.out_proj.weight"))
    m, cfg, _ = TM.load(run.dir / "final.pt", torch.device("cpu"))
    assert cfg["adapter_every"] == 1 and sum(1 for k in m.state_dict() if ".adapter." in k) > 0


def test_cov_only_rows_all_have_covariates(tmp_path):
    p, _ = backbone(tmp_path)
    run = AT.AdapterRun(tiny("AD.1"), 0, tmp_path, "cpu", "code", p, log=lambda *a, **k: None).build()
    assert run.batcher.cov_only
    x, role, lgid = run.batcher._generate(32, 256)
    assert x.shape[0] == 32 and TB.has_covariate(role, lgid).all()
