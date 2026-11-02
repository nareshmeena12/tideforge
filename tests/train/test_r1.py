"""R1: dropout as a config field, the m30 recipe, gradient accumulation, the job file."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

from tideforge import model as TM
from tideforge.expcfg import load_exp
from tideforge.train import batcher as TB
from tideforge.train import r1job as J
from tideforge.train import recipe as RC
from tideforge.train import trainer as TR

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tests" / "train"))
from test_trainer import run, tiny  # noqa: E402

spec = importlib.util.spec_from_file_location("m30_check", REPO / "tools" / "m30_check.py")
MC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MC)


def test_existing_run_ids_unchanged_by_the_new_fields():
    rc = RC.load("E0.7")
    assert rc.sections["model"]["dropout"] == 0.1 and "micro_rows" not in rc.sections["run"]
    h = rc.train_hash()
    rc.sections["run"]["micro_rows"] = 512
    assert rc.train_hash() == h
    assert RC.load("R1.1").train_hash() != h and RC.load("R1.1").ft3_cfg()["dropout"] == 0.0


def forward_pair(r):
    b = r.batcher.next_batch()
    r.model.train()
    a = TB.packed_forward(r.model, b)["_q"].detach()
    r.model.eval()
    return a, TB.packed_forward(r.model, b)["_q"].detach()


def test_dropout_0_train_equals_eval(tmp_path):
    r = run(tmp_path, tiny(dropout=0.0))
    assert r.cfg["dropout"] == 0.0
    assert torch.equal(*forward_pair(r))
    assert all(x.p == 0 for x in r.model.modules() if isinstance(x, torch.nn.Dropout))
    assert all(x.dropout_p == 0 for x in r.model.modules() if hasattr(x, "dropout_p"))


def test_default_dropout_is_on(tmp_path):
    assert not torch.equal(*forward_pair(run(tmp_path, tiny())))


def test_accumulation_gives_the_full_batch_gradients(tmp_path):
    r = run(tmp_path, tiny(dropout=0.0))
    b = r.batcher.next_batch()
    r.model.zero_grad()
    full = TB.packed_forward(r.model, b)["loss"]
    full.backward()
    g1 = [p.grad.clone() for p in r.model.parameters()]
    r.model.zero_grad()
    acc = TR.accum_backward(r.model, b, 8, r.cfg["max_segs_per_row"], lambda: torch.autocast("cpu", enabled=False))
    assert float(acc.detach()) == pytest.approx(float(full.detach()), rel=1e-6)
    for a, p in zip(g1, r.model.parameters()):
        assert torch.allclose(a, p.grad, rtol=1e-4, atol=1e-7)


def test_split_rows_refuses_a_cut_through_a_block(tmp_path):
    r = run(tmp_path, tiny())
    with pytest.raises(ValueError):
        next(TB.split_rows(r.batcher.next_batch(), 4, r.cfg["max_segs_per_row"]))


def test_trainer_step_with_micro_rows_matches_full(tmp_path):
    a = run(tmp_path / "a", tiny(dropout=0.0))
    b = run(tmp_path / "b", tiny(dropout=0.0, micro_rows=8))
    a.n_bad = b.n_bad = 0
    assert a.run_id == b.run_id
    torch.manual_seed(5)                              # next_batch draws crops from the global generator
    la = a.one_step()
    torch.manual_seed(5)
    assert la == pytest.approx(b.one_step(), rel=1e-5)
    for x, y in zip(a.model.parameters(), b.model.parameters()):
        assert torch.allclose(x, y, rtol=1e-5, atol=1e-7)


def test_m30_is_ft3_mains_architecture():
    rc = RC.load("R1.3")
    cfg = rc.ft3_cfg()
    ref = json.loads((REPO / "configs" / "ft3_main_cfg.json").read_text(encoding="utf-8"))
    assert MC.arch_diff(cfg, ref) == []
    assert TM.n_params(TM.build(cfg)) == rc.sections["model"]["params"]
    assert 28e6 < rc.sections["model"]["params"] < 31e6
    a, b = RC.load("E0.7").sections, rc.sections
    assert {k: v for k, v in a["optim"].items() if k != "peak_lr"} == {k: v for k, v in b["optim"].items() if k != "peak_lr"}
    assert a["batch"] == b["batch"] and a["run"] == b["run"] and b["optim"]["peak_lr"] == 3e-4


def test_arch_diff_fails_on_a_difference_but_not_dropout():
    ref = dict(d_model=512, dropout=0.1)
    assert MC.arch_diff(dict(d_model=512, dropout=0.0), ref) == []
    assert MC.arch_diff(dict(d_model=256, dropout=0.0), ref)


def test_r1_configs():
    c2 = load_exp("C2").gen.config_hash()
    assert load_exp("R1.2").gen.config_hash() == c2 == load_exp("R1.4").gen.config_hash()
    assert load_exp("R1.1").gen.config_hash() == load_exp("E0.7").gen.config_hash() == load_exp("R1.3").gen.config_hash()
    assert [RC.load(e).sections["model"]["d_model"] for e in J.RUNS] == [256, 256, 512, 512, 256, 256]


def test_job_file():
    t = J.render("30:00:00", 512, "m0000_g")
    assert 'chain 0 "R1.3:512"' in t and 'chain 1 "R1.4:512"' in t and 'chain 2 "R1.1:0 R1.5:0"' in t and 'chain 3 "R1.2:0 R1.6:0"' in t
    assert "srun" not in t and all(f"CUDA_VISIBLE_DEVICES={i} OMP_NUM_THREADS=32 python -m tideforge.eval.score" in t for i in range(4))
    assert "#SBATCH -q regular" in t and "#SBATCH -t 30:00:00" in t and t.count("--modes on --skip-time") == 6
    assert "LOOKS.md" not in t and "\r" not in t and "R1.3:0" in J.render("30:00:00", 0, "m0000_g")
    with pytest.raises(ValueError):
        J.render("soon", 0, "x")
