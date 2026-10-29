"""T3 trainer on CPU with a tiny model: the model port (same weights, outputs and loss as ft3 / tidemark), the 5M parameter
count, the model-option hook, the LR schedule, loss goes down, resume reproduces the next step exactly, a stop signal saves
and exits 3, the run id changes with every setting, finish without a dev set (synthetic, a few steps; no real training)."""
import json
from pathlib import Path

import pytest
import torch

import tidemark.real.ft3_core as C
import tidemark.real.model as TMR
from tideforge import model as TM
from tideforge.expcfg import load_exp
from tideforge.train import finish as FN
from tideforge.train import recipe as RC
from tideforge.train import trainer as TR

REPO = Path(__file__).resolve().parents[2]


def tiny(exp="E0.7", **train):
    r = RC.load_recipe()
    r["model"].update(d_model=32, n_layers=1, n_heads=2, d_kv=16, d_ff=64, params=None)
    r["batch"].update(pack_rows=16, max_ctx=256, len_bucket_edges=[16, 32, 64, 128, 256], long_min_ctx=128,
                      gen_pool_points=5e4, gen_pool_max_rows=512, max_segs_per_row=8)
    r["run"].update(compile=False, val_batches=1, val_every=10 ** 9, print_every=5, ckpt_every_s=1e9)
    r["optim"].update(peak_lr=3e-3, warmup_frac=0.01, target_points=1e12, milestone_points=0)
    e = load_exp(exp)
    if train:
        e = type(e)(**{**e.__dict__, "train": dict(e.train, **train)})
    return RC.merge(e, r)


def run(tmp_path, rc=None, seed=0, code="testcode"):
    return TR.Run(rc or tiny(), seed, tmp_path, "cpu", code, log=lambda *a, **k: None).build(check_params=False)


def test_5m_parameter_count_and_model_port():
    rc = RC.load("E0.7")
    cfg = rc.ft3_cfg()
    torch.manual_seed(0)
    m = TM.build(cfg)
    assert TM.n_params(m) == rc.sections["model"]["params"] == 5_447_968
    torch.manual_seed(0)
    ref = TMR.build_masked_model(cfg)
    torch.manual_seed(0)
    ref_core = C.build_model(cfg)
    for (k, a), (k2, b), (k3, c) in zip(m.state_dict().items(), ref.state_dict().items(), ref_core.state_dict().items()):
        assert k == k2 == k3 and torch.equal(a, b) and torch.equal(a, c)
    with pytest.raises(NotImplementedError):
        TM.build(cfg, {"known_attn": "whole_window"})
    with pytest.raises(KeyError):
        TM.build(cfg, {"lanes": 16})


def test_outputs_and_loss_equal_the_reference(tmp_path):
    r = run(tmp_path)
    torch.manual_seed(1)
    b = r.batcher.next_batch()
    ref = TMR.build_masked_model(r.cfg)
    ref.load_state_dict(r.model.state_dict())
    core = C.build_model(r.cfg)
    core.load_state_dict(r.model.state_dict())
    for m in (r.model, ref, core):
        m.eval()
    from tideforge.train.batcher import packed_forward
    a = packed_forward(r.model, b)
    c = C.packed_forward(core, b)
    assert torch.equal(a["loss"], c["loss"]) and torch.equal(a["_q"], c["_q"])
    assert torch.equal(a["loss"], TMR.packed_forward_masked(ref, b)["loss"])


def test_lr_schedule():
    o = dict(peak_lr=5e-4, warmup_frac=0.02, warmup_start_frac=0.04, final_lr_frac=0.0, target_points=100e9)
    assert TR.lr_at(0, o) == pytest.approx(5e-4 / 25)
    assert TR.lr_at(2e9, o) == pytest.approx(5e-4)
    assert TR.lr_at(51e9, o) == pytest.approx(2.5e-4, rel=1e-3)
    assert TR.lr_at(100e9, o) == pytest.approx(0.0, abs=1e-12)


def test_loss_goes_down(tmp_path):
    r = run(tmp_path)
    r.n_bad = 0
    losses = [r.one_step() for _ in range(40)]
    first, last = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    print(f"\ntiny model: loss {first:.4f} -> {last:.4f}")
    assert last < first * 0.9 and r.n_bad == 0


def test_resume_reproduces_the_next_step(tmp_path):
    a = run(tmp_path)
    a.dir.mkdir(parents=True)
    a.n_bad = 0
    for _ in range(4):
        a.one_step()
    a.save_last()
    want = a.one_step()
    pa = [p.detach().clone() for p in a.model.parameters()]
    b = run(tmp_path)
    b.load(torch.load(a.dir / "last.pt", weights_only=False))
    b.n_bad = 0
    got = b.one_step()
    assert got == want and b.step == a.step and b.points == a.points
    assert all(torch.equal(x, y) for x, y in zip(pa, b.model.parameters()))


def test_stop_signal_saves_and_exits_3(tmp_path):
    r = run(tmp_path)
    TR._STOP.update(flag=True, why="SIGUSR1")
    try:
        assert r.train(max_steps=50) == TR.EXIT_REQUEUE
    finally:
        TR._STOP.update(flag=False, why="")
    st = torch.load(r.dir / "last.pt", weights_only=False)
    assert st["step"] == 1 and json.loads((r.dir / "run.json").read_text())["run_id"] == r.run_id
    r2 = run(tmp_path)
    assert r2.train(max_steps=3) == TR.EXIT_DONE                    # resumes at step 1 and finishes at the step cap
    assert (r2.dir / "final.pt").exists() and r2.step == 3


def test_run_id_changes_with_every_setting():
    base = tiny()
    ids = {base.run_id(0, "c1")}
    for rid in (base.run_id(1, "c1"), base.run_id(0, "c2"), tiny("E0.8").run_id(0, "c1"),
                tiny(long_row_frac=0.5).run_id(0, "c1"), tiny(pack_rows=32).run_id(0, "c1"),
                tiny(lead_noise=dict(p_noisy=0.5)).run_id(0, "c1")):
        assert rid not in ids
        ids.add(rid)
    with pytest.raises(KeyError):
        tiny(not_a_setting=1)


def test_finish_without_dev_set(tmp_path):
    r = run(tmp_path)
    assert r.train(max_steps=2) == TR.EXIT_DONE
    res = tmp_path / "results.csv"
    row = FN.finish(r.dir / "final.pt", dev=str(tmp_path / "nodev"), probe_root=str(tmp_path / "noprobes"),
                    results=str(res), device="cpu", log=lambda *a: None)
    assert row["dev"] == "dev set missing" and row["probes"] == "probes missing" and row["holdout_note"] == FN.HOLDOUT_NOTE
    assert row["run_id"] == r.run_id and res.read_text().count("\n") == 2
