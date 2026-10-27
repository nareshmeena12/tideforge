"""The real-data trainer end to end on CPU with a tiny model and tiny data: strict init, LR schedule, milestones, exact resume."""
import json
from dataclasses import replace

import pytest
import torch

from tidemark.pretrain.train import Dist
from tidemark.real import ft3_core as C
from tidemark.real import runcfg as RC
from tidemark.real import sampler as SM
from tidemark.real import train as T
from test_real_eval import tiny_cfg
from test_real_sampler import make_store


def lr_cfg(**kw):
    return RC.RunConfig(target_points=100.0, **kw)


def test_lr_schedule_shape():
    rc = lr_cfg()
    assert T.lr_at(0, rc) == pytest.approx(1.2e-4 * 0.04)
    assert T.lr_at(5.0, rc) == pytest.approx(1.2e-4)                       # end of the 5% warmup
    assert T.lr_at(2.5, rc) == pytest.approx(1.2e-4 * (0.04 + 0.96 * 0.5))
    assert T.lr_at(100.0, rc) == pytest.approx(1.2e-5)                     # cosine to 10% of the peak
    assert T.lr_at(1e9, rc) == pytest.approx(1.2e-5)
    mid = T.lr_at(52.5, rc)
    assert mid == pytest.approx(1.2e-4 * (0.1 + 0.9 * 0.5), rel=1e-6)
    pts = [T.lr_at(x, rc) for x in range(5, 101)]
    assert all(a >= b for a, b in zip(pts, pts[1:]))                       # monotone after the warmup


@pytest.fixture()
def world(tmp_path):
    cfg = tiny_cfg()
    torch.manual_seed(0)
    m = C.build_model(dict(cfg, dropout=0.0))
    ck = tmp_path / "ft3_tiny.pt"
    torch.save(dict(model=m.state_dict(), cfg=cfg, step=1, points_seen=1e9, packed=True), ck)
    real_root = tmp_path / "real"
    real_root.mkdir()
    make_store(real_root, {"a": (30, 600, 0.0), "b": (30, 300, 0.2)})            # writes real_root/shards
    (real_root / "plan.json").write_text(json.dumps({"targets": {"a": 0.5, "b": 0.5}}))
    rc = RC.RunConfig(name="t", ckpt=str(ck), real_root=str(real_root), plan=str(real_root / "plan.json"), out_dir=str(tmp_path / "out"),
                      copy_to=str(tmp_path / "perm"), target_points=30000.0, milestones=(10000.0, 20000.0, 30000.0), mix=SM.MixConfig(rows=24),
                      val_every=4, print_every=2, save_every_s=0.0, compile=False, val_batches=1)
    return rc


def checksum(path):
    return sum(float(v.double().sum()) for v in torch.load(path, map_location="cpu", weights_only=False)["model"].values())


def test_train_to_the_end_writes_milestones_and_copies(world, capsys):
    rc = world
    assert T.run(rc, Dist("cpu"), log=print) == T.EXIT_DONE
    out = capsys.readouterr().out
    from pathlib import Path
    names = sorted(p.name for p in Path(rc.out_dir).glob("*.pt"))
    ms = sorted(T.milestone_name(m) for m in rc.milestones)
    assert set(ms) | {"final.pt", "last.pt"} <= set(names) and T.milestone_name(10e9) == "ckpt_10B.pt"
    assert sorted(p.name for p in Path(rc.copy_to).glob("*.pt")) == ms
    assert "fresh start" in out and "synthetic holdout loss" in out and "real shares" in out and "DONE" in out
    fin = torch.load(Path(rc.out_dir) / "final.pt", map_location="cpu", weights_only=False)
    assert fin["points_seen"] >= 30000 and fin["packed"] and fin["cfg"]["d_model"] == 64
    # the milestone file loads like ft3_main.pt does (strict, standard decoder)
    from tidemark.real.score_ft3 import load_ft3
    model, cfg, _ = load_ft3(Path(rc.out_dir) / T.milestone_name(20000.0), torch.device("cpu"))
    assert cfg["d_model"] == 64
    lasts = [float(l.split("lr ")[1].split()[0]) for l in out.splitlines() if l.startswith("step ")]
    assert lasts[-1] < lasts[0] and lasts == sorted(lasts, reverse=True)       # the tiny budget is past its warmup after one step: decay


def test_resume_is_exact(world, tmp_path):
    from pathlib import Path
    rc = world
    straight = replace(rc, out_dir=str(tmp_path / "straight"), copy_to="")
    assert T.run(straight, Dist("cpu"), log=lambda *a, **k: None) == T.EXIT_DONE
    stopped = replace(rc, out_dir=str(tmp_path / "stopped"), copy_to="")
    T.run(stopped, Dist("cpu"), max_steps=3, log=lambda *a, **k: None)         # last.pt is written every step (save_every_s = 0)
    assert (Path(stopped.out_dir) / "last.pt").exists() and not (Path(stopped.out_dir) / "final.pt").exists()
    msgs = []
    assert T.run(stopped, Dist("cpu"), log=lambda *a, **k: msgs.append(" ".join(map(str, a)))) == T.EXIT_DONE
    assert any("RESUMED" in m for m in msgs)
    a, b = torch.load(Path(straight.out_dir) / "final.pt", map_location="cpu", weights_only=False), \
        torch.load(Path(stopped.out_dir) / "final.pt", map_location="cpu", weights_only=False)
    assert a["step"] == b["step"] and a["points_seen"] == b["points_seen"]
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), k                    # bit-identical weights


def test_stop_file_saves_and_exits_for_requeue(world):
    from pathlib import Path
    rc = replace(world, copy_to="")
    Path(rc.out_dir).mkdir(parents=True)
    (Path(rc.out_dir) / "STOP").write_text("")
    assert T.run(rc, Dist("cpu"), log=lambda *a, **k: None) == T.EXIT_REQUEUE
    assert (Path(rc.out_dir) / "last.pt").exists()
    (Path(rc.out_dir) / "STOP").unlink()
    assert T.run(rc, Dist("cpu"), log=lambda *a, **k: None) == T.EXIT_DONE


def test_resume_refuses_a_changed_configuration(world):
    rc = replace(world, copy_to="")
    T.run(rc, Dist("cpu"), max_steps=2, log=lambda *a, **k: None)
    with pytest.raises(SystemExit, match="configuration changed"):
        T.run(replace(rc, peak_lr=2e-4), Dist("cpu"), log=lambda *a, **k: None)


def test_loader_benchmark_runs_and_reports_shares(world, capsys):
    from tidemark.real import bench_loader as BL
    assert BL.main(["--real-root", world.real_root, "--ckpt", world.ckpt, "--rows", "24", "--steps", "2", "--warm", "1", "--device", "cpu"]) == 0
    out = capsys.readouterr().out
    assert "points/s" in out and "mixed" in out and "synthetic" in out and "realised" in out and "a " in out


import os
import shutil
import subprocess
import sys
from pathlib import Path

bash = shutil.which("bash")
ROOT = Path(__file__).parents[1]


@pytest.mark.skipif(bash is None, reason="needs bash")
def test_interactive_script_dry_run(tmp_path):
    scratch = tmp_path / "scratch"
    root = scratch / "tidemark"
    (root / "repo" / "scripts" / "perlmutter").mkdir(parents=True)
    (root / "repo" / "scripts" / "perlmutter" / "env.sh").write_text(f'export ROOT="{root.as_posix()}"\n')
    script = ROOT / "scripts" / "perlmutter" / "real_train_interactive.sh"
    env = {**os.environ, "SCRATCH": scratch.as_posix(), "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]}
    r = subprocess.run([bash, str(script), "--smoke", "--dry-run"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "torchrun --standalone --nproc_per_node=4 -m tidemark.real.train --run real30b --smoke" in r.stdout and "smoke1b" in r.stdout
    assert "MISSING:" in r.stdout and "nothing started" in r.stdout
    r = subprocess.run([bash, str(script), "--bench", "--dry-run"], capture_output=True, text=True, env=env, timeout=120)
    assert "tidemark.real.bench_loader" in r.stdout
    assert subprocess.run([bash, str(script), "--bogus"], capture_output=True, text=True, env=env).returncode == 2


def test_generated_sbatch_scripts():
    from tidemark.pretrain import slurm
    for smoke, name in ((True, "real_smoke"), (False, "real_30b")):
        text = slurm.real_script(smoke)
        assert f"#SBATCH -J {name}" in text and "#SBATCH --gpus-per-node=4" in text and "#SBATCH --requeue" in text
        assert "torchrun --standalone --nproc_per_node=4 -m tidemark.real.train" in text and ("--smoke" in text) == smoke
        assert "scontrol requeue" in text and "trap on_term TERM USR1" in text and "REAL_COPY_TO" in text
        assert (ROOT / "scripts" / "perlmutter" / f"{name}.sbatch").read_text() == text            # the committed file is the generator's
