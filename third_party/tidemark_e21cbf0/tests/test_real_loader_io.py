"""Loader I/O (persistent descriptors, positioned reads, missing positions in RAM, worker threads, prefetch), the worker sweep, the real200b
presets and launchers. The batches themselves are pinned by test_real_loader_golden.py."""
import os
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from test_real_eval import tiny_cfg
from test_real_sampler import make_store
from test_real_train import world  # noqa: F401  (fixture)

from tidemark.real import bench_loader as BL
from tidemark.real import ft3_core as C
from tidemark.real import plan as P
from tidemark.real import runcfg as RC
from tidemark.real import sampler as SM
from tidemark.real import store as ST
from tidemark.real import train as T

ROOT = Path(__file__).parents[1]
bash = shutil.which("bash")


@pytest.fixture()
def emulated_preadv(monkeypatch):
    """os.preadv does not exist on Windows: emulate it with lseek + read so the descriptor path of the store runs here too."""
    def preadv(fd, buffers, offset):
        os.lseek(fd, offset, 0)
        data = os.read(fd, len(buffers[0]))
        buffers[0][:len(data)] = data
        return len(data)
    monkeypatch.setattr(os, "preadv", preadv, raising=False)
    monkeypatch.setattr(ST, "_HAS_PREAD", True)


def test_pread_and_mmap_read_the_same_values_and_positions(tmp_path, emulated_preadv):
    store = make_store(tmp_path, {"a": (25, 700, 0.15), "b": (25, 300, 0.4)})
    rng = np.random.default_rng(3)
    for di in range(2):
        for _ in range(300):
            ser = int(rng.integers(0, len(store.lens[di])))
            c = int(rng.integers(33, int(store.lens[di][ser]) + 1))
            start = int(rng.integers(0, int(store.lens[di][ser]) - c + 1))
            got = {}
            for io in ("mmap", "pread"):
                store.configure_io(io)
                out = np.full(c + 5, -1.0, dtype=np.float32)
                store.read_window(di, ser, start, c, out)
                got[io] = (out.copy(), store.window_missing(di, ser, start, c))
                assert (out[c:] == -1.0).all()                                         # only the first c points are written
            np.testing.assert_array_equal(got["mmap"][0], got["pread"][0])
            np.testing.assert_array_equal(got["mmap"][1], got["pread"][1])
            base = int(store.offs[di][ser]) + start
            np.testing.assert_array_equal(got["pread"][1], store.missing_in(di, base, base + c))     # the series pointer finds the same positions
    assert len(store._fds) == 2                                                          # one persistent descriptor per dataset
    store.close()
    assert not store._fds


def test_thread_budget_keeps_four_threads_free():
    SM.MixConfig(fetch_workers=14, threads_per_worker=2).check()                         # 28
    SM.MixConfig(fetch_workers=24, threads_per_worker=1).check()
    for w, t in ((16, 2), (24, 2), (29, 1)):
        with pytest.raises(ValueError, match="at most 28"):
            SM.MixConfig(fetch_workers=w, threads_per_worker=t).check()
    assert SM.MixConfig(fetch_workers=0).fetch_threads == 0 and SM.MixConfig(fetch_workers=8, threads_per_worker=2).fetch_threads == 16


def test_a_failing_read_is_reported_not_swallowed(tmp_path):
    store = make_store(tmp_path, {"a": (30, 600, 0.0)})
    C.set_group_options(dict(tiny_cfg(), pack_rows=24))
    mix = SM.MixConfig(rows=24, real_fraction=1 / 3, fetch_workers=3, threads_per_worker=1)
    rb = SM.RealBatcher(dict(tiny_cfg(), pack_rows=24), torch.device("cpu"), store, {"a": 1.0}, mix, rows=8, seed=1)

    def boom(*a, **k):
        raise OSError("disk went away")
    rb.store.read_window = boom
    with pytest.raises(OSError, match="disk went away"):
        rb.next_batch()
    rb.close()


def test_timers_and_pending_plan_survive_a_checkpoint(tmp_path):
    store = make_store(tmp_path, {"a": (30, 600, 0.1), "b": (30, 300, 0.2)})
    cfg = dict(tiny_cfg(), pack_rows=24)
    C.set_group_options(cfg)
    mix = SM.MixConfig(rows=24, real_fraction=1 / 3, fetch_workers=4, threads_per_worker=2)
    rb = SM.RealBatcher(cfg, torch.device("cpu"), store, {"a": 0.5, "b": 0.5}, mix, rows=8, seed=2)
    rb.next_batch()
    assert rb._pending is not None and rb.timers["batches"] == 1 and rb.timers["plan"] > 0 and rb.timers["wait"] >= 0
    sd = rb.state_dict()
    assert sd["pending"] is not None and all("gx" not in b and "futs" not in b for b in sd["pending"]["buckets"])
    torch.manual_seed(5)
    want = rb.next_batch()
    rb2 = SM.RealBatcher(cfg, torch.device("cpu"), store, {"a": 0.5, "b": 0.5}, replace(mix, prefetch=False, fetch_workers=0), rows=8, seed=2)
    rb2.load_state_dict(sd)
    torch.manual_seed(5)
    got = rb2.next_batch()                                                               # the restored plan is fetched and built again
    for k in ("x", "mask", "loss_ok", "blank", "gid", "role"):
        assert torch.equal(want[k], got[k]), k
    rb.close()


def test_resume_ignores_io_settings_but_not_data_settings():
    saved = RC.RunConfig().to_json()
    now = RC.RunConfig().to_json()
    now["mix"] = dict(now["mix"], fetch_workers=24, threads_per_worker=1, io="mmap", prefetch=False)
    assert T.config_mismatch(saved, now) == {}
    now["mix"] = dict(now["mix"], real_fraction=0.5)
    assert set(T.config_mismatch(saved, now)) == {"mix.real_fraction"}
    old = {k: v for k, v in saved.items() if k not in ("shards_dir",)}
    old["mix"] = {k: v for k, v in saved["mix"].items() if k not in ("io", "fetch_workers")}          # a checkpoint of the earlier version
    assert T.config_mismatch(old, saved) == {}
    assert "peak_lr" in T.config_mismatch(saved, dict(saved, peak_lr=2e-4))


def test_benchmark_sweep_and_comparison_run(world, capsys):  # noqa: F811
    base = ["--real-root", world.real_root, "--ckpt", world.ckpt, "--rows", "24", "--steps", "2", "--warm", "1", "--device", "cpu"]
    assert BL.main(base + ["--sweep"]) == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip() and ln.split()[0].isdigit()]
    assert {ln.split()[0] for ln in lines} == {"1", "4", "8", "12", "16", "24"}
    assert sum("skipped" in ln for ln in lines) == 2                                      # 16 x 2 = 32 and 24 x 2 = 48 threads
    assert sum("skipped" not in ln for ln in lines) == 10 and "fastest:" in out and "real pts/s/GPU" in out
    assert BL.main(base + ["--compare-legacy"]) == 0
    out = capsys.readouterr().out
    assert "first loader (mmap, 1 thread, no prefetch)" in out and "new (" in out
    assert BL.main(base + ["--sim-step-ms", "5"]) == 0
    assert "wait for reads" in capsys.readouterr().out


# ---- real200b
def test_real200b_preset():
    rc = RC.real200b_config().resolve()
    assert rc.name == "real200b" and rc.target_points == 200e9 and rc.shards_dir == "shards_full"
    assert rc.milestones == tuple(20e9 * i for i in range(1, 11)) and rc.plan.endswith("plan_200b.json")
    assert rc.mix.rows == 288 and rc.mix.real_fraction == 0.5 and rc.mix.split_rows(8) == (144, 144)
    assert (rc.peak_lr, rc.warmup_frac, rc.final_lr_frac, rc.weight_decay, rc.grad_clip) == (1.2e-4, 0.05, 0.10, 0.01, 1.0)
    assert T.lr_at(200e9, rc) == pytest.approx(1.2e-5) and T.lr_at(10e9, rc) == pytest.approx(1.2e-4)         # warmup ends at 5% = 10B
    sm = RC.smoke_config(RC.real200b_config(), RC.SMOKE_POINTS["real200b"])
    assert sm.name == "smoke2b" and sm.target_points == 2e9 and sm.milestones == (1e9, 2e9) and sm.mix.real_fraction == 0.5
    assert RC.smoke_config().name == "smoke1b" and RC.RunConfig().mix.real_fraction == pytest.approx(1 / 3)       # real30b unchanged


def test_plan_non_weather_at_cap(tmp_path, capsys):
    import json
    uniq = {"dominick": 0.5e9, "wind_farms_hourly": 0.075e9, "weatherbench_daily": 3e9, "weatherbench_hourly_geopotential": 70e9,
            "weatherbench_weekly": 1e9}
    man = tmp_path / "idx.json"
    man.write_text(json.dumps({"datasets": {k: {"total_points": v} for k, v in uniq.items()}}))
    out = tmp_path / "p.json"
    assert P.main(["--manifest", str(man), "--real-points", "100e9", "--max-epochs", "4", "--non-weather-at-cap", "--weather-split", "none",
                   "--out", str(out)]) == 0
    pl = json.loads(out.read_text())["datasets"]
    assert pl["dominick"]["epochs"] == pytest.approx(4) and pl["wind_farms_hourly"]["epochs"] == pytest.approx(4)
    w = [pl[k]["epochs"] for k in pl if k.startswith("weatherbench")]
    assert max(w) - min(w) < 1e-9 and w[0] == pytest.approx((100e9 - 4 * 0.575e9) / 74e9)               # weather: the rest, equal passes
    assert sum(r["planned_points"] for r in pl.values()) == pytest.approx(100e9)


def make_env(tmp_path):
    scratch = tmp_path / "scratch"
    root = scratch / "tidemark"
    (root / "repo" / "scripts" / "perlmutter").mkdir(parents=True)
    (root / "repo" / "scripts" / "perlmutter" / "env.sh").write_text(f'export ROOT="{root.as_posix()}"\n')
    return scratch, {**os.environ, "SCRATCH": scratch.as_posix(), "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]}


@pytest.mark.skipif(bash is None, reason="needs bash")
def test_launcher_real200b_options(tmp_path):
    scratch, env = make_env(tmp_path)
    script = str(ROOT / "scripts" / "perlmutter" / "real_train_interactive.sh")
    run = lambda *args: subprocess.run([bash, script, *args], capture_output=True, text=True, env=env, timeout=120)  # noqa: E731
    r = run("--run", "real200b", "--smoke", "--dry-run")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "--run real200b --smoke" in r.stdout and "smoke2b" in r.stdout and "--shards-dir shards_full" in r.stdout and "plan_200b.json" in r.stdout
    r = run("--run", "real200b", "--dry-run")
    assert "--name real200b" in r.stdout and "real_run/real200b" in r.stdout
    r = run("--run", "real200b", "--sweep", "--dry-run")
    assert "torchrun --standalone --nproc_per_node=4 -m tidemark.real.bench_loader --sweep" in r.stdout and "--real-fraction 0.5" in r.stdout
    assert run("--run", "nope", "--dry-run").returncode == 2
    r30 = run("--smoke", "--dry-run")                                                         # real30b unchanged
    assert "--run real30b --smoke" in r30.stdout and "smoke1b" in r30.stdout and "--shards-dir shards " in r30.stdout


def test_sbatch_200b():
    from tidemark.pretrain import slurm
    text = slurm.real_script(False, "real200b")
    assert "#SBATCH -q regular" in text and "#SBATCH -t 08:00:00" in text and "#SBATCH --gpus-per-node=4" in text and "#SBATCH --requeue" in text
    assert "#SBATCH --signal=B:TERM@120" in text and "--run real200b" in text and "plan_200b.json" in text and "shards_full" in text
    assert "scontrol requeue" in text and "trap on_term TERM USR1" in text
    assert (ROOT / "scripts" / "perlmutter" / "real_200b.sbatch").read_text() == text
    with pytest.raises(ValueError):
        slurm.real_script(True, "real200b")
