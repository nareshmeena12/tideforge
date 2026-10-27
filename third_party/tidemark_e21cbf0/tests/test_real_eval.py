"""ft3 checkpoint loading (strict), forward parity against the reference definitions, and the baseline table (tiny model on CPU)."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import torch

from tidemark.real import ft3_core as C
from tidemark.real import score_ft3 as S

ROOT = Path(__file__).parents[1]
REF = ROOT / "reference" / "tide_v5_ft3.py"


def tiny_cfg():
    return dict(C.CONFIG, d_model=64, d_kv=16, n_heads=4, d_ff=128, n_layers=2, max_ctx=256, pack_rows=16, group_lanes=8,
                len_bucket_edges=[32, 64, 128, 256], long_min_ctx=64, gen_pool_points=2e5, gen_pool_max_rows=512,
                max_segs_per_row=8, cov_r3=True, dropout=0.1)


@pytest.fixture(scope="module")
def tiny_ckpt(tmp_path_factory):
    cfg = tiny_cfg()
    torch.manual_seed(0)
    m = C.build_model(dict(cfg, dropout=0.0))
    p = tmp_path_factory.mktemp("ck") / "ft3_tiny.pt"
    torch.save(dict(model=m.state_dict(), cfg=cfg, step=7, points_seen=1e9, packed=True), p)
    return p


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "real" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_strict_load_and_a_key_mismatch_is_refused(tiny_ckpt, tmp_path):
    model, cfg, ck = S.load_ft3(tiny_ckpt, torch.device("cpu"))
    assert not model.training and cfg["d_model"] == 64 and ck["step"] == 7
    bad = torch.load(tiny_ckpt, weights_only=False)
    bad["model"] = {k: v for k, v in list(bad["model"].items())[:-1]}
    torch.save(bad, tmp_path / "bad.pt")
    with pytest.raises(RuntimeError):                      # torch's strict load names the missing key
        S.load_ft3(tmp_path / "bad.pt", torch.device("cpu"))


def test_forward_is_finite_and_deterministic(tiny_ckpt):
    cf = load_script("check_forward")
    model, cfg, _ = S.load_ft3(tiny_ckpt, torch.device("cpu"))
    b1 = cf.fixed_batch(C, cfg, torch.device("cpu"))
    b2 = cf.fixed_batch(C, cfg, torch.device("cpu"))
    assert cf.batches_equal(b1, b2) == []
    o1, o2 = cf.run_forward(model, b1, C.packed_forward), cf.run_forward(model, b2, C.packed_forward)
    assert torch.isfinite(o1["loss"]) and torch.equal(o1["_q"], o2["_q"])


@pytest.mark.skipif(not REF.exists(), reason="reference/ is git-ignored and absent on Perlmutter")
def test_parity_with_the_reference_definitions(tiny_ckpt, tmp_path):
    cf = load_script("check_forward")
    assert cf.main(["--ckpt", str(tiny_ckpt), "--reference", str(REF), "--device", "cpu", "--save-golden", str(tmp_path / "g.pt")]) == 0
    assert cf.main(["--ckpt", str(tiny_ckpt), "--golden", str(tmp_path / "g.pt"), "--device", "cpu"]) == 0


def test_golden_mismatch_fails(tiny_ckpt, tmp_path):
    cf = load_script("check_forward")
    cf.main(["--ckpt", str(tiny_ckpt), "--device", "cpu", "--save-golden", str(tmp_path / "g.pt")])
    g = torch.load(tmp_path / "g.pt", weights_only=False)
    g["loss"] += 1.0
    torch.save(g, tmp_path / "g2.pt")
    assert cf.main(["--ckpt", str(tiny_ckpt), "--golden", str(tmp_path / "g2.pt"), "--device", "cpu"]) == 1


def test_forecast_adapter_runs_on_cpu(tiny_ckpt):
    model, _, _ = S.load_ft3(tiny_ckpt, torch.device("cpu"))
    import numpy as np
    series = [np.sin(np.arange(200) / 5).astype("float32"), np.arange(70, dtype="float32")]
    pt, q = C.forecast_left_padded(model, series, 16, torch.device("cpu"), max_ctx=256, batch_size=4)
    assert pt.shape == (2, 16) and q.shape == (2, 9, 16) and (q[:, 1:] >= q[:, :-1] - 1e-5).all()


def test_ckpt_names():
    assert S.parse_ckpts(["ft3main=/x/a.pt", "p10b=/y.pt"]) == {"ft3main_s0": "/x/a.pt", "p10b_s0": "/y.pt"}
    for bad in ("ft3_main=/x", "noequals", "UPPER=/x"):
        with pytest.raises(SystemExit):
            S.parse_ckpts([bad])


def test_table_change_and_expectations(tmp_path):
    board = tmp_path / "eval" / "board"
    board.mkdir(parents=True)
    pd.DataFrame({"MASE (norm.)": [0.687, 0.667], "CRPS (norm.)": [0.575, 0.560]}, index=["ft3main_s0", "p10b_s0"]).to_csv(board / "time_board.csv")
    pd.DataFrame({"skill_score": [0.4209, 0.4241], "win_rate": [0.60, 0.61]}, index=["ft3main_s0", "p10b_s0"]).to_csv(board / "fev_board.csv")
    df = S.results_table(tmp_path, ["ft3main_s0", "p10b_s0"], "ft3main")
    row = df[df.checkpoint == "p10b"].iloc[0]
    assert row["TIME MASE change"] == pytest.approx(100 * (0.667 / 0.687 - 1)) and row["fev skill_score (x100) change"] == pytest.approx(0.32)
    lines = S.check_expectations(df, "ft3main", 0.687, 42.09, time_crps=0.575)
    assert "PASS" in lines[0] and "PASS" in lines[1] and any("matches 42.09" in l for l in lines)
    assert sum("fev" in l for l in lines) == 1 and not any("win_rate" in l for l in lines)           # win rate is not compared
    assert "FAIL" in S.check_expectations(df, "ft3main", 0.690, 42.09)[0]


# ---- the shell script, dry run only
import os
import shutil
import subprocess
import sys

bash = shutil.which("bash")


@pytest.mark.skipif(bash is None, reason="needs bash")
def test_real_eval_dry_run(tmp_path):
    scratch = tmp_path / "scratch"
    root = scratch / "tidemark"
    (root / "repo" / "scripts" / "perlmutter").mkdir(parents=True)
    (root / "repo" / "scripts" / "perlmutter" / "env.sh").write_text(f'export ROOT="{root.as_posix()}"\n')
    script = ROOT / "scripts" / "perlmutter" / "real_eval.sh"
    env = {**os.environ, "SCRATCH": scratch.as_posix(), "PATH": os.path.dirname(sys.executable) + os.pathsep + os.environ["PATH"]}
    r = subprocess.run([bash, str(script), "--baseline", "--dry-run"], capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    assert "check_forward.py" in out and "score_ft3 score" in out and "ft3main=/pscratch/sd/n/naresh/tidemark/ckpt/ft3_main.pt" in out
    assert "--expect-time 0.687 --expect-crps 0.575 --expect-fev 42.09" in out and "size_real" in out
    both = subprocess.run([bash, str(script), "--baseline", "--both", "--dry-run"], capture_output=True, text=True, env=env, timeout=120)
    assert both.returncode == 0, both.stdout + both.stderr
    assert "score_groups score" in both.stdout and "--modes on,off" in both.stdout and "score_groups table" in both.stdout         and "--gate" in both.stdout and "compare-log" in both.stdout and "ON + OFF" in both.stdout
    lst = subprocess.run([bash, str(script), "--both", "--dry-run", "p10b=/x/a.pt", "p20b=/x/b.pt"], capture_output=True, text=True, env=env, timeout=120)
    assert "--ckpt p10b=/x/a.pt --ckpt p20b=/x/b.pt" in lst.stdout and "--gate" not in lst.stdout
    assert subprocess.run([bash, str(script), "--bogus"], capture_output=True, text=True, env=env).returncode == 2
    assert subprocess.run([bash, str(script)], capture_output=True, text=True, env=env).returncode == 2
