"""The generated jobs are valid bash and run the check steps in order."""
import shutil
import subprocess

import pytest

from tideforge.score import slurm as SL


def test_check_and_score_jobs(tmp_path, monkeypatch):
    monkeypatch.setenv("NERSC_ACCOUNT", "m0000_g")
    monkeypatch.setenv("TF_ROOT", str(tmp_path))
    assert SL.main(["check"]) == 0
    t = (tmp_path / "jobs" / "score_check.sbatch").read_text()
    order = [t.index(s) for s in ("profile_fev", "--fev-bench", "--groups both", "--tier full")]
    assert order == sorted(order) and "#SBATCH --gpus=1" in t
    assert SL.main(["score", "--model", "/x/final.pt", "--model-id", "r13", "--tier", "core"]) == 0
    s = (tmp_path / "jobs" / "score_r13_core.sbatch").read_text()
    assert "--tier core" in s and "--locked" not in s
    if shutil.which("bash"):
        for f in (tmp_path / "jobs").glob("*.sbatch"):
            r = subprocess.run(["bash", "-n"], input=f.read_bytes(), capture_output=True)
            assert r.returncode == 0, r.stderr.decode()
    assert SL.main(["rescale", "--models", "ft3_main=/a/ft3.pt,r11=/b/r11.pt"]) == 0
    r = (tmp_path / "jobs" / "score_rescale.sbatch").read_text()
    assert r.count("diag_scale --ckpt") == 2 and r.count("--rescale off") == 2 and "--fev-bench" in r
    if shutil.which("bash"):
        assert subprocess.run(["bash", "-n"], input=r.encode(), capture_output=True).returncode == 0
    with pytest.raises(SystemExit):
        SL.main(["rescale", "--models", "bad id=/x.pt"])
    with pytest.raises(ValueError):
        SL.header("x", "4h", "a")
