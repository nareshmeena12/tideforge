"""The committed .sbatch files are exactly what tideforge/slurm.py writes."""
from tideforge import slurm


def test_sbatch_files_match_the_generator():
    assert slurm.main(["--check"]) == 0


import pytest


@pytest.mark.parametrize("over", [dict(qos="shared", gpus=3, cpus=96), dict(qos="shared", gpus=2, cpus=96),
                                  dict(qos="regular", gpus=5, cpus=128), dict(qos="regular", gpus=4, cpus=160),
                                  dict(qos="regular", gpus=1, cpus=64), dict(qos="regular", time="49:00:00"),
                                  dict(qos="premium_x")])
def test_check_refuses_requests_above_the_qos(over):
    job = dict(slurm.JOBS["e03_score"], **over)
    with pytest.raises(ValueError):
        slurm.check(job)


def test_check_accepts_the_limits():
    slurm.check(dict(qos="shared", gpus=2, cpus=64, time="48:00:00"))
    slurm.check(dict(qos="regular", gpus=4, cpus=128, time="06:00:00"))


def test_e03_default_is_a_full_regular_node_and_fits_its_time():
    j = slurm.JOBS["e03_score"]
    assert (j["qos"], j["gpus"], j["cpus"]) == ("regular", 4, 128)
    assert slurm.makespan([2.5, 2.5, 2.0, 2.0, 1.5], 4) == 3.5
    assert "GPU-h" in slurm.estimate("e03_score")


def test_e03_units_match_the_launcher():
    sh = (slurm.OUT / "e03_nan.sh").read_text(encoding="utf-8")
    assert 'UNITS="' + " ".join(slurm.UNITS_H["e03_score"]) + '"' in sh
