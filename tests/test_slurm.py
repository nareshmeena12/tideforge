"""The committed .sbatch files are exactly what tideforge/slurm.py writes."""
from tideforge import slurm


def test_sbatch_files_match_the_generator():
    assert slurm.main(["--check"]) == 0
