"""The catalog builder reproduces the committed catalog exactly (owner 2026-10-31): a rebuild can never undo a hand fix,
because every hand fix lives in the generator too (catalog/CHANGELOG.md)."""
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

CAT = Path(__file__).resolve().parents[1] / "catalog"
OUTPUTS = ("dataset_catalog_v2.csv", "excluded_benchmark.csv", "benchmark_datasets.csv")


def test_expand_catalog_output_equals_the_committed_csvs(tmp_path):
    for f in ("expand_catalog.py", "dataset_catalog_v1.csv"):
        shutil.copy(CAT / f, tmp_path / f)
    subprocess.run([sys.executable, "-I", "expand_catalog.py"], cwd=tmp_path, check=True, capture_output=True)
    for f in OUTPUTS:
        built = pd.read_csv(tmp_path / f, dtype=str, keep_default_na=False)
        committed = pd.read_csv(CAT / f, dtype=str, keep_default_na=False)
        pd.testing.assert_frame_equal(built, committed, obj=f)
