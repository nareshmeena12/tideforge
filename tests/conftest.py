"""Puts the repo root and the pinned tidemark copy on sys.path; skips `gpu` tests without CUDA and `slow` tests without
--runslow."""
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
THIRD_PARTY = REPO / "third_party" / "tidemark_e21cbf0"
for p in (REPO, THIRD_PARTY):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="also run the tests marked slow (minutes on CPU)")


def pytest_collection_modifyitems(config, items):
    gpu = pytest.mark.skip(reason="needs CUDA (run on Perlmutter: pytest -m gpu)")
    slow = pytest.mark.skip(reason="slow: run with --runslow")
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(gpu)
        if "slow" in item.keywords and not config.getoption("--runslow"):
            item.add_marker(slow)
