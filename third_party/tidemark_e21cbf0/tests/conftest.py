import pytest
import torch

from tidemark.models.gla import fla_available

HAS_GPU_FLA = torch.cuda.is_available() and fla_available()


def pytest_collection_modifyitems(config, items):
    if HAS_GPU_FLA:
        return
    skip = pytest.mark.skip(reason="needs CUDA + fla Triton kernels (run in the Colab notebook)")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
