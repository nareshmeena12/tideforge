"""T3 on a GPU (Perlmutter; skipped without CUDA). The generator runs in deterministic mode, the model in torch's default mode
(E0.9 decision), and generation never overlaps a model kernel launch (owner 2026-10-29: the flag is process-global).

1. test_gpu_50_steps_det_generation: 50 training steps of the 5M model (torch.compile on, pack_rows 64) with generation in
   strict deterministic mode between the steps: no error (strict mode raises on index_add_ / scatter backward if the flag
   leaked into the model's backward), the flag is off again after every generation, the loss is finite.
2. test_gpu_batches_bit_identical: two batchers from the same seed in fresh state produce bit-identical batches on the GPU
   (pools generated in deterministic mode).
Run: CUBLAS_WORKSPACE_CONFIG=:4096:8 python -m pytest -q -s -rA -m gpu tests/train/test_train_gpu.py
"""
import os

import pytest
import torch

from tideforge.expcfg import load_exp
from tideforge.train import recipe as RC
from tideforge.train import trainer as TR

pytestmark = pytest.mark.gpu


@pytest.fixture(autouse=True)
def model_in_default_mode():
    """tests/conftest.py puts gpu tests in deterministic mode; here only the generator is (the trainer's setting)."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    before = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(before)


def small(exp="E0.7"):
    r = RC.load_recipe()
    r["batch"].update(pack_rows=64)
    r["run"].update(val_batches=1, val_every=10 ** 9, print_every=10, ckpt_every_s=1e9)
    return RC.merge(load_exp(exp), r)


def test_gpu_50_steps_det_generation(tmp_path):
    r = TR.Run(small(), 0, tmp_path, "cuda", "gputest", log=print).build()
    assert r.det and r.n_params == 5_447_968
    r.n_bad = 0
    losses = []
    for _ in range(50):
        losses.append(r.one_step())
        assert not torch.are_deterministic_algorithms_enabled()
    torch.cuda.synchronize()
    print(f"\n50 steps: loss {losses[0]:.4f} -> {losses[-1]:.4f}, generation {r.batcher.gen_seconds:.1f} s, bad {r.n_bad}")
    assert r.n_bad == 0 and all(l == l for l in losses)


def test_gpu_batches_bit_identical(tmp_path):
    def batches(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        r = TR.Run(small(), 0, tmp_path / f"r{seed}", "cuda", "gputest", log=print).build()
        return [r.batcher.next_batch() for _ in range(3)]
    a, b = batches(0), batches(0)
    for x, y in zip(a, b):
        for k in ("x", "mask", "blank", "role", "gid", "seg_patch"):
            assert torch.equal(x[k], y[k]), k
