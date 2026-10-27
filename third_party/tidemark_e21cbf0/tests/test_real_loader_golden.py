"""The real loader's batches must not change when its I/O is restructured (threads, pread, prefetch): a frozen output of the
first loader (memmap, synchronous) on a small fixed store. Regenerate ONLY by deliberate decision:
    python tests/test_real_loader_golden.py --write      (run on the OLD code)
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent))

from test_real_eval import tiny_cfg  # noqa: E402
from test_real_sampler import make_store  # noqa: E402

from tidemark.real import ft3_core as C  # noqa: E402
from tidemark.real import sampler as SM  # noqa: E402

GOLDEN = Path(__file__).parent / "golden" / "real_loader_v1.npz"
KEYS_INT = ("seg_patch", "gid", "role", "slot_valid", "slot_ctx_len", "slot_loss", "blank", "mask", "loss_ok")


def run_batches(tmp: Path, n: int = 5, **mix_kw):
    spec = {"a": (30, 600, 0.0), "b": (30, 300, 0.2), "c": (12, 1500, 0.05)}
    store = make_store(tmp, spec)
    cfg = dict(tiny_cfg(), pack_rows=24)
    C.set_group_options(cfg)
    torch.manual_seed(123)
    mb = SM.MixedBatcher(cfg, torch.device("cpu"), store, {"a": 0.4, "b": 0.3, "c": 0.3}, SM.MixConfig(rows=24, real_fraction=1 / 3, **mix_kw), seed=5)
    out = [mb.next_batch() for _ in range(n)]
    return out, mb


def flatten(batches):
    d = {}
    for i, b in enumerate(batches):
        d[f"x{i}"] = b["x"].numpy()
        for k in KEYS_INT:
            d[f"{k}{i}"] = b[k].numpy()
        d[f"points{i}"] = np.array([b["points"], b["points_real"], b["points_syn"], b["n_series"]])
    return d


def write():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        batches, mb = run_batches(Path(t))
        d = flatten(batches)
        GOLDEN.parent.mkdir(exist_ok=True)
        np.savez_compressed(GOLDEN, **d)
        print("written", GOLDEN, len(d), "arrays")


@pytest.mark.parametrize("mix_kw", [dict(), dict(fetch_workers=4, io="pread", prefetch=True), dict(fetch_workers=0, io="mmap", prefetch=False)])
def test_batches_equal_the_first_loader(tmp_path, mix_kw):
    if mix_kw and not hasattr(SM.MixConfig(), "fetch_workers"):
        pytest.skip("the loader has no I/O options yet")
    batches, mb = run_batches(tmp_path, **mix_kw)
    got, want = flatten(batches), np.load(GOLDEN)
    assert set(want.files) == set(got)
    for k in got:
        if k.startswith("x"):
            np.testing.assert_allclose(got[k], want[k], rtol=1e-5, atol=1e-6, err_msg=k)
        else:
            np.testing.assert_array_equal(got[k], want[k], err_msg=k)


if __name__ == "__main__":
    if "--write" in sys.argv:
        write()
