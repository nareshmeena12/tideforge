"""Content hashes of committed arrays (owner 2026-10-31).

A file's bytes are not a stable identity across machines: zlib builds compress the same data to different deflate streams,
and numpy versions write different .npy headers (the calendar table: Perlmutter's rebuild had another file sha256 and the
same content). So every check of a built table, pool or probe set compares content_sha256, a sha256 over each array's name,
dtype, shape and raw values, in name order; file_sha256 stays in the manifests as information only. Arrays are hashed in
chunks along the first axis, so a memory-mapped pool is never read into memory at once.
"""
from __future__ import annotations

import hashlib

import numpy as np

CHUNK_BYTES = 1 << 26


def update_array(h, name: str, x) -> None:
    x = np.asarray(x)
    h.update(f"{name}|{x.dtype.str}|{x.shape}|".encode())
    if x.ndim == 0 or x.size == 0:
        h.update(np.ascontiguousarray(x).tobytes())
        return
    row = max(1, x.nbytes // max(1, x.shape[0]))
    step = max(1, CHUNK_BYTES // row)
    for i in range(0, x.shape[0], step):
        h.update(np.ascontiguousarray(x[i:i + step]).tobytes())


def content_sha(arrays: dict) -> str:
    """sha256 over {name: array} (names sorted): independent of zip, deflate and .npy header bytes."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        update_array(h, name, arrays[name])
    return h.hexdigest()


def file_sha(path) -> str:
    """sha256 of a file's bytes (information only; see the module docstring)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()
