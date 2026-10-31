"""Offline trajectory pools (G9.1 ODEs; optional for G9.2 tar and G10.1 garch_jump): built once on Perlmutter, cropped at
generation. Never committed; only the manifest is (tideforge/gen/ext/pools/<name>.json).

A pool is a directory <pool_dir>/<name>/ with
    traj.npy   float16 (N, T, C): N trajectories of T steps, C channels (unused channels of a system are 0)
    meta.npy   float32 (N, M): per trajectory: system id, number of channels used, then the system's parameters
and a manifest <name>.json: {"name", "content": {file: content sha256}, "files": {file: file sha256}, "shape", "builder",
"seed", "versions", ...}. .npy (not .npz): the files are memory-mapped (no copy per DataLoader worker). load() refuses a pool
whose arrays' content sha256 (tideforge.hashing: dtype, shape, values) differs from the committed manifest; the file sha256
is information only (numpy versions write different .npy headers for the same array; owner 2026-10-31).

pool_dir default: $TF_DATA/pools. manifest_dir default: this package's pools/ directory (the committed manifests).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch

from tideforge import hashing as H

MANIFESTS = Path(__file__).resolve().parent / "pools"
_CACHE: dict[tuple[str, str, str], dict] = {}


def sha256(path: Path) -> str:
    """The file's bytes (information only)."""
    return H.file_sha(path)


def content(path: Path) -> str:
    """The array's content sha256 (what load() checks)."""
    return H.content_sha({Path(path).stem: np.load(path, mmap_mode="r")})


def default_dir() -> Path:
    d = os.environ.get("TF_DATA")
    if not d:
        raise RuntimeError("pool_dir not given and $TF_DATA is not set (pools live in $TF_DATA/pools)")
    return Path(d) / "pools"


def write(out_dir: Path, name: str, traj: np.ndarray, meta: np.ndarray, info: dict, manifest_dir: Path) -> dict:
    """Save a pool and write its manifest (builders only)."""
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "traj.npy", traj.astype(np.float16))
    np.save(d / "meta.npy", meta.astype(np.float32))
    fs = ("traj.npy", "meta.npy")
    man = dict(name=name, content={f: content(d / f) for f in fs}, files={f: sha256(d / f) for f in fs},
               shape=list(traj.shape), **info)
    Path(manifest_dir).mkdir(parents=True, exist_ok=True)
    (Path(manifest_dir) / f"{name}.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n", newline="\n")
    return man


def load(name: str, pool_dir: str = "", manifest_dir: str = "") -> dict:
    """{"traj": memmap (N, T, C) float16, "meta": (N, M) float32, "manifest": dict}; the sha check runs once per process."""
    pd = Path(pool_dir) if pool_dir else default_dir()
    md = Path(manifest_dir) if manifest_dir else MANIFESTS
    key = (name, str(pd), str(md))
    if key in _CACHE:
        return _CACHE[key]
    mf = md / f"{name}.json"
    if not mf.exists():
        raise RuntimeError(f"pool {name!r}: no committed manifest {mf} (build the pool, then commit its manifest)")
    man = json.loads(mf.read_text())
    if "content" not in man:
        raise RuntimeError(f"pool {name!r}: manifest {mf} has no content sha256 (built before 2026-10-31): rebuild it")
    for f, want in man["content"].items():
        p = pd / name / f
        if not p.exists():
            raise RuntimeError(f"pool {name!r}: {p} is missing (build it with the command in the manifest's 'builder')")
        got = content(p)
        if got != want:
            raise RuntimeError(f"pool {name!r}: {p} has content sha256 {got}, the manifest says {want}: refusing to generate")
    out = dict(traj=np.load(pd / name / "traj.npy", mmap_mode="r"), meta=np.load(pd / name / "meta.npy"), manifest=man)
    _CACHE[key] = out
    return out


def crop(g: torch.Generator, pool: dict, n: int, L: int, dev, stride_choices=(1,), rows=None):
    """n random crops of L steps (every s-th step, s from stride_choices with s L <= T; s = 1 if none fits).
    Returns x (n, L, C) float32 on dev, the trajectory index (n,) and the stride (n,). Draws from g (CPU indices)."""
    traj = pool["traj"]
    N, T, _ = traj.shape
    if L > T:
        raise ValueError(f"pool {pool['manifest']['name']!r}: crops of {L} steps, trajectories have {T}")
    ok = [s for s in stride_choices if s * L <= T] or [1]
    cg = torch.Generator().manual_seed(int(torch.randint(0, 2 ** 62, (1,), generator=g, device=g.device)))
    idx = torch.randint(0, N, (n,), generator=cg) if rows is None else torch.as_tensor(rows)[torch.randint(0, len(rows), (n,), generator=cg)]
    s = torch.tensor(ok)[torch.randint(0, len(ok), (n,), generator=cg)]
    start = (torch.rand(n, generator=cg) * (T - s * (L - 1))).long().clamp(max=T - 1)
    order = np.argsort(idx.numpy(), kind="stable")                 # sorted reads from the memmap
    out = np.empty((n, L, traj.shape[2]), np.float32)
    for i in order:
        a, st = int(start[i]), int(s[i])
        out[i] = traj[int(idx[i]), a:a + st * (L - 1) + 1:st]
    return torch.from_numpy(out).to(dev), idx.to(dev), s.to(dev)
