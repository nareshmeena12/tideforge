"""Compact memory-mapped shards of the downloaded real datasets, and window access to them.

Layout under <real_root>/shards/<dataset>/:  values.f32 (all series back to back, float32, interior gaps forward-filled),
missing.i64 (sorted positions in values.f32 that were originally missing), offsets.i64 (n_series + 1), meta.json.
<real_root>/shards/index.json lists the datasets with their unique points (what the plan and the loader use).

Missing values (owner, 2026-10-24, second rule): interior missing values are FORWARD-FILLED in the stored inputs, as the scorers do
(forecast_groups._lane_array) and as ft3_core._L_corruption does for synthetic data, and the point is observed (mask True) for the
model's input. The originally missing positions are kept in missing.i64 (sorted global positions in values.f32) so the loader can
build a separate LOSS mask that leaves them out of the loss. Non-finite values count as missing. Cleaning: leading and trailing
missing values are trimmed (a series starts and ends on a valid point); finite values are clamped to +-1e9 (ft3_core._CLAMP); series
with fewer than `min_len` VALID points (default 33 = patch_len + 1, the shortest context the causal decoder can use) are dropped, with
all-missing series. Only VALID (originally observed) points count as a dataset's unique points (epoch caps). See
tidemark/REAL_NOTES.md.
Nothing else is filtered: whole datasets are in or out (tidemark/real/allowlist.py).
"""
from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

CLAMP = 1e9
MIN_LEN = 33


def clean_series(a: np.ndarray) -> tuple[np.ndarray | None, np.ndarray, int]:
    """(float32 series trimmed to its first and last valid point with interior gaps FORWARD-FILLED, positions that were missing,
    number of valid points); (None, empty, 0) if nothing is valid. Non-finite values are missing; finite values are clamped."""
    a = np.asarray(a, dtype=np.float64)
    ok = np.isfinite(a)
    if not ok.any():
        return None, np.zeros(0, dtype=np.int64), 0
    first, last = int(np.argmax(ok)), len(ok) - 1 - int(np.argmax(ok[::-1]))
    a, ok = a[first:last + 1], ok[first:last + 1]
    miss = np.flatnonzero(~ok).astype(np.int64)
    if len(miss):
        idx = np.where(ok, np.arange(len(a)), 0)
        np.maximum.accumulate(idx, out=idx)
        a = a[idx]
    return np.clip(a, -CLAMP, CLAMP).astype(np.float32), miss, int(ok.sum())


def _list_columns(batch):
    import pyarrow as pa
    for name, col in zip(batch.schema.names, batch.columns):
        t = col.type
        if name == "timestamp" or not (pa.types.is_list(t) or pa.types.is_large_list(t)):
            continue
        if pa.types.is_floating(t.value_type) or pa.types.is_integer(t.value_type):
            yield name, col


def _batches(path: Path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    if path.suffix == ".parquet":
        yield from pq.ParquetFile(path).iter_batches(batch_size=512)
        return
    with pa.memory_map(str(path)) as src:
        try:
            yield from pa.ipc.open_stream(src)
        except pa.ArrowInvalid:
            reader = pa.ipc.open_file(src)
            for i in range(reader.num_record_batches):
                yield reader.get_batch(i)


def _convert_file(path, fv, min_len: int) -> dict:
    """Convert one arrow / parquet file, appending the kept series to the open binary file `fv`. Returns the lengths and valid counts
    of the kept series, the originally missing positions relative to the start of THIS file's output, and the counters."""
    import pyarrow as pa
    import pyarrow.compute as pc
    lens: list[int] = []
    valid: list[int] = []
    miss_parts: list[np.ndarray] = []
    stored = 0
    stat = dict(series_in=0, dropped_short=0, dropped_empty=0, points_in=0, nan_points_in=0)
    for b in _batches(Path(path)):
        for _, col in _list_columns(b):
            ln = pc.fill_null(pc.list_value_length(col), 0).to_numpy(zero_copy_only=False).astype(np.int64)
            flat = pc.list_flatten(col).cast(pa.float64()).to_numpy(zero_copy_only=False)
            bounds = np.concatenate([[0], np.cumsum(ln)])
            for i in range(len(ln)):
                o, e = int(bounds[i]), int(bounds[i + 1])
                stat["series_in"] += 1
                stat["points_in"] += e - o
                if e - o == 0:
                    stat["dropped_empty"] += 1
                    continue
                arr, miss, nv = clean_series(flat[o:e])
                stat["nan_points_in"] += (e - o) - int(np.isfinite(flat[o:e]).sum())
                if arr is None:
                    stat["dropped_empty"] += 1
                    continue
                if nv < min_len:
                    stat["dropped_short"] += 1
                    continue
                arr.tofile(fv)
                if len(miss):
                    miss_parts.append(miss + stored)
                stored += len(arr)
                lens.append(len(arr))
                valid.append(nv)
    return {"lens": np.asarray(lens, dtype=np.int64), "valid": np.asarray(valid, dtype=np.int64),
            "miss": np.concatenate(miss_parts) if miss_parts else np.zeros(0, dtype=np.int64), "stored": stored, "stat": stat}


def _convert_part(args) -> dict:
    """Worker: convert one file into its own part file (merged in order by the parent)."""
    path, part, min_len = args
    with open(part, "wb") as fv:
        return _convert_file(path, fv, min_len)


def convert_dataset(name: str, files: list[Path], out_dir: Path, min_len: int = MIN_LEN, meta_extra: dict | None = None,
                    workers: int = 1, log=None) -> dict:
    """Stream the dataset's arrow / parquet files into values.f32 + offsets.i64 + missing.i64. Rewrites the dataset's folder.
    workers > 1: files are converted in parallel into part files that are appended to values.f32 in file order (each part is deleted as
    soon as it is appended, so the extra disk is about one part per worker); the result is identical to workers = 1."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vpath, tmp = out_dir / "values.f32", out_dir / "values.f32.tmp"
    lens_l: list[np.ndarray] = []
    valid_l: list[np.ndarray] = []
    miss_l: list[np.ndarray] = []
    stored = 0
    stat = dict(series_in=0, dropped_short=0, dropped_empty=0, points_in=0, nan_points_in=0)

    def take(res: dict) -> None:
        nonlocal stored
        lens_l.append(res["lens"])
        valid_l.append(res["valid"])
        if len(res["miss"]):
            miss_l.append(res["miss"] + stored)
        stored += res["stored"]
        for k, v in res["stat"].items():
            stat[k] += v

    t0 = time.time()
    if workers <= 1 or len(files) <= 1:
        with open(tmp, "wb") as fv:
            for i, f in enumerate(files):
                take(_convert_file(f, fv, min_len))
                if log:
                    log(f"    {name}: file {i + 1}/{len(files)}  {stored:,} points stored  {time.time() - t0:.0f} s")
    else:
        import multiprocessing as mp
        parts = [out_dir / f"part{i:05d}.f32.tmp" for i in range(len(files))]
        with mp.get_context("spawn").Pool(min(workers, len(files))) as pool, open(tmp, "wb") as fv:
            for i, res in enumerate(pool.imap(_convert_part, [(f, parts[i], min_len) for i, f in enumerate(files)])):
                take(res)
                with open(parts[i], "rb") as src:
                    shutil.copyfileobj(src, fv, 16 * 2 ** 20)
                parts[i].unlink()
                if log:
                    log(f"    {name}: file {i + 1}/{len(files)}  {stored:,} points stored  {time.time() - t0:.0f} s")
    os.replace(tmp, vpath)
    seconds = time.time() - t0
    lens_a = np.concatenate(lens_l) if lens_l else np.zeros(0, dtype=np.int64)
    valid_a = np.concatenate(valid_l) if valid_l else np.zeros(0, dtype=np.int64)
    offsets = np.concatenate([[0], np.cumsum(lens_a)]).astype(np.int64)
    offsets.tofile(out_dir / "offsets.i64")
    (np.concatenate(miss_l) if miss_l else np.zeros(0, dtype=np.int64)).astype(np.int64).tofile(out_dir / "missing.i64")
    assert stored == int(lens_a.sum())
    meta = {"name": name, "n_series": int(len(lens_a)), "unique_points": int(valid_a.sum()), "stored_points": stored,
            "nan_fraction_stored": (1 - int(valid_a.sum()) / stored) if stored else 0.0,
            "min_len": int(lens_a.min()) if len(lens_a) else 0, "max_len": int(lens_a.max()) if len(lens_a) else 0,
            "mean_len": float(lens_a.mean()) if len(lens_a) else 0.0, "min_valid_kept": min_len,
            "files": len(files), "workers": workers, "convert_seconds": round(seconds, 1),
            "points_in_per_s": stat["points_in"] / max(seconds, 1e-9), **stat, **(meta_extra or {})}
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def write_index(shard_root: Path) -> dict:
    idx = {"datasets": {}}
    for d in sorted(Path(shard_root).glob("*/meta.json")):
        m = json.loads(d.read_text())
        idx["datasets"][m["name"]] = {"total_points": m["unique_points"], "stored_points": m["stored_points"],
                                      "nan_fraction": m["nan_fraction_stored"], "n_series": m["n_series"], "min_len": m["min_len"],
                                      "max_len": m["max_len"], "mean_len": m["mean_len"]}
    (Path(shard_root) / "index.json").write_text(json.dumps(idx, indent=1))
    return idx


_HAS_PREAD = hasattr(os, "preadv")          # Linux; on Windows (the dev machine) reads go through the memmap


class RealStore:
    """The converted datasets. `names` fixes the dataset order used everywhere (targets, counters).

    Reading a window of the pool, two ways (`configure_io`):
      "mmap"   the first loader: np.memmap slices (a page fault per window; every fault on a cold file also triggers the kernel's read-ahead)
               and a binary search on the memory-mapped missing.i64 for every window.
      "pread"  one persistent file descriptor per dataset opened with POSIX_FADV_RANDOM (no read-ahead for a random-access pool), os.preadv
               straight into the destination buffer (releases the GIL, so threads read in parallel), and the missing positions held in RAM
               with a pointer per series (a window looks only into its own series' few positions). Same bytes, same positions.
    """

    def __init__(self, shard_root, names: list[str] | None = None, io: str = "mmap"):
        self.root = Path(shard_root)
        avail = sorted(p.parent.name for p in self.root.glob("*/meta.json"))
        self.names = list(names) if names is not None else avail
        missing = [n for n in self.names if n not in avail]
        if missing:
            raise FileNotFoundError(f"not converted under {self.root}: {missing}")
        self.lens, self.offs, self.order, self.sorted_len = [], [], [], []
        for n in self.names:
            d = self.root / n
            off = np.fromfile(d / "offsets.i64", dtype=np.int64)
            ln = np.diff(off)
            self.offs.append(off[:-1])
            self.lens.append(ln)
            o = np.argsort(ln, kind="stable")
            self.order.append(o)
            self.sorted_len.append(ln[o])
        self._vals: dict[int, np.memmap] = {}
        self._fds: dict[int, int] = {}
        self._miss_mm: dict[int, np.ndarray] = {}
        self._miss_ram: dict[int, np.ndarray] = {}
        self._miss_ptr: dict[int, np.ndarray] = {}
        self._lock = __import__("threading").Lock()
        self.io = "mmap"
        self.configure_io(io)

    # ---- I/O mode
    def configure_io(self, io: str) -> None:
        if io not in ("mmap", "pread"):
            raise ValueError(f"io must be 'mmap' or 'pread', not {io!r}")
        self.io = io
        if io == "pread":
            for di in range(len(self.names)):
                self._missing_ram(di)

    def close(self) -> None:
        for fd in self._fds.values():
            try:
                os.close(fd)
            except OSError:
                pass
        self._fds.clear()

    # ---- values
    @property
    def vals(self) -> "_ValsView":
        return _ValsView(self)

    @property
    def miss(self) -> "_MissView":
        return _MissView(self)

    def _memmap(self, di: int) -> np.memmap:
        m = self._vals.get(di)
        if m is None:
            m = self._vals[di] = np.memmap(self.root / self.names[di] / "values.f32", dtype=np.float32, mode="r")
        return m

    def _fd(self, di: int) -> int:
        fd = self._fds.get(di)
        if fd is None:
            with self._lock:
                fd = self._fds.get(di)
                if fd is None:
                    fd = os.open(self.root / self.names[di] / "values.f32", os.O_RDONLY | getattr(os, "O_BINARY", 0))
                    if hasattr(os, "posix_fadvise"):
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)          # random access: no read-ahead of data nobody asked for
                    self._fds[di] = fd
        return fd

    def read_window(self, di: int, ser: int, start: int, c: int, out: np.ndarray) -> None:
        """out[:c] = the c stored points of series `ser` of dataset di from point `start`. Thread-safe."""
        base = int(self.offs[di][ser]) + start
        dst = out[:c]
        if self.io == "pread" and _HAS_PREAD:
            mv = memoryview(dst).cast("B")
            n, off, got, fd = c * 4, base * 4, 0, self._fd(di)
            while got < n:
                r = os.preadv(fd, [mv[got:]], off + got)
                if r <= 0:
                    raise EOFError(f"short read in {self.names[di]}/values.f32 at byte {off + got}")
                got += r
        else:
            dst[:] = self._memmap(di)[base:base + c]

    # ---- missing positions
    def _missing_mm(self, di: int) -> np.ndarray:
        m = self._miss_mm.get(di)
        if m is None:
            mp = self.root / self.names[di] / "missing.i64"
            m = np.memmap(mp, dtype=np.int64, mode="r") if mp.exists() and mp.stat().st_size else np.zeros(0, dtype=np.int64)
            self._miss_mm[di] = m
        return m

    def _missing_ram(self, di: int):
        m = self._miss_ram.get(di)
        if m is None:
            mp = self.root / self.names[di] / "missing.i64"
            m = np.fromfile(mp, dtype=np.int64) if mp.exists() and mp.stat().st_size else np.zeros(0, dtype=np.int64)
            ends = np.append(self.offs[di], self.offs[di][-1] + self.lens[di][-1]) if len(self.offs[di]) else np.zeros(1, dtype=np.int64)
            self._miss_ram[di] = m
            self._miss_ptr[di] = np.searchsorted(m, ends, side="left")             # series s owns m[ptr[s]:ptr[s + 1]]
        return m

    def missing_in(self, di: int, lo: int, hi: int) -> np.ndarray:
        """Positions (relative to lo) of originally missing points among the stored positions [lo, hi) of dataset di."""
        m = self._missing_mm(di) if self.io == "mmap" else self._missing_ram(di)
        a, b = int(np.searchsorted(m, lo, side="left")), int(np.searchsorted(m, hi, side="left"))
        return np.asarray(m[a:b]) - lo

    def window_missing(self, di: int, ser: int, start: int, c: int) -> np.ndarray:
        """The same positions for the window (series `ser`, from `start`, c points), relative to the window."""
        base = int(self.offs[di][ser]) + start
        if self.io == "mmap":
            return self.missing_in(di, base, base + c)
        m = self._missing_ram(di)
        ptr = self._miss_ptr[di]
        lo_i, hi_i = int(ptr[ser]), int(ptr[ser + 1])
        if lo_i == hi_i:
            return np.zeros(0, dtype=np.int64)
        seg = m[lo_i:hi_i]
        a, b = int(np.searchsorted(seg, base, side="left")), int(np.searchsorted(seg, base + c, side="left"))
        return seg[a:b] - base

    def stored_points(self) -> dict[str, int]:
        return {n: int(l.sum()) for n, l in zip(self.names, self.lens)}

    def n_eligible(self, di: int, c: int) -> int:
        """Series of dataset di with at least c stored points."""
        return len(self.lens[di]) - int(np.searchsorted(self.sorted_len[di], c, side="left"))


class _MissView:
    """store.miss[di] -> the sorted positions of the originally missing points of dataset di."""

    def __init__(self, store: RealStore):
        self.s = store

    def __getitem__(self, di: int):
        return self.s._missing_mm(di) if self.s.io == "mmap" else self.s._missing_ram(di)


class _ValsView:
    """store.vals[di][a:b] -> the stored points as an array (memmap slice): kept for tests and tools that index the values directly."""

    def __init__(self, store: RealStore):
        self.s = store

    def __getitem__(self, di: int):
        return self.s._memmap(di)

    def __len__(self):
        return len(self.s.names)
