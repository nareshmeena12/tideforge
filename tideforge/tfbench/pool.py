"""Run one job per process, at most n at a time, and survive a dead worker (owner 2026-11-13: a worker OOM-killed under
ProcessPoolExecutor left the full canon hanging for 3 h).

Each job gets its own process, so a crash, an OOM kill, a timeout or the memory cap fails that job only:
- worker died (exit code / signal, likely out of memory) -> failure "worker died (likely out of memory), exit <code>"
- over `timeout_s` -> killed, failure "timeout after <m> min"
- resident memory over `mem_bytes` (polled from /proc on Linux) -> killed, failure "memory cap <g> GB exceeded"
`on_start(key)` is called when a job starts (the log shows which jobs were in flight at a hang or crash).
"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any, Callable, Iterator

PAGE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def _child(conn, fn, args):
    try:
        res = ("ok", fn(*args))
    except BaseException as e:  # noqa: BLE001  any error comes back as a failure, never kills the run
        res = ("error", f"{type(e).__name__}: {str(e)[:300]}\n{traceback.format_exc()[-1500:]}")
    try:
        conn.send(res)
    finally:
        conn.close()


def rss_bytes(pid: int) -> int | None:
    """Resident memory of a process (Linux /proc), None elsewhere."""
    try:
        return int(Path(f"/proc/{pid}/statm").read_text().split()[1]) * PAGE
    except (OSError, ValueError, IndexError):
        return None


def default_mem_bytes(n: int) -> int | None:
    """90% of the cgroup (container) memory limit, split over n workers; None when there is no limit to read."""
    for f in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(f).read_text().strip()
        except OSError:
            continue
        if v.isdigit() and int(v) < 1 << 60:
            return int(int(v) * 0.9 / max(n, 1))
    return None


def run(fn: Callable, jobs: list[tuple[str, tuple]], n: int, timeout_s: float | None = None,
        mem_bytes: int | None = None, on_start: Callable[[str], Any] | None = None,
        poll_s: float = 1.0) -> Iterator[tuple[str, bool, Any]]:
    """Yield (key, ok, result) as jobs finish; result is fn's return value, or the failure text when ok is False."""
    ctx = mp.get_context("fork" if hasattr(os, "fork") else "spawn")
    todo = list(jobs)[::-1]
    live: dict = {}                                   # conn -> (key, process, start time)
    while todo or live:
        while todo and len(live) < max(n, 1):
            key, args = todo.pop()
            r, w = ctx.Pipe(duplex=False)
            p = ctx.Process(target=_child, args=(w, fn, args), daemon=False)
            p.start()
            w.close()
            live[r] = (key, p, time.monotonic())
            if on_start:
                on_start(key)
        ready = wait(list(live) + [p.sentinel for _, p, _ in live.values()], timeout=poll_s)
        now = time.monotonic()
        for conn in list(live):
            key, p, t0 = live[conn]
            out = None
            if conn in ready or p.sentinel in ready:
                try:
                    if conn.poll():
                        st, val = conn.recv()
                        out = (key, st == "ok", val)
                except (EOFError, OSError):
                    pass
                if out is None:
                    p.join(5)
                    if p.exitcode is None and not conn.poll():
                        continue                          # pipe closed early but still running: check again later
                    if p.exitcode is not None:
                        out = (key, False, f"worker died (likely out of memory), exit {p.exitcode}")
            if out is None and timeout_s and now - t0 > timeout_s:
                out = (key, False, f"timeout after {timeout_s / 60:.0f} min")
            if out is None and mem_bytes:
                rss = rss_bytes(p.pid)
                if rss is not None and rss > mem_bytes:
                    out = (key, False, f"memory cap {mem_bytes / 1e9:.1f} GB exceeded (rss {rss / 1e9:.1f} GB)")
            if out is None:
                continue
            if p.is_alive():
                p.kill()
            p.join(10)
            conn.close()
            del live[conn]
            yield out
