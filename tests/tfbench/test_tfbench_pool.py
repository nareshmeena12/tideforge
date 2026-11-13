"""pool.run: a dead worker, a timeout or an exception fails that job only; the others finish."""
from __future__ import annotations

import os
import time

from tideforge.tfbench import pool


def job(kind: str):
    if kind == "die":
        os._exit(137)                    # like an OOM kill: no result, no exception
    if kind == "slow":
        time.sleep(30)
    if kind == "raise":
        raise ValueError("bad file")
    return {"kind": kind}


def test_pool_survives_dead_worker_timeout_and_errors():
    started = []
    jobs = [(k, (k,)) for k in ("a", "die", "b", "slow", "raise", "c")]
    t0 = time.monotonic()
    res = {k: (ok, v) for k, ok, v in pool.run(job, jobs, 3, timeout_s=3, on_start=started.append, poll_s=0.2)}
    assert time.monotonic() - t0 < 25
    assert sorted(started) == sorted(k for k, _ in jobs)
    assert res["a"] == (True, {"kind": "a"}) and res["c"] == (True, {"kind": "c"})
    assert not res["die"][0] and "worker died" in res["die"][1]
    assert not res["slow"][0] and "timeout" in res["slow"][1]
    assert not res["raise"][0] and "ValueError: bad file" in res["raise"][1]
