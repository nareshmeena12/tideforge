"""Names of Stage B reruns at another start rate (no torch: used by the launcher, the eval views and the tables).

A rerun of softmax at 2.4e-4, seed 0, size 100m lives in <root>/runs_100m_rerun/softmax_s0_lr0.00024_<id>/ and is
evaluated under the tag softmax_lr24_s0 (the rate in units of 1e-5), so it never collides with the original softmax_s0.
"""
from __future__ import annotations

import re

RUN_NAME = re.compile(r"^(?P<arm>[a-z0-9]+)_s(?P<seed>\d+)_lr(?P<lr>[0-9.e+-]+?)_(?P<id>[0-9a-f]{6,})$")


def rerun_tag(arm: str, seed: int, lr: float) -> str:
    """softmax, 0, 0.00024 -> softmax_lr24_s0"""
    return f"{arm}_lr{int(round(lr * 1e5))}_s{seed}"


def split_run_name(name: str) -> dict | None:
    """'softmax_s0_lr0.00024_ab12cd34' -> {arm, seed, lr, id}; None if it is not a run folder name."""
    m = RUN_NAME.match(name)
    return {"arm": m["arm"], "seed": int(m["seed"]), "lr": float(m["lr"]), "id": m["id"]} if m else None


def view_name(name: str) -> str:
    """Folder name of a rerun inside the evaluation view: the tag's arm part carries the rate (softmax_lr24_s0_lr0.00024_<id>)."""
    p = split_run_name(name)
    if p is None:
        raise ValueError(f"not a run folder name: {name!r}")
    return f"{rerun_tag(p['arm'], p['seed'], p['lr'])}_lr{name.split('_lr', 1)[1]}"
