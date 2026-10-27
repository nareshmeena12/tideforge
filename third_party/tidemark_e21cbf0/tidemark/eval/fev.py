"""Helpers around the scoring of the twins (the scoring itself is tidemark.eval.score_twins, wired to
the owner's reference/time_fev_eval.py): finding and loading twins, and the long score table that
tidemark.eval.gaps reads.

Long table columns: model, arm, seed, benchmark ("fev" | "time"), metric ("test_error" for fev-bench,
the error its skill score uses; "MASE" / "CRPS" for TIME), task, window ("all": fev-bench summaries
are already averaged over a task's windows), err. Older tables without benchmark / metric work too.
"""
from __future__ import annotations

import csv
from pathlib import Path

SCORE_COLUMNS = ("model", "arm", "seed", "benchmark", "metric", "task", "window", "err")


def twins(folder) -> list[Path]:
    """final.pt of every finished twin run under `folder` (runs/<arm>_s<seed>_lr<lr>_<id>/)."""
    return sorted(Path(folder).glob("*/final.pt"))


def load(path, device, window: int | None = None, gla_backend: str = "fla"):
    from tidemark.pretrain.inference import load_twin
    from tidemark.pretrain.model import restrict_window
    model, cfg, ck = load_twin(path, device, gla_backend)
    if window is not None:
        restrict_window(model, window)
    name = f"{cfg.arm}_s{cfg.seed}" + (f"_local{window}" if window else "")
    return model, cfg, name


def write_scores(rows: list[dict], out) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SCORE_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in SCORE_COLUMNS})


def read_scores(path, benchmark: str | None = None, metric: str | None = None) -> list[dict]:
    """The rows of a long table, optionally only one benchmark / metric."""
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if benchmark is not None:
        rows = [r for r in rows if r.get("benchmark", benchmark) == benchmark]
    if metric is not None:
        rows = [r for r in rows if r.get("metric", metric) == metric]
    return rows
