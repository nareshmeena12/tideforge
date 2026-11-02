"""LOOKS rows: one per look at fev-bench or TIME (CLAUDE.md rule 5). Code writes only to $TF_ROOT/looks_pending.md, never to a
file in the repo, so a job cannot dirty the checkout. The owner moves the rows into the tracked LOOKS.md and commits them."""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

from tideforge.eval.thirdparty import REPO

LOOKS = REPO / "LOOKS.md"
PENDING = "looks_pending.md"
HEADER = "| Date | Experiment | Checkpoint | Benchmark | Why this look | Result |"
PLACEHOLDER = "| | | | | | |"


def _cell(s) -> str:
    return str(s).replace("|", "/").replace("\n", " ").strip()


def format_row(*, experiment: str, checkpoint: str, benchmark: str, why: str, result: str, date: str | None = None) -> str:
    date = date or _dt.date.today().isoformat()
    return "| " + " | ".join(_cell(x) for x in (date, experiment, checkpoint, benchmark, why, result)) + " |"


def _in_repo(p: Path) -> bool:
    return p.resolve().is_relative_to(REPO.resolve())


def target(path: Path | str | None = None) -> Path:
    """Where rows go: path, or $TF_ROOT/looks_pending.md when path is None or inside the repo."""
    if path is not None and not _in_repo(Path(path)):
        return Path(path)
    root = os.environ.get("TF_ROOT")
    if not root:
        raise SystemExit("LOOKS rows need TF_ROOT or a --looks-file outside the repo")
    p = Path(root) / PENDING
    if _in_repo(p):
        raise SystemExit(f"TF_ROOT is inside the repo: {p}")
    return p


def append_looks(rows: list[str], path: Path | str | None = None) -> list[str]:
    """Add rows to target(path). A file with the table header gets them right under it (newest first, placeholder dropped);
    any other file, or none, gets them appended. A row whose Experiment, Checkpoint and Benchmark cells equal an existing
    row's is skipped (re-running a table does not count a second look). Returns the rows actually added."""
    p = target(path)
    if path is not None and p != Path(path):
        print(f"LOOKS rows redirected from {path} to {p}: code never writes in the repo")
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    if HEADER not in lines:
        key = lambda ln: tuple(c.strip() for c in ln.strip("|").split("|"))[1:4]   # noqa: E731
        have = {key(ln) for ln in lines if ln.startswith("|")}
        new = [r for r in rows if key(r) not in have]
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8", newline="\n") as f:
            f.writelines(r + "\n" for r in new)
        return new
    h = lines.index(HEADER)
    sep = h + 1
    body = [ln for ln in lines[sep + 1:] if ln.startswith("|")]
    key = lambda ln: tuple(c.strip() for c in ln.strip("|").split("|"))[1:4]   # noqa: E731
    have = {key(ln) for ln in body}
    new = [r for r in rows if key(r) not in have]
    rest = [ln for ln in lines[sep + 1:] if ln.strip() != PLACEHOLDER]
    p.write_bytes(("\n".join(lines[:sep + 1] + new + rest) + "\n").encode("utf-8"))
    return new
