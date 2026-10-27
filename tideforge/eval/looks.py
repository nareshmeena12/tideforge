"""LOOKS.md: one row per look at fev-bench or TIME, newest first (CLAUDE.md rule 5)."""
from __future__ import annotations

import datetime as _dt
from pathlib import Path

from tideforge.eval.thirdparty import REPO

LOOKS = REPO / "LOOKS.md"
HEADER = "| Date | Experiment | Checkpoint | Benchmark | Why this look | Result |"
PLACEHOLDER = "| | | | | | |"


def _cell(s) -> str:
    return str(s).replace("|", "/").replace("\n", " ").strip()


def format_row(*, experiment: str, checkpoint: str, benchmark: str, why: str, result: str, date: str | None = None) -> str:
    date = date or _dt.date.today().isoformat()
    return "| " + " | ".join(_cell(x) for x in (date, experiment, checkpoint, benchmark, why, result)) + " |"


def append_looks(rows: list[str], path: Path | str = LOOKS) -> list[str]:
    """Insert rows right under the table header (newest first) and drop the empty placeholder row. A row whose Experiment,
    Checkpoint and Benchmark cells equal an existing row's is skipped (re-running a table does not count a second look).
    Returns the rows actually added."""
    p = Path(path)
    lines = p.read_text(encoding="utf-8").splitlines()
    h = lines.index(HEADER)
    sep = h + 1
    body = [ln for ln in lines[sep + 1:] if ln.startswith("|")]
    key = lambda ln: tuple(c.strip() for c in ln.strip("|").split("|"))[1:4]   # noqa: E731
    have = {key(ln) for ln in body}
    new = [r for r in rows if key(r) not in have]
    rest = [ln for ln in lines[sep + 1:] if ln.strip() != PLACEHOLDER]
    p.write_bytes(("\n".join(lines[:sep + 1] + new + rest) + "\n").encode("utf-8"))
    return new
