"""Import the pinned tidemark copy (third_party/tidemark_e21cbf0) without installing it."""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
THIRD_PARTY = REPO / "third_party" / "tidemark_e21cbf0"


def add_path() -> None:
    if str(THIRD_PARTY) not in sys.path:
        sys.path.insert(0, str(THIRD_PARTY))


def ft3_core():
    add_path()
    import tidemark.real.ft3_core as C
    return C


def score_groups():
    add_path()
    import tidemark.real.score_groups as G
    return G


def score_twins():
    add_path()
    import tidemark.eval.score_twins as S
    return S
