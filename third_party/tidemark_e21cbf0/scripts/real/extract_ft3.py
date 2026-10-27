"""Copy the importable part of reference/tide_v5_ft3.py (model, generators, group builder, PackedBatcher, inference helpers, probes,
holdout and the reference training loop) into tidemark/real/ft3_core.py, byte for byte.

  python scripts/real/extract_ft3.py [--reference reference/tide_v5_ft3.py] [--out tidemark/real/ft3_core.py] [--check]

The reference file starts a training run when it is imported, so it is never imported. Everything before its "RUN" section
(line 4089 in the copy this was written against) is definitions and constants; the RUN section (config of the three fine-tunes, the
self-test and the calls that train) is dropped. The copy is a mechanical slice: no line is edited, so a test can compare it with the
reference whenever reference/ is present (reference/ is git-ignored and absent on Perlmutter; ft3_core.py is committed).
--check only verifies that the committed copy equals the current slice and that the slice has no statement that does work on import.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

RUN_MARKER = "# ============================ RUN ==================================="
FIRST_LINE = 69          # "import math, os, ..." : the lines before it are the Kaggle instructions
HEADER = '''"""Verbatim copy of reference/tide_v5_ft3.py lines {first}-{last} (Fluxion-Tide v5, round 3): model, synthetic generators, group
builder, PackedBatcher, inference helpers, probes, holdout and the reference single-GPU training loop. Produced by
scripts/real/extract_ft3.py; DO NOT EDIT BY HAND (a test compares it with the reference). Nothing here runs on import except the
definitions and the constants of the reference, and `_T0 = time.time()`. The RUN section of the reference (it trains) is not copied.
"""
'''


def slice_reference(ref: Path) -> tuple[str, int, int]:
    lines = ref.read_text(encoding="utf-8").splitlines()
    try:
        end = next(i for i, ln in enumerate(lines) if ln.startswith(RUN_MARKER))
    except StopIteration:
        raise SystemExit(f"{ref}: RUN marker not found; the reference changed, review before extracting")
    body = lines[FIRST_LINE - 1:end]
    while body and not body[-1].strip():
        body.pop()
    return "\n".join(body) + "\n", FIRST_LINE, FIRST_LINE + len(body) - 1


def import_time_statements(src: str) -> list[str]:
    """Top-level statements that are not an import, a def/class, an assignment or a docstring."""
    bad = []
    for node in ast.parse(src).body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign, ast.AugAssign)):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        bad.append(f"line {node.lineno}: {type(node).__name__}")
    return bad


def build(ref: Path) -> str:
    body, first, last = slice_reference(ref)
    bad = import_time_statements(body)
    if bad:
        raise SystemExit("the slice has statements that run on import: " + "; ".join(bad))
    return HEADER.format(first=first, last=last) + body


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reference", default="reference/tide_v5_ft3.py")
    ap.add_argument("--out", default="tidemark/real/ft3_core.py")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    text = build(Path(a.reference))
    out = Path(a.out)
    if a.check:
        same = out.exists() and out.read_text(encoding="utf-8") == text
        print("ft3_core.py is identical to the slice" if same else "ft3_core.py DIFFERS from the slice (or is missing)")
        return 0 if same else 1
    out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {out}: {len(text.splitlines())} lines")
    return 0


if __name__ == "__main__":
    sys.exit(main())
