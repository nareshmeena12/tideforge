"""Write tideforge/gen/{_state,primitives,families,profiles,layers,batch,groups}.py from ft3_core.py (generator v0).

    python tools/port_gen_v0.py            # (re)write the modules
    python tools/port_gen_v0.py --check    # exit 1 if the modules on disk differ from what this script would write

Every ported line is copied verbatim from third_party/tidemark_e21cbf0/tidemark/real/ft3_core.py. The only edits:
  * imports at the top of each module (math, torch, F, and the names defined in the earlier modules);
  * the four profile globals that generate_batch sets and other sections read (_PERIOD_BANK_ACTIVE, _PERIOD_EXT_SHARE,
    _TV_SEASON, _ONE_OFF_P) live in tideforge/gen/_state.py: on the lines listed in STATE_LINES a read or write of NAME
    becomes _state.NAME, and the `global` statement of generate_batch (L2352) becomes a comment;
  * ROLE_TARGET, ROLE_PAST, ROLE_KNOWN (L503, the model section) are copied into groups.py.
The group options (GROUP_KIND_PROBS, _COV_*, _SECOND_GROUP_P) are read and written only inside the groups section, so they
stay plain module globals of groups.py, as in the source. tests/test_gen_source.py re-derives the source from the modules.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "third_party" / "tidemark_e21cbf0" / "tidemark" / "real" / "ft3_core.py"
SRC_REL = "third_party/tidemark_e21cbf0/tidemark/real/ft3_core.py"
OUT = REPO / "tideforge" / "gen"

STATE_NAMES = ("_PERIOD_BANK_ACTIVE", "_PERIOD_EXT_SHARE", "_TV_SEASON", "_ONE_OFF_P")
STATE_LINES = (821, 824, 831, 1215, 1783, 1788, 2062, 2355, 2356, 2357, 2358)
GLOBAL_LINE = 2352
GLOBAL_COMMENT = "# [port] `global` statement removed: these four names live in tideforge/gen/_state.py"
BEGIN = "# >>> ft3_core.py L{a}-L{b}"
END = "# <<< ft3_core.py"

# module -> (title, [(first line, last line), ...]) in import order; ranges are 1-based and inclusive
MODULES = {
    "primitives": ("Section 3 header, period banks and the low-level primitives.", [(766, 784), (790, 921)]),
    "_state": ("Profile state that generate_batch sets and the families / layers read (module globals in the source).",
               [(785, 789)]),
    "families": ("Base dynamical families, clean-corpus families and the clean2 additions (section 3c), incl. _L_breaks "
                 "and _one_off.", [(922, 1923)]),
    "profiles": ("FAMILIES, weights, GEN_PROFILES, clean2 profile builder, resolve_profile.", [(1924, 2042)]),
    "layers": ("Stochastic layers and _finalize.", [(2043, 2318)]),
    "batch": ("_allocate and generate_batch (top-level API).", [(2319, 2398)]),
    "groups": ("Section 3d: group kinds, set_group_options, multivariatizers, generate_blocks.", [(503, 503), (2399, 2677)]),
}


def ported_lines(src: list[str], a: int, b: int) -> list[str]:
    out = []
    for i in range(a, b + 1):
        ln = src[i - 1]
        if i == GLOBAL_LINE:
            assert ln.strip().startswith("global "), ln
            ln = ln[: len(ln) - len(ln.lstrip())] + GLOBAL_COMMENT
        elif i in STATE_LINES:
            new = re.sub(r"(?<![\w.])(" + "|".join(STATE_NAMES) + r")\b", r"_state.\1", ln)
            assert new != ln, f"L{i}: no state name to rewrite: {ln!r}"
            ln = new
        out.append(ln)
    return out


def top_names(code: str) -> set[str]:
    names = set()
    for n in ast.parse(code).body:
        if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            names.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                names |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
    return names


def loaded_names(code: str) -> set[str]:
    return {x.id for x in ast.walk(ast.parse(code)) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}


def build() -> dict[str, str]:
    src = SRC.read_text(encoding="utf-8").splitlines()
    bodies = {m: [(a, b, ported_lines(src, a, b)) for a, b in rs] for m, (_, rs) in MODULES.items()}
    code = {m: "\n".join(l for _, _, ls in blocks for l in ls) + "\n" for m, blocks in bodies.items()}
    tops = {m: top_names(c) for m, c in code.items()}
    files, earlier = {}, []
    for m, (title, ranges) in MODULES.items():
        need = loaded_names(code[m]) - tops[m]
        imports = ["import math", "", "import torch", "import torch.nn.functional as F", ""]
        uses_state = m != "_state" and any(f"_state.{n}" in code[m] for n in STATE_NAMES)
        if uses_state and m != "primitives":
            imports.append("from tideforge.gen import _state")
        for e in earlier:
            if e == "_state":
                continue
            got = sorted(need & tops[e])
            if got:
                imports.append(f"from tideforge.gen.{e} import (  # noqa: F401")
                imports += [f"    {n}," for n in got]
                imports.append(")")
            need -= tops[e]
        for later in list(MODULES)[list(MODULES).index(m) + 1:]:
            assert not (need & tops[later]), f"{m} uses {sorted(need & tops[later])} from the later module {later}"
        rng = ", ".join(f"L{a}-L{b}" for a, b in ranges)
        doc = (f'"""Generator v0: {title}\n\nPorted from {SRC_REL} {rng} (tidemark commit e21cbf0) by tools/port_gen_v0.py; do not edit by hand\n'
               f"(edit and rerun the tool). Edits vs the source: imports only"
               + ("; the profile globals read / written as _state.NAME" if any(a <= l <= b for a, b in ranges for l in STATE_LINES) else "")
               + ("; generate_batch's `global` line is a comment" if any(a <= GLOBAL_LINE <= b for a, b in ranges) else "")
               + ".\nMust stay bit-exact with ft3_core (tests/test_gen_parity.py, tests/test_gen_source.py).\n\"\"\"")
        parts = [doc, "# flake8: noqa", *imports, ""]
        for a, b, ls in bodies[m]:
            parts += ["", BEGIN.format(a=a, b=b), *ls, END]
        if m == "primitives":   # _state reads PERIOD_BANK from here, so it is imported last (looked up at call time)
            parts += ["", "", "from tideforge.gen import _state  # noqa: E402"]
        files[m] = "\n".join(parts) + "\n"
        earlier.append(m)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    files = build()
    bad = []
    for m, text in files.items():
        p = OUT / f"{m}.py"
        if a.check:
            if not p.exists() or p.read_text(encoding="utf-8") != text:
                bad.append(str(p))
        else:
            p.write_bytes(text.encode("utf-8"))
            print(f"wrote {p.relative_to(REPO)} ({text.count(chr(10))} lines)")
    if bad:
        print("differ from the tool's output:", *bad, sep="\n  ")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
