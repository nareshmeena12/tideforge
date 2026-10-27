"""E0.1 (source side): every line between the port markers in tideforge/gen/*.py is the ft3_core.py line it names, apart from the
documented edits (`_state.` in front of the four profile globals, generate_batch's `global` line as a comment, and the hook lines
marked `# [hook Hn]`: inserted lines, and wrapped calls that unwrap to the source line); together the blocks cover ft3_core L503
and L766-L2677 exactly once; and tools/port_gen_v0.py would write the same files."""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "third_party" / "tidemark_e21cbf0" / "tidemark" / "real" / "ft3_core.py"
GEN = REPO / "tideforge" / "gen"
BEGIN = re.compile(r"^# >>> ft3_core\.py L(\d+)-L(\d+)$")
END = "# <<< ft3_core.py"
HOOK = re.compile(r"  # \[hook (H\d|tick)\]$")
LAYER_WRAP = re.compile(r'_hooks\.layer\("\w+", (\w+)\)\(')
FINAL_WRAP = re.compile(r"return _hooks\.final_blocks\((.*), device\)$")


def unhook(line):
    """(kind, line): 'insert' for an inserted hook line, 'wrap' with the unwrapped line, or None with the line itself."""
    m = HOOK.search(line)
    if not m:
        return None, line
    body = line[:m.start()]
    if "_hooks.layer(" in body:
        return "wrap", LAYER_WRAP.sub(r"\1(", body)
    if "_hooks.final_blocks(" in body:
        return "wrap", FINAL_WRAP.sub(r"return \1", body)
    return "insert", None


def blocks():
    for p in sorted(GEN.glob("*.py")):
        lines = p.read_text(encoding="utf-8").splitlines()
        i = 0
        while i < len(lines):
            m = BEGIN.match(lines[i])
            if m:
                j = lines.index(END, i)
                yield p.name, int(m.group(1)), int(m.group(2)), lines[i + 1:j]
                i = j
            i += 1


def test_ported_lines_equal_the_source():
    src = SRC.read_text(encoding="utf-8").splitlines()
    covered, state_edits, global_edits, inserts, wraps = [], 0, 0, 0, 0
    for name, a, b, body in blocks():
        kept = []
        for line in body:
            kind, line = unhook(line)
            inserts += kind == "insert"
            wraps += kind == "wrap"
            if kind != "insert":
                kept.append(line)
        assert len(kept) == b - a + 1, f"{name} L{a}-L{b}: {len(kept)} lines"
        for k, line in enumerate(kept):
            n = a + k
            orig = src[n - 1]
            covered.append(n)
            if line == orig:
                continue
            if "_state." in line and line.replace("_state.", "") == orig:
                state_edits += 1
            elif orig.strip().startswith("global ") and line.strip().startswith("# [port] `global` statement removed"):
                global_edits += 1
            else:
                raise AssertionError(f"{name}: L{n} differs from ft3_core:\n  port: {line!r}\n  src:  {orig!r}")
    assert sorted(covered) == [503] + list(range(766, 2678)), "blocks must cover L503 and L766-L2677 exactly once"
    assert (state_edits, global_edits) == (11, 1), (state_edits, global_edits)
    assert (inserts, wraps) == (9, 10), (inserts, wraps)          # H2-H7 + 2 ticks (H5 is 2 lines); H1 x 9 layers + H8


def test_hook_lines_only_call_hooks():
    for p in sorted(GEN.glob("*.py")):
        for line in p.read_text(encoding="utf-8").splitlines():
            kind, _ = unhook(line)
            if kind == "insert":
                assert "_hooks." in line and "torch." not in line, line


def test_port_tool_check():
    r = subprocess.run([sys.executable, str(REPO / "tools" / "port_gen_v0.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
