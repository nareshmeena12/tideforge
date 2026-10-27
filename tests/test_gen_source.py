"""E0.1 (source side): every line between the port markers in tideforge/gen/*.py is the ft3_core.py line it names, apart from the
documented edits (`_state.` in front of the four profile globals, and generate_batch's `global` line as a comment); together the
blocks cover ft3_core L503 and L766-L2677 exactly once; and tools/port_gen_v0.py would write the same files."""
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "third_party" / "tidemark_e21cbf0" / "tidemark" / "real" / "ft3_core.py"
GEN = REPO / "tideforge" / "gen"
BEGIN = re.compile(r"^# >>> ft3_core\.py L(\d+)-L(\d+)$")
END = "# <<< ft3_core.py"


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
    covered, state_edits, global_edits = [], 0, 0
    for name, a, b, body in blocks():
        assert len(body) == b - a + 1, f"{name} L{a}-L{b}: {len(body)} lines"
        for k, line in enumerate(body):
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


def test_port_tool_check():
    r = subprocess.run([sys.executable, str(REPO / "tools" / "port_gen_v0.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
