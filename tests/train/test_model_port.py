"""tideforge/model/ft3_model.py is exactly what tools/port_model.py writes from the pinned third_party sources."""
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_model_port_matches_source():
    r = subprocess.run([sys.executable, str(REPO / "tools" / "port_model.py"), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
