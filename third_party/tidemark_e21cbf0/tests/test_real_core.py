"""ft3_core.py is a mechanical copy of the reference: nothing is edited, nothing runs on import."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
CORE = ROOT / "tidemark" / "real" / "ft3_core.py"
REF = ROOT / "reference" / "tide_v5_ft3.py"


def test_core_has_only_definitions_and_constants():
    tree = ast.parse(CORE.read_text(encoding="utf-8"))
    ok = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign, ast.AugAssign)
    bad = [n.lineno for n in tree.body if not isinstance(n, ok) and not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert bad == []
    assert "RUN_TRAINING   = True" in CORE.read_text(encoding="utf-8")               # the flags are constants, nothing calls train()


@pytest.mark.skipif(not REF.exists(), reason="reference/ is git-ignored and absent on Perlmutter")
def test_core_equals_the_reference_slice():
    import importlib.util
    spec = importlib.util.spec_from_file_location("extract_ft3", ROOT / "scripts" / "real" / "extract_ft3.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert CORE.read_text(encoding="utf-8") == mod.build(REF)


def test_core_imports_and_exposes_the_pieces_the_track_needs():
    from tidemark.real import ft3_core as C
    for name in ("build_model", "PackedBatcher", "packed_forward", "_multivariatize", "build_val_batches", "evaluate_packed",
                 "forecast_left_padded", "forecast_groups", "load_packed_checkpoint", "set_group_options", "CONFIG"):
        assert hasattr(C, name), name
