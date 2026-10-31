"""Probe tests build small sets in tmp folders: they run against an empty registry, so registry.json (Perlmutter's 256-per-cell
build) never applies to them. test_common.py's registry test reads the committed file through REAL_REGISTRY."""
import pytest

from tideforge.probes import common

REAL_REGISTRY = common.REGISTRY


@pytest.fixture(autouse=True)
def _isolated_registry(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "REGISTRY", tmp_path / "registry_empty.json")
