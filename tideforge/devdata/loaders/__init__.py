"""Loaders: raw files -> long DataFrame (series_id, timestamp [UTC, naive], value, + the strata columns of the source).

    read_files(src, files)   files = [(path, vars)] where vars are the url-template variables of that file
    read_index(src, path)    the series list of a source with an index (series id + strata columns)

`table` (tabular.py) is driven by the sources.yaml entry; `adapter:<name>` calls tideforge/devdata/loaders/adapters/<name>.py,
which defines read_files(src, files) (and optionally read_index(src, path)).
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd

from tideforge.devdata.loaders import tabular

LONG = ["series_id", "timestamp", "value"]


def _adapter(src):
    return importlib.import_module(f"tideforge.devdata.loaders.adapters.{src.loader.split(':', 1)[1]}")


def read_files(src, files: list[tuple[Path, dict]]) -> pd.DataFrame:
    if src.loader == "table":
        df = tabular.read_files(src, files)
    else:
        df = _adapter(src).read_files(src, files)
    missing = [c for c in LONG if c not in df.columns]
    if missing:
        raise ValueError(f"{src.id}: loader output lacks {missing}")
    return df


def read_index(src, paths: list[tuple[Path, dict]]) -> pd.DataFrame:
    if src.loader != "table" and hasattr(_adapter(src), "read_index"):
        return _adapter(src).read_index(src, paths)
    return tabular.read_index(src, paths)
