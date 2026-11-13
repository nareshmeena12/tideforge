"""The real config files load and every overrides.yaml entry is valid (owner 2026-11-13: a bad escape in a double-quoted
regex crashed the full build). Regex values are single-quoted in overrides.yaml (backslashes literal)."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd
import pytest
import yaml

from tideforge.tfbench import rules

ROOT = Path(__file__).resolve().parents[2]
OVERRIDES = ROOT / "tideforge" / "devdata" / "tfbench" / "overrides.yaml"
SELECTION = ROOT / "tideforge" / "devdata" / "tfbench" / "selection.csv"

REGEX = {"files", "table", "drop_parts", "drop_items", "id_from_path", "targets_match"}
STR = {"skip", "note", "time_col", "start_col", "strata", "freq", "event_step", "event_sum", "audit_only", "layout"}
LISTS = {"id_cols", "targets", "covariates", "known", "static", "table_cols"}
BOOLS = {"fill_zero", "synthetic_time", "sum_rows"}
OTHER = {"calendar", "agg", "diff", "ref_ids", "n_targets_match"}
KEYS = REGEX | STR | LISTS | BOOLS | OTHER
CAL_KEYS = {"year", "month", "week", "quarter", "day", "hour", "yyyymm", "yearweek", "fiscal_year", "week_rule"}


def yaml_files() -> list[Path]:
    try:
        out = subprocess.run(["git", "ls-files", "*.yaml", "*.yml"], cwd=ROOT, capture_output=True, text=True, check=True)
        return [ROOT / p for p in out.stdout.split()]
    except (OSError, subprocess.CalledProcessError):
        return sorted(p for p in ROOT.rglob("*.y*ml") if ".git" not in p.parts and "third_party" not in p.parts
                      and "reference" not in p.parts and "data" not in p.parts)


@pytest.mark.parametrize("path", yaml_files(), ids=lambda p: p.relative_to(ROOT).as_posix())
def test_every_tracked_yaml_parses(path):
    yaml.safe_load(path.read_text(encoding="utf-8"))


def test_regex_values_single_quoted():
    bad = [f"{i}: {ln.strip()}" for i, ln in enumerate(OVERRIDES.read_text(encoding="utf-8").splitlines(), 1)
           if re.match(r"\s*(" + "|".join(REGEX | {"path"}) + r"):\s*\"", ln)]
    assert not bad, "use single quotes for regex values (backslashes are escapes in double quotes):\n" + "\n".join(bad)


def test_overrides_entries_valid():
    ov = yaml.safe_load(OVERRIDES.read_text(encoding="utf-8"))
    ids = set(pd.read_csv(SELECTION)["dataset_id"])
    errs = []
    for did, e in ov.items():
        if did not in ids:
            errs.append(f"{did}: not a dataset_id in selection.csv")
        if not isinstance(e, dict):
            errs.append(f"{did}: entry is not a mapping")
            continue
        for k, v in e.items():
            where = f"{did}.{k}"
            if k not in KEYS:
                errs.append(f"{where}: unknown key")
            elif k in REGEX:
                if not isinstance(v, str):
                    errs.append(f"{where}: regex must be a string")
                    continue
                try:
                    rx = re.compile(v)
                except re.error as ex:
                    errs.append(f"{where}: bad regex {v!r}: {ex}")
                    continue
                if k == "id_from_path" and rx.groups < 1:
                    errs.append(f"{where}: needs a capture group (the id)")
            elif k in STR and not isinstance(v, str):
                errs.append(f"{where}: must be a string")
            elif k in LISTS and not (isinstance(v, list) and all(isinstance(x, str) for x in v)):
                errs.append(f"{where}: must be a list of column names")
            elif k in BOOLS and not isinstance(v, bool):
                errs.append(f"{where}: must be true / false")
            elif k == "calendar" and not (v == "auto" or v is True or (isinstance(v, dict) and set(v) <= CAL_KEYS
                                                                       and v.get("week_rule", "mmwr") in ("mmwr", "iso"))):
                errs.append(f"{where}: {v!r} (keys {sorted(CAL_KEYS)}, week_rule mmwr | iso)")
            elif k == "diff" and not (v is False or (isinstance(v, list) and all(isinstance(x, str) for x in v))):
                errs.append(f"{where}: diff is a list of columns or false")
            elif k == "agg" and not (isinstance(v, dict) and set(v.values()) <= {"sum", "mean", "max", "min", "last", "first"}):
                errs.append(f"{where}: agg is {{column: sum | mean | ...}}")
            elif k == "ref_ids" and not (isinstance(v, dict) and isinstance(v.get("path"), str)):
                errs.append(f"{where}: ref_ids needs path (regex)")
            elif k == "ref_ids":
                try:
                    re.compile(v["path"])
                except re.error as ex:
                    errs.append(f"{where}.path: bad regex: {ex}")
        if e.get("layout") not in (None, "long", "wide", "nested", "events"):
            errs.append(f"{did}.layout: {e['layout']!r}")
        for k in ("freq", "event_step"):
            if isinstance(e.get(k), str):
                try:
                    rules.freq_seconds(e[k])
                except Exception as ex:
                    errs.append(f"{did}.{k}: {e[k]!r} is not a pandas frequency ({ex})")
        if e.get("skip") is not None and not str(e["skip"]).strip():
            errs.append(f"{did}.skip: needs a reason")
    assert not errs, "\n".join(errs)


def test_shard_and_split_ids():
    from tideforge.tfbench.run import SHARD_ID
    for s in ("data/train-00001-of-00004.parquet/x", "ground/test.csv", "a/val/b", "test_x.parquet"):
        assert SHARD_ID.search(s), s
    for s in ("Val-de-Marne", "contest", "dev_17", "train_ids", "station_42"):
        assert not SHARD_ID.search(s), s


def test_shard_ids_flagged_only_when_a_series_is_cut_across_files():
    from tideforge.tfbench.run import base_id, split_collisions
    lotsa = [f"pems03/data-0000{k}-of-00002.arrow/{k * 10 + j}" for k in range(2) for j in range(5)]
    chronos = ["nn5/train-00000-of-00001.parquet/NN5-009", "nn5/train-00000-of-00001.parquet/NN5-010"]
    assert split_collisions(lotsa) == [] and split_collisions(chronos) == []
    assert base_id("nn5/train-00000-of-00001.parquet/NN5-009") == "nn5/NN5-009"
    assert base_id("residential_load_power/train/0000.parquet/unit_73") == "residential_load_power/0000.parquet/unit_73"
    cut = ["data/train.csv/V", "data/test.csv/V", "data/train.csv/W"]
    assert [k for k, _ in split_collisions(cut)] == ["data/V"]
    shards = ["x/data-00000-of-00002.arrow/T1", "x/data-00001-of-00002.arrow/T1"]
    assert [k for k, _ in split_collisions(shards)] == ["x/T1"]
