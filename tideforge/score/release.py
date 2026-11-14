"""The tfbench release on disk (fetched once), its task lists by tier and split, the locked guard and the locked ledger; the
fev-bench task list (from the old scorer's evaluation root) behind the same interface.

A task spec is a dict: {"kw": fev.Task kwargs with a LOCAL dataset_path, "row": the manifest row (task, dataset,
source_group, config, domain, freq, band, horizon_class, H, variant, multivariate, split, weight, covariates, ...)}.

Locked tasks are reachable only with locked=True (run.py --locked); a tuning run never opens locked/. Every locked run is
appended to results/tfbench/locked_ledger.csv in the repo (committed) and to $TF_ROOT/looks_pending.md (looks.py).
"""
from __future__ import annotations

import csv
import datetime as dt
import os
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[2]
LEDGER = REPO / "results" / "tfbench" / "locked_ledger.csv"
LEDGER_COLS = ["date", "model_id", "model", "version", "tier", "groups", "n_tasks", "skill_MASE", "skill_WQL", "out"]
TIERS = ("fast", "core", "full")
DEFAULT_REPO = "nareshmeena12/tfbench"


class LockedError(PermissionError):
    """A locked task was requested without --locked."""


def default_release_root() -> Path:
    r = os.environ.get("TF_ROOT")
    return Path(r) / "tfbench_release" if r else REPO / "data" / "tfbench_release"


def fetch(version: str, root: Path | None = None, repo: str | None = None) -> Path:
    """Download the release at tag `version` into root/version (resumable; run on a node with internet). The token comes
    from HF_TOKEN in the environment and is passed through, never printed or written."""
    from huggingface_hub import snapshot_download
    dest = Path(root or default_release_root()) / version
    snapshot_download(repo_id=repo or os.environ.get("TF_BENCH_REPO") or DEFAULT_REPO, repo_type="dataset", revision=version,
                      local_dir=str(dest), token=os.environ.get("HF_TOKEN") or None)
    return dest


class Release:
    def __init__(self, path: Path):
        self.path = Path(path)
        if not (self.path / "manifest.csv").exists():
            raise FileNotFoundError(f"{self.path}: no manifest.csv (fetch the release first: run fetch --version ...)")
        man = pd.read_csv(self.path / "manifest.csv")
        tags = pd.read_csv(self.path / "tags.csv") if (self.path / "tags.csv").exists() else pd.DataFrame(columns=["task"])
        keep = [c for c in ("task", "covariates", "intermittency", "history_length_band", "series_length_band") if c in tags]
        self.manifest = man.merge(tags[keep], on="task", how="left")
        self.version = self.path.name

    def tasks(self, tier: str, split: str = "tuning", locked: bool = False) -> list[dict]:
        if tier not in TIERS:
            raise ValueError(f"tier {tier!r}: one of {TIERS}")
        if split not in ("tuning", "locked"):
            raise ValueError(f"split {split!r}: tuning or locked")
        if split == "locked" and not locked:
            raise LockedError("locked tasks need --locked (final reporting only; never for selection)")
        base = self.path / ("locked/tasks" if split == "locked" else "tasks")
        tl = (yaml.safe_load((base / f"{tier}.yaml").read_text(encoding="utf-8")) or {}).get("tasks", [])
        man = self.manifest[(self.manifest["split"] == split)].set_index("task")
        out = []
        for t in tl:
            kw = dict(t)
            cfg = kw.get("dataset_config") or Path(kw["dataset_path"]).parent.name
            data = self.path / ("locked/data" if split == "locked" else "data") / cfg / "data.parquet"
            kw.update(dataset_path=str(data), dataset_config=None)
            name = kw["task_name"]
            if name not in man.index:
                raise KeyError(f"task {name} is in {tier}.yaml but not in manifest.csv ({split})")
            row = man.loc[name].to_dict()
            row.update(task=name, weight=float(row.get(f"weight_{tier}", 1.0)), tier=tier, split=split)
            out.append({"kw": kw, "row": row})
        return out


def fevbench_tasks(eval_root: Path) -> list[dict]:
    """fev-bench tasks from the old scorer's evaluation root (eval/data/fev/tasks.yaml): weight 1 per task, each task its
    own source group; domain from catalog/fev_domains.csv when listed."""
    p = Path(eval_root) / "eval" / "data" / "fev" / "tasks.yaml"
    tl = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("tasks", [])
    dom: dict[str, str] = {}
    f = REPO / "catalog" / "fev_domains.csv"
    if f.exists():
        d = pd.read_csv(f)
        dom = {str(k).lower(): str(v) for k, v in zip(d["dataset"], d["domain"])}

    def domain_of(name: str) -> str:            # longest catalog dataset name that prefixes the task's config / name
        n = str(name).lower()
        hits = [k for k in dom if n.startswith(k)]
        return dom[max(hits, key=len)] if hits else "unknown"
    out = []
    for t in tl:
        kw = dict(t)
        name = kw.get("task_name") or kw.get("dataset_config")
        kw["task_name"] = name
        cov = "known" if kw.get("known_dynamic_columns") else "past" if kw.get("past_dynamic_columns") else "none"
        tgt = kw.get("target", "target")
        out.append({"kw": kw, "row": dict(task=name, dataset=name, source_group=name,
                                          domain=domain_of(kw.get("dataset_config") or name), horizon_class="",
                                          covariates=cov, multivariate=isinstance(tgt, list) and len(tgt) > 1, weight=1.0,
                                          tier="fev-bench", split="fev-bench")})
    return out


def ledger_append(row: dict, path: Path = LEDGER) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LEDGER_COLS, lineterminator="\n")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in LEDGER_COLS} | {"date": row.get("date") or dt.date.today().isoformat()})
