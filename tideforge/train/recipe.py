"""The 5M recipe (configs/train/ab5m_cpm.yaml) merged with an experiment's train: section, and the run id.

    rc = recipe.load("E0.7")              # Recipe: the experiment, its generator config, the merged settings
    rc.ft3_cfg()                          # the cfg dict ft3's model / batcher code reads (ft3 CONFIG + ft3_main cfg + recipe)
    rc.run_id(seed, code)                 # '<exp>-<gen fragment>-t<train hash>-s<seed>-<code>'

train: keys an experiment may set (tideforge.expcfg.TRAIN_KEYS): long_row_frac (G3.3), lead_noise (G1.2), crop_breaks (G4.3),
and the recipe's own batch / optim / run values for the throughput and smoke configs (E0.9: max_steps, ...). The run id hashes
the generator config (its config_hash covers every switch parameter and the install plan), the merged train settings, the
seed and the code version, so changing any of them gives a new id.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from tideforge.expcfg import Exp, load_exp
from tideforge.gen.config import load_ft3_cfg

REPO = Path(__file__).resolve().parents[2]
TRAIN_DIR = REPO / "configs" / "train"
SECTIONS = ("model", "batch", "optim", "run")
RUN_EXTRA = {"max_steps": None, "smoke_points": None}          # run settings an experiment / the smoke script may add


def load_recipe(name: str = "ab5m_cpm") -> dict:
    d = yaml.safe_load((TRAIN_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
    return {k: dict(d[k]) for k in SECTIONS} | {"name": d["name"]}


def code_version(repo: Path = REPO) -> str:
    """git HEAD (12 chars), '+dirty' when tracked files differ from it."""
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "nogit"
    return sha[:12] + ("+dirty" if dirty else "")


@dataclass(frozen=True)
class Recipe:
    exp: Exp
    sections: dict
    items: dict = field(default_factory=dict)           # train-side items: lead_noise, crop_breaks

    def ft3_cfg(self) -> dict:
        """ft3 CONFIG + ft3_main's cfg + the recipe's model / batch values + the generator config's cfg keys."""
        from tideforge.train import thirdparty_ft3
        C = thirdparty_ft3()
        cfg = dict(C.CONFIG)
        cfg.update(load_ft3_cfg())
        m = {k: v for k, v in self.sections["model"].items() if k != "params"}
        cfg.update(m)
        cfg.update(self.sections["batch"])
        g = self.exp.gen
        cfg.update(g.to_ft3_cfg())
        cfg.update(group_prob=g.group_prob, group_lanes=g.group_lanes if g.group_lanes > 1 else cfg["group_lanes"],
                   val_corpus=self.sections["run"]["val_corpus"], val_group_prob=self.sections["run"]["val_group_prob"],
                   lr=self.sections["optim"]["peak_lr"], grad_clip=self.sections["optim"]["grad_clip"],
                   target_points=self.sections["optim"]["target_points"])
        return cfg

    def train_hash(self) -> str:
        s = json.dumps(dict(sections=self.sections, items=self.items), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]

    def run_id(self, seed: int, code: str) -> str:
        return f"{self.exp.run_id()}-t{self.train_hash()}-s{int(seed)}-{code.replace('+', '')}"


def merge(exp: Exp, recipe: dict | None = None) -> Recipe:
    rc = {k: dict(v) for k, v in (recipe or load_recipe()).items() if k in SECTIONS}
    items = {}
    for k, v in (exp.train or {}).items():
        if k in ("lead_noise", "crop_breaks"):
            items[k] = dict(v)
            continue
        for sec in SECTIONS:
            if k in rc[sec]:
                rc[sec][k] = v
                break
        else:
            if k in RUN_EXTRA:
                rc["run"][k] = v
            else:
                raise KeyError(f"{exp.id}: train setting {k!r} is not a recipe value, lead_noise or crop_breaks")
    return Recipe(exp, rc, items)


def load(exp_id: str, recipe: str = "ab5m_cpm") -> Recipe:
    return merge(load_exp(exp_id), load_recipe(recipe))
