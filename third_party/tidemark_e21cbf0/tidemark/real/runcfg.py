"""Run configuration of the real-data post-training and the pieces both the loader benchmark and the trainer read.

Everything that is a decision of the owner is a field here (budget, ratio, rows, LR), none is hard-coded elsewhere. The synthetic side is
ft3_main's own configuration: the checkpoint's `cfg` dict is the base of the batcher / model config and is only overridden where
`overrides` says so.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from tidemark.real.sampler import MixConfig


@dataclass
class RunConfig:
    name: str = "real30b"
    ckpt: str = "/pscratch/sd/n/naresh/tidemark/ckpt/ft3_main.pt"          # weights-only; the optimizer starts fresh
    real_root: str = ""                                                      # $SCRATCH/tidemark/real (shards/, plan.json)
    out_dir: str = ""                                                        # $SCRATCH/tidemark/real_run/<name>
    copy_to: str = ""                                                        # permanent storage for milestone files ("" = none)
    plan: str = ""                                                           # plan.json (default <real_root>/plan.json)
    shards_dir: str = "shards"                                               # folder of the converted datasets under real_root (shards_full: all 107 hourly shards)
    plan_file: str = "plan.json"                                             # file name of the plan under real_root when `plan` is empty
    target_points: float = 30e9                                              # total VALID points = real + synthetic
    milestones: tuple[float, ...] = (10e9, 20e9, 30e9)
    peak_lr: float = 1.2e-4
    warmup_frac: float = 0.05
    warmup_start_frac: float = 0.04                                          # first LR = peak x this (ft3's OneCycle starts at peak/25)
    final_lr_frac: float = 0.10
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    dropout: float = 0.1
    seed: int = 0
    mix: MixConfig = field(default_factory=MixConfig)
    val_batches: int = 2                                                     # synthetic holdout (forgetting monitor)
    val_every: int = 400
    print_every: int = 20
    save_every_s: float = 600.0                                              # last.pt at least this often
    compile: bool = True
    max_hours: float | None = None                                           # stop cleanly (last.pt) after this many hours
    overrides: dict = field(default_factory=dict)                            # ft3 cfg keys changed for this run (none by default)

    def resolve(self) -> "RunConfig":
        scratch = os.environ.get("SCRATCH", ".")
        self.real_root = self.real_root or str(Path(scratch) / "tidemark" / "real")
        self.out_dir = self.out_dir or str(Path(scratch) / "tidemark" / "real_run" / self.name)
        self.plan = self.plan or str(Path(self.real_root) / self.plan_file)
        return self

    def to_json(self) -> dict:
        d = asdict(self)
        d["milestones"] = list(self.milestones)
        return d

    @staticmethod
    def from_json(d: dict) -> "RunConfig":
        d = dict(d)
        d["mix"] = MixConfig(**d.get("mix", {}))
        d["milestones"] = tuple(d.get("milestones", ()))
        return RunConfig(**d)


def real200b_config() -> RunConfig:
    """real200b (owner, 2026-10-25): 100B real + 100B synthetic (1:1 per step on every rank, MixConfig.real_fraction), the full weatherbench pool
    (shards_full), a checkpoint every 20B points, LR 1.2e-4 with 5% warmup and a cosine to 10% over the whole 200B. Everything else as real30b."""
    from dataclasses import replace
    b = RunConfig()
    return replace(b, name="real200b", target_points=200e9, milestones=tuple(20e9 * i for i in range(1, 11)), shards_dir="shards_full",
                   plan_file="plan_200b.json", mix=replace(b.mix, real_fraction=0.5), val_every=1000, print_every=50)


RUNS = {"real30b": RunConfig, "real200b": real200b_config}
SMOKE_POINTS = {"real30b": 1e9, "real200b": 2e9}


def smoke_config(base: RunConfig | None = None, points: float = 1e9) -> RunConfig:
    """A smoke run of the same configuration: `points` total (1B for real30b: about 0.33B real; 2B for real200b: 1B real at 1:1), milestones at
    half and at the end, the LR schedule over the smoke budget, same everything else."""
    from dataclasses import replace
    b = base or RunConfig()
    return replace(b, name=f"smoke{points / 1e9:g}b", target_points=points, milestones=(points / 2, points), val_every=100, print_every=10)


def load_base_cfg(ckpt_path: str) -> dict:
    import torch
    return dict(torch.load(ckpt_path, map_location="cpu", weights_only=False)["cfg"])


def model_cfg(base: dict, rc: RunConfig) -> dict:
    """The ft3 cfg used for model and batchers: the checkpoint's, dropout from the run, rows from the mix, then explicit overrides."""
    return {**base, "dropout": rc.dropout, "pack_rows": rc.mix.rows, "seed": rc.seed, **rc.overrides}


def load_plan(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def plan_targets(plan: dict) -> dict[str, float]:
    return {d: float(s) for d, s in plan["targets"].items() if s > 0}
