"""Train a covariate adapter on a frozen backbone: the 5M optimiser recipe on covariate-only batches, budget 5B points (a config field).

    python -m tideforge.train.adapter_train --exp AD.2 --seed 0 --backbone-ckpt <run dir>/final.pt --out $TF_ROOT/runs --lr 3e-4
                                            [--points 1e9] [--no-finish] [--device cuda]

A trainer.Run with three changes: the backbone's weights are loaded and frozen, the optimiser holds the adapter parameters only, and
the batcher keeps blocks with covariate lanes only (train: cov_only). --lr and --points change the train hash, so every pilot has its own
run id. run.json records the backbone's sha256. final.pt loads through tideforge.model.load like any checkpoint (cfg: adapter_every),
and resume, requeue and finish are the trainer's.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

from tideforge.model import adapter as AD
from tideforge.probes.models import sha256_file
from tideforge.train import recipe as RC
from tideforge.train import trainer as TR

MATCH = ("d_model", "n_layers", "n_heads", "d_kv", "d_ff", "patch_len", "group_lanes", "known_attn", "attn_gate")


class AdapterRun(TR.Run):
    def __init__(self, rc, seed, out, device, code, backbone_ckpt: str | Path, log=print):
        super().__init__(rc, seed, out, device, code, log=log)
        self.backbone_ckpt = Path(backbone_ckpt)

    def build(self, check_params: bool = False):
        super().build(check_params=False)
        ck = torch.load(self.backbone_ckpt, map_location=self.device, weights_only=False)
        off = dict(known_attn="off", attn_gate="none")
        bad = [k for k in MATCH if ck["cfg"].get(k, off.get(k)) != self.cfg.get(k, off.get(k))]
        if bad:
            raise SystemExit(f"{self.backbone_ckpt}: differs from {self.rc.exp.id} in {bad}")
        res = self.model.load_state_dict(ck["model"], strict=False)
        if res.unexpected_keys or any(".adapter." not in k for k in res.missing_keys):
            raise SystemExit(f"backbone does not match: missing {res.missing_keys[:5]} unexpected {res.unexpected_keys[:5]}")
        self.n_adapter = AD.freeze_backbone(self.model)
        o = self.opt_cfg
        self.opt = torch.optim.AdamW(AD.adapter_parameters(self.model), lr=o["peak_lr"] * o["warmup_start_frac"],
                                     betas=tuple(o["betas"]), weight_decay=o["weight_decay"], fused=self.device.type == "cuda")
        return self

    def info(self) -> dict:
        return dict(super().info(), backbone=str(self.backbone_ckpt), backbone_sha256=sha256_file(self.backbone_ckpt),
                    n_adapter=self.n_adapter)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backbone-ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--points", type=float, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-finish", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    a = ap.parse_args(argv)
    code = RC.code_version()
    if code.endswith("+dirty") and not a.allow_dirty:
        raise SystemExit(f"code version {code}: commit first")
    rc = RC.load(a.exp)
    if not rc.items.get("cov_only") or not rc.exp.model.get("adapter_every"):
        raise SystemExit(f"{a.exp}: an adapter experiment needs train: cov_only and model: adapter_every")
    rc.sections["optim"]["peak_lr"] = a.lr
    if a.points:
        rc.sections["optim"]["target_points"] = a.points
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None and torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    TR.install_signals()
    run = AdapterRun(rc, a.seed, Path(a.out), a.device, code, a.backbone_ckpt)
    if (run.dir / "final.pt").exists():
        print(f"run {run.run_id} is already finished; nothing to do", flush=True)
        return TR.EXIT_DONE
    run.build()
    print(f"run {run.run_id}: {run.n_adapter:,} adapter parameters, {run.n_params:,} in all, lr {a.lr:g}, "
          f"{rc.sections['optim']['target_points'] / 1e9:g}B points", flush=True)
    res = run.train(max_steps=a.max_steps)
    if res == TR.EXIT_DONE and not a.no_finish:
        from tideforge.train import finish
        finish.finish(run.dir / "final.pt", run.dir, device=a.device)
    return res


if __name__ == "__main__":
    sys.exit(main())
