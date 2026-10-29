"""Single-GPU 5M trainer (T3): the ab5m_cpm recipe on tideforge.gen data, resumable, one experiment per GPU.

    python -m tideforge.train.trainer --exp E0.7 --seed 0 --out $TF_ROOT/runs [--device cuda] [--smoke-points 5e8]
                                      [--max-steps N] [--gen-mode det|default] [--allow-dirty] [--no-finish]

A step: the next batch is generated on the MAIN thread (owner 2026-10-29: torch.use_deterministic_algorithms is process-global, so
generation never overlaps a model kernel launch; the batcher sets the flag around generation only), then forward in bf16
through the loss-masked decoder, backward, clip 1.0, AdamW (0.9, 0.95), weight decay 0.01; the LR is a function of the points
seen: linear warmup over 2% of the budget from peak / 25, then cosine to 0 (lr_at). Generation time is logged as gen_frac.

Run directory <out>/<run id> (run id: recipe.Recipe.run_id = experiment + generator fragment + train hash + seed + code):
  run.json       the merged recipe, the generator config, code version, parameter count (written once; a resume checks it)
  last.pt        resumable state: model, optimizer, step, points, batcher state, CPU / CUDA RNG states (captured between
                 steps, just before the next batch), every ckpt_every_s seconds and at a stop
  ckpt_<N>B.pt   milestones and final.pt: weights only, in ft3_main.pt's format (model, cfg, step, points_seen, packed): the
                 scorers and the probes load them (tideforge.probes.models.M5)
  metrics.jsonl  one line per print_every steps (loss, lr, points/s, gen_frac) and per validation (holdout loss: a training
                 monitor only; the holdout is B0's own generator, so it favours B0's distribution. Decisions use the dev
                 score and the probes, never the holdout loss)
SIGUSR1 / SIGTERM: the current step finishes, last.pt is saved, exit code 3 (the batch script requeues the job). A requeued run
continues from last.pt and regenerates the very next batch (batcher pools are pure functions of seed and pool number).
At the end: final.pt, then tideforge.train.finish (dev fast score + probes + one results row) unless --no-finish.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import sys
import time
from pathlib import Path

import torch

import tideforge.gen as G
from tideforge import model as TM
from tideforge.train import batcher as TB
from tideforge.train import recipe as RC

EXIT_DONE, EXIT_REQUEUE = 0, 3
_STOP = {"flag": False, "why": ""}


def _on_signal(signum, frame):
    _STOP.update(flag=True, why=signal.Signals(signum).name)


def install_signals():
    for s in ("SIGUSR1", "SIGTERM"):
        if hasattr(signal, s):
            signal.signal(getattr(signal, s), _on_signal)


def lr_at(points: float, opt: dict) -> float:
    f = min(max(points / opt["target_points"], 0.0), 1.0)
    w, peak = opt["warmup_frac"], opt["peak_lr"]
    if f < w:
        return peak * (opt["warmup_start_frac"] + (1 - opt["warmup_start_frac"]) * f / w)
    prog = (f - w) / (1 - w)
    return peak * (opt["final_lr_frac"] + (1 - opt["final_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * prog)))


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def slim(raw, cfg, step, points, run_id, note) -> dict:
    return dict(model=raw.state_dict(), cfg=cfg, step=step, points_seen=points, packed=True, synthetic_only=True,
                run_name=run_id, note=note)


class Run:
    """Everything one run needs; build() does the setup, train() the loop (tests drive both with a tiny recipe)."""

    def __init__(self, rc: RC.Recipe, seed: int, out: Path, device: str, code: str, gen_mode: str | None = None,
                 log=print):
        self.rc, self.seed, self.code, self.log = rc, int(seed), code, log
        self.device = torch.device(device)
        self.run_id = rc.run_id(seed, code) + (f"-{gen_mode}" if gen_mode else "")
        self.dir = Path(out) / self.run_id
        self.det = rc.sections["run"]["deterministic_gen"] if gen_mode is None else gen_mode == "det"
        self.opt_cfg = dict(rc.sections["optim"])

    # ---------------------------------------------------------------------------------------------------- setup
    def build(self, check_params: bool = True):
        from tideforge.train import thirdparty_ft3
        C = thirdparty_ft3()
        rc, dev = self.rc, self.device
        self.cfg = rc.ft3_cfg()
        torch.manual_seed(self.seed)
        self.model = TM.build(self.cfg).to(dev)
        n = TM.n_params(self.model)
        want = rc.sections["model"].get("params")
        if check_params and want and n != int(want):
            raise SystemExit(f"parameter count {n:,} != the recipe's {int(want):,}")
        self.n_params = n
        # the holdout: ft3_core's own generator and val corpus (B0's, fixed seed); it never touches tideforge.gen's state
        C.set_group_options(None)
        self.val = C.build_val_batches(dict(self.cfg, corpus=self.cfg["corpus"]), dev, rc.sections["run"]["val_batches"])
        self.C = C
        prof = G.configure(rc.exp.gen, seed=self.seed)
        self.batcher = TB.PackedBatcher(self.cfg, dev, prof, seed=self.seed + 1, deterministic_gen=self.det,
                                        crop_breaks=rc.items.get("crop_breaks"), lead_noise=rc.items.get("lead_noise"))
        o = self.opt_cfg
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=o["peak_lr"] * o["warmup_start_frac"],
                                     betas=tuple(o["betas"]), weight_decay=o["weight_decay"], fused=dev.type == "cuda")
        self.fwd = torch.compile(self.model) if (rc.sections["run"]["compile"] and dev.type == "cuda") else self.model
        self.amp_on = dev.type == "cuda" and rc.sections["run"]["amp"] == "bf16"
        self.step, self.points = 0, 0.0
        return self

    def info(self) -> dict:
        return dict(run_id=self.run_id, exp=self.rc.exp.id, seed=self.seed, code=self.code, recipe=self.rc.sections,
                    items=self.rc.items, gen=self.rc.exp.gen.diff(self.rc.exp.base), gen_hash=self.rc.exp.gen.config_hash(),
                    deterministic_gen=self.det, n_params=self.n_params, device=str(self.device))

    # ---------------------------------------------------------------------------------------------------- state
    def state(self) -> dict:
        return dict(model=self.model.state_dict(), opt=self.opt.state_dict(), step=self.step, points=self.points,
                    batcher=self.batcher.state_dict(), cpu=torch.get_rng_state(),
                    cuda=torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
                    run_id=self.run_id, cfg=self.cfg)

    def load(self, st: dict) -> None:
        if st["run_id"] != self.run_id:
            raise SystemExit(f"last.pt belongs to {st['run_id']}, not {self.run_id}")
        self.model.load_state_dict(st["model"])
        self.opt.load_state_dict(st["opt"])
        self.step, self.points = int(st["step"]), float(st["points"])
        self.batcher.load_state_dict(st["batcher"])
        torch.set_rng_state(st["cpu"])
        if st["cuda"] is not None:
            torch.cuda.set_rng_state(st["cuda"], self.device)

    def save_last(self) -> None:
        atomic_save(self.state(), self.dir / "last.pt")

    # ---------------------------------------------------------------------------------------------------- loop
    def one_step(self) -> float:
        """Generate (main thread), forward, backward, clip, step. Returns the loss (nan if skipped)."""
        b = self.batcher.next_batch()
        for gp in self.opt.param_groups:
            gp["lr"] = lr_at(self.points, self.opt_cfg)
        with torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp_on):
            loss = TB.packed_forward(self.fwd, b)["loss"]
        self.step += 1
        if not torch.isfinite(loss):
            self.opt.zero_grad(set_to_none=True)
            self.n_bad += 1
            return float("nan")
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt_cfg["grad_clip"])
        self.opt.step()
        self.points += b["points"]
        return float(loss.detach())

    def validate(self) -> dict:
        vl, bands = self.C.evaluate_packed(self.fwd, self.val, torch.bfloat16, self.amp_on, self.device)
        self.model.train()
        return dict(kind="val", step=self.step, points=self.points, holdout=vl, **{f"ctx {k}": v for k, v in bands.items()})

    def train(self, max_steps: int | None = None, target_points: float | None = None) -> int:
        rs = self.rc.sections["run"]
        target = float(target_points or self.opt_cfg["target_points"])
        max_steps = max_steps or rs.get("max_steps")
        self.dir.mkdir(parents=True, exist_ok=True)
        info = self.info()
        rj = self.dir / "run.json"
        if rj.exists():
            old = json.loads(rj.read_text(encoding="utf-8"))
            if old["run_id"] != info["run_id"]:
                raise SystemExit(f"{rj} is another run")
        else:
            rj.write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
        if (self.dir / "last.pt").exists():
            self.load(torch.load(self.dir / "last.pt", map_location=self.device, weights_only=False))
            self.log(f"resumed {self.run_id} at step {self.step:,}, {self.points / 1e9:.2f}B points")
        mpts = float(self.opt_cfg.get("milestone_points") or 0)
        next_ms = (math.floor(self.points / mpts) + 1) * mpts if mpts > 0 else float("inf")
        self.n_bad = 0
        self.model.train()
        mfile = open(self.dir / "metrics.jsonl", "a", encoding="utf-8")
        t_save = t_win = time.time()
        g0, pts_win, loss_win, n_win = self.batcher.gen_seconds, 0.0, 0.0, 0
        try:
            while self.points < target and (max_steps is None or self.step < max_steps):
                p0 = self.points
                loss = self.one_step()
                pts_win += self.points - p0
                if loss == loss:
                    loss_win += loss
                    n_win += 1
                if self.step % rs["print_every"] == 0:
                    if self.device.type == "cuda":
                        torch.cuda.synchronize()
                    el = time.time() - t_win
                    row = dict(kind="train", step=self.step, points=self.points, loss=loss_win / max(n_win, 1),
                               lr=self.opt.param_groups[0]["lr"], pts_per_s=pts_win / max(el, 1e-9),
                               gen_frac=(self.batcher.gen_seconds - g0) / max(el, 1e-9), bad=self.n_bad)
                    mfile.write(json.dumps(row) + "\n")
                    mfile.flush()
                    self.log(f"step {self.step:,} loss {row['loss']:.4f} lr {row['lr']:.2e} {row['pts_per_s'] / 1e6:.2f}M pts/s "
                             f"gen {row['gen_frac']:.1%} {self.points / 1e9:.2f}B")
                    t_win, g0, pts_win, loss_win, n_win = time.time(), self.batcher.gen_seconds, 0.0, 0.0, 0
                if self.step % rs["val_every"] == 0:
                    mfile.write(json.dumps(self.validate()) + "\n")
                    mfile.flush()
                if self.points >= next_ms:
                    atomic_save(slim(self.model, self.cfg, self.step, self.points, self.run_id, "milestone"),
                                self.dir / f"ckpt_{int(round(next_ms / 1e9))}B.pt")
                    next_ms += mpts
                if time.time() - t_save >= rs["ckpt_every_s"]:
                    self.save_last()
                    t_save = time.time()
                if _STOP["flag"]:
                    self.save_last()
                    self.log(f"{_STOP['why']}: saved last.pt at step {self.step:,}; exit {EXIT_REQUEUE} (requeue)")
                    return EXIT_REQUEUE
            v = self.validate()
            mfile.write(json.dumps(dict(v, kind="final_val")) + "\n")
            atomic_save(slim(self.model, self.cfg, self.step, self.points, self.run_id, "final"), self.dir / "final.pt")
            self.save_last()
            self.log(f"done {self.run_id}: {self.step:,} steps, {self.points / 1e9:.2f}B points, holdout {v['holdout']:.4f}")
            return EXIT_DONE
        finally:
            mfile.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--recipe", default="ab5m_cpm")
    ap.add_argument("--smoke-points", type=float, default=None, help="train this many points (LR schedule over them)")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--gen-mode", choices=["det", "default"], default=None, help="E0.9: override the generator mode")
    ap.add_argument("--allow-dirty", action="store_true", help="smoke / throughput only: run from a tree with changes")
    ap.add_argument("--no-finish", action="store_true")
    a = ap.parse_args(argv)
    code = RC.code_version()
    if code.endswith("+dirty") and not a.allow_dirty:
        raise SystemExit(f"code version {code}: commit first (real runs need a clean tree; smoke runs: --allow-dirty)")
    rc = RC.load(a.exp, a.recipe)
    if a.smoke_points:
        rc.sections["optim"]["target_points"] = float(a.smoke_points)
        rc.sections["optim"]["milestone_points"] = 0.0
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") is None and torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"           # needed by cuBLAS whenever deterministic mode is on
    install_signals()
    run = Run(rc, a.seed, Path(a.out), a.device, code, a.gen_mode)
    if (run.dir / "final.pt").exists():                 # a requeued node job restarts every slot: finished runs stop here
        print(f"run {run.run_id} is already finished ({run.dir / 'final.pt'}); nothing to do", flush=True)
        return EXIT_DONE
    run.build()
    print(f"run {run.run_id}: {run.n_params:,} parameters, generator {'deterministic' if run.det else 'default'} mode, "
          f"{rc.sections['batch']['pack_rows']} rows/step, target {rc.sections['optim']['target_points'] / 1e9:g}B points",
          flush=True)
    rc_ = run.train(max_steps=a.max_steps)
    if rc_ == EXIT_DONE and not a.no_finish:
        from tideforge.train import finish
        finish.finish(run.dir / "final.pt", run.dir, device=a.device)
    return rc_


if __name__ == "__main__":
    sys.exit(main())
