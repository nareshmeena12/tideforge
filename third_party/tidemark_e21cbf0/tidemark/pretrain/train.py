"""DDP training of one twin (LEAN_PLAN 4.4). Launched with torchrun, one process per GPU:

  torchrun --standalone --nproc_per_node=4 -m tidemark.pretrain.train \\
      --root $SCRATCH/tidemark --arm gla --seed 0 --lr-from pilot      (twin, rate from the pilot)
      --root ... --arm gla --seed 0 --lr 3e-4 --pause-points 5e9        (one pilot run)

Without torchrun it runs as one process (tests, CPU). What it does:
  * per-rank data from DataStream: a function of (seed, rank, step) only; every rank's
    batch is global_rows / world rows, cut into micro-batches (gradient accumulation) when
    fewer rows fit on a GPU, so the data never depends on the GPU;
  * ONE all-reduce per step of [loss sum, series, points, grad norm, rank 0's save and stop
    requests]: global loss, global points, the LR schedule on global points, and every skip /
    rollback / stop decision are identical on all ranks;
  * holdout every val_every steps, sharded over ranks and all-reduced; the probes run on
    rank 0 (reported only);
  * rank 0 writes checkpoints, then a barrier: last.pt (every ~10 min, at a pause, on STOP),
    best.pt (best holdout; rollbacks reload it), milestones/ckpt_25B.pt and ckpt_50B.pt,
    final.pt (weights only) and summary.json;
  * resume from last.pt; a STOP file in the run dir (written by the sbatch script's SIGUSR1
    trap two minutes before the time limit) -> save last.pt and exit with code 3, and the
    sbatch script requeues the job;
  * torch.compile on everything but the time mixer (owner, 2026-10-15).
Exit codes: 0 finished or paused (pilot), 3 saved for a requeue, 4 unstable (too many
rollbacks), 5 nothing to do (arm blocked in pilot_choice.json).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
import threading
import signal
import time
from pathlib import Path

import torch
import torch.distributed as dist

from tidemark.pretrain import safety as S
from tidemark.models.kernel_dims import check_model
from tidemark.pretrain.arms import ALL_ARMS, STAGE_B_DONOR, canonical, lr_donor, uses
from tidemark.pretrain.scale import SIZES, sized_config
from tidemark.pretrain.config import (M4_ARMS, PILOT_POINTS, TwinConfig, config_from_json, config_json, run_id,
                                      twin_config)
from tidemark.pretrain.model import build_twin, param_report
from tidemark.pretrain.rerun import rerun_tag, split_run_name
from tidemark.pretrain.packing import DataStream, holdout_shard, loss_series, packed_forward, slice_batch

EXIT_DONE, EXIT_REQUEUE, EXIT_UNSTABLE, EXIT_BLOCKED = 0, 3, 4, 5
SAVE_EVERY_S = 600
VAL_BANDS = [(1, 64), (65, 512), (513, 2048), (2049, 10 ** 9)]


# ================================ paths =====================================

def run_dir(root, cfg: TwinConfig, rerun: bool = False) -> Path:
    """30M twins: <root>/runs (unchanged). Other sizes (Stage B): <root>/runs_<size>, so a harness pointed at one
    folder never sees two models with the same <arm>_s<seed> tag. Reruns at another start rate (--rerun) go to
    <root>/runs_<size>_rerun, never into the folder of the original twins."""
    from tidemark.pretrain.scale import size_of
    size = size_of(cfg)
    folder = "runs" if size == "30m" else f"runs_{size}"
    return Path(root) / (folder + "_rerun" if rerun else folder) / f"{cfg.name}_{run_id(cfg)}"


def rerun_meta(root, cfg: TwinConfig) -> dict:
    """What a rerun records about itself: that it is one, and the original twin (same arm, size, seed) it corresponds to."""
    from tidemark.pretrain.scale import size_of
    size = size_of(cfg)
    orig_dirs = sorted((Path(root) / ("runs" if size == "30m" else f"runs_{size}")).glob(f"{cfg.arm}_s{cfg.seed}_lr*"))
    orig = []
    for o in orig_dirs:
        item = {"run_dir": o.name, "start_lr": None, "final_summary": (o / "summary.json").exists()}
        try:
            item["start_lr"] = float(json.loads((o / "summary.json").read_text())["lr"])
        except Exception:  # noqa: BLE001
            parsed = split_run_name(o.name)
            item["start_lr"] = parsed["lr"] if parsed else None
        orig.append(item)
    return {"rerun": True, "rerun_of": orig[0] if len(orig) == 1 else (orig or None),
            "rerun_tag": rerun_tag(cfg.arm, cfg.seed, cfg.lr), "rerun_start_lr": cfg.lr}


def checks_dir(root) -> Path:
    return Path(root) / "checks"


def human(n: float) -> str:
    for u in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.2f}{u}"
        n /= 1000
    return f"{n:.2f}P"


# ============================== distributed =================================

class Dist:
    def __init__(self, device_arg: str | None = None):
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.rank = int(os.environ.get("RANK", "0"))
        self.local = int(os.environ.get("LOCAL_RANK", "0"))
        cuda = torch.cuda.is_available() and device_arg != "cpu"
        self.device = torch.device(f"cuda:{self.local}") if cuda else torch.device("cpu")
        if cuda:
            torch.cuda.set_device(self.device)
        self.on = self.world > 1
        if self.on and not dist.is_initialized():
            dist.init_process_group("nccl" if cuda else "gloo")

    def all_sum(self, t: torch.Tensor) -> torch.Tensor:
        if self.on:
            dist.all_reduce(t)
        return t

    def barrier(self):
        if self.on:
            dist.barrier()

    def close(self):
        if self.on and dist.is_initialized():
            dist.destroy_process_group()


_SIGTERM = threading.Event()


def install_sigterm_stop() -> None:
    """Preemption path (stage_b_twin_preempt.sbatch sets TIDEMARK_SIGTERM_STOP=1): a SIGTERM that reaches a training process
    (Slurm signals every process of a preempted job, torchrun included) sets a flag instead of killing it. The training loop
    sees the flag at the end of the current step, saves last.pt and exits with the requeue code. Not installed otherwise."""
    signal.signal(signal.SIGTERM, lambda *_: _SIGTERM.set())


def stop_requested(out: Path | None) -> bool:
    """A STOP file in the run dir, the file named by $TIDEMARK_STOP_FILE (written by the sbatch script's USR1 / TERM trap), or a
    SIGTERM received by this process (install_sigterm_stop)."""
    env = os.environ.get("TIDEMARK_STOP_FILE")
    return bool(_SIGTERM.is_set() or (out is not None and (out / "STOP").exists()) or (env and Path(env).exists()))


def _dropout_seed(seed: int, rank: int, step: int) -> int:
    return ((seed * 1_000_003 + rank) * 1_000_003 + step) % (2 ** 62)


# ================================ options ===================================

class Options:
    """What does not change results (not in the run id)."""

    def __init__(self, micro_rows: int | None = None, compile: bool = True, packed_kernel: str = "chunk",
                 gla_backend: str = "fla", pause_points: float | None = None, max_steps: int | None = None,
                 save: bool = True, validate: bool = True, probes: bool = True, save_every_s: float = SAVE_EVERY_S,
                 log=print, batch_hook=None, bench_warm: int | None = None, meta: dict | None = None):
        self.micro_rows, self.compile, self.packed_kernel = micro_rows, compile, packed_kernel
        self.gla_backend, self.pause_points, self.max_steps = gla_backend, pause_points, max_steps
        self.save, self.validate, self.probes, self.save_every_s = save, validate, probes, save_every_s
        self.log = log
        self.batch_hook = batch_hook          # tests only: (batch, rank, step) -> batch
        self.bench_warm = bench_warm          # bench: throughput measured after this many steps
        self.meta = meta or {}                # extra summary.json fields (reruns); never part of the run id


# ================================ holdout ===================================

@torch.no_grad()
def evaluate(model, batches, d: Dist, bf16: bool) -> dict:
    """Holdout loss (mean over series), by context-length band, and one- / multi-patch loss;
    sums all-reduced over ranks, so the value does not depend on the number of GPUs."""
    was = model.training
    model.eval()
    acc = torch.zeros(2 * (len(VAL_BANDS) + 3), dtype=torch.float64, device=d.device)
    for b in batches:
        with torch.autocast(d.device.type, dtype=torch.bfloat16, enabled=bf16 and d.device.type == "cuda"):
            out = packed_forward(model, b)
        v = b["slot_valid"]
        loss, c = out["per_series_loss"][v].double(), b["slot_ctx_len"][v]
        fin = torch.isfinite(loss)
        loss, c = loss[fin], c[fin]
        acc[0] += loss.sum()
        acc[1] += loss.numel()
        for i, (lo, hi) in enumerate(VAL_BANDS):
            m = (c >= lo) & (c <= hi)
            acc[2 + 2 * i] += loss[m].sum()
            acc[3 + 2 * i] += m.sum()
        for j, key in enumerate(("_per_one", "_per_multi")):
            x = out[key].double()
            x = x[torch.isfinite(x)]
            acc[2 + 2 * len(VAL_BANDS) + 2 * j] += x.sum()
            acc[3 + 2 * len(VAL_BANDS) + 2 * j] += x.numel()
    d.all_sum(acc)
    model.train(was)
    mean = lambda i: float(acc[i] / acc[i + 1]) if acc[i + 1] > 0 else float("nan")  # noqa: E731
    out = {"holdout": mean(0)}
    for i, (lo, hi) in enumerate(VAL_BANDS):
        out[f"ctx {lo}-{hi if hi < 10 ** 9 else 'max'}"] = mean(2 + 2 * i)
    out["one_patch"] = mean(2 + 2 * len(VAL_BANDS))
    out["multi_patch"] = mean(4 + 2 * len(VAL_BANDS))
    return out


# =============================== checkpoints ================================

def _atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def gla_kernel_choice(root) -> str:
    """The fla kernel for packed GLA rows, from the debug job's check (checks/gla_kernel.json)."""
    p = checks_dir(root) / "gla_kernel.json"
    if not p.exists():
        raise SystemExit(f"{p} is missing: run the debug job (GPU tests + GLA kernel check) before training GLA")
    return json.loads(p.read_text())["kernel"]


def micro_rows_choice(root, size: str = "30m") -> int | None:
    p = checks_dir(root) / ("bench_rows.json" if size == "30m" else f"bench_rows_{size}.json")
    return int(json.loads(p.read_text())["micro_rows"]) if p.exists() else None


# ================================ training ==================================

def train_twin(cfg: TwinConfig, opts: Options, out: Path | None, d: Dist | None = None) -> dict:
    """Train (or resume) one twin. Returns a summary dict with "exit" (an EXIT_* code)."""
    d = d or Dist()
    log = (lambda *a, **k: None) if d.rank else opts.log
    tc, dev = cfg.train, d.device
    bf16 = tc.bf16 and dev.type == "cuda"
    M = cfg.data.max_segs_per_row
    rows = tc.global_rows // d.world
    micro = min(opts.micro_rows or rows, rows)
    if rows % micro or (micro % cfg.model.group_lanes):
        raise SystemExit(f"micro_rows {micro} must divide the {rows} rows per rank in whole lane blocks")
    accum = rows // micro
    torch.manual_seed(cfg.seed)
    raw = build_twin(cfg, opts.gla_backend, opts.packed_kernel).to(dev)
    check_model(raw)                           # a clear error instead of Triton's 'K >= 16' assertion in the backward pass
    model = raw
    if d.on:
        model = torch.nn.parallel.DistributedDataParallel(raw, device_ids=[d.local] if dev.type == "cuda" else None)
    fwd = torch.compile(model) if opts.compile else model
    opt = torch.optim.AdamW(raw.parameters(), lr=cfg.lr, betas=tc.betas, weight_decay=tc.weight_decay,
                            fused=dev.type == "cuda")
    guard = S.SpikeGuard(tc.spike_warm, tc.spike_n, tc.spike_loss_x, tc.spike_gn_x)
    watch = S.WindowWatch(tc.win_rollback_x, tc.win_rollback_count)
    st = dict(step=0, points=0.0, peak_lr=cfg.lr, rollbacks=0, best_holdout=float("inf"), n_skip=0, n_bad=0,
              train_s=0.0, history=[], milestones_done=[])
    pr = param_report(raw)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        if opts.meta and d.rank == 0:
            (out / "rerun.json").write_text(json.dumps(opts.meta, indent=1))
        if (out / "final.pt").exists() and opts.pause_points is None:
            log(f"{out.name}: final.pt exists, the twin is finished; nothing to do")
            return {"exit": EXIT_DONE, "run": cfg.name, "finished_before": True}
    last = out / "last.pt" if out is not None else None
    if last is not None and last.exists():
        ck = torch.load(last, map_location=dev, weights_only=False)
        if ck["config_json"] != config_json(cfg):
            raise SystemExit(f"{last} belongs to another config")
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        st.update(ck["state"])
        guard.load_state_dict(ck["guard"])
        watch.load_state_dict(ck["watch"])
        log(f"resumed {out.name} at step {st['step']:,}, {human(st['points'])} points, peak LR {st['peak_lr']:.2e}, "
            f"{st['rollbacks']} rollbacks")
    start_step = st["step"]
    stream = DataStream(cfg, dev, d.rank, d.world)
    holdout = holdout_shard(cfg, dev, d.rank, d.world) if opts.validate else []
    log(f"twin {cfg.name} ({run_id(cfg)}): {pr['params_total']:,} params (time mixer {pr['params_time_mixer']:,}), "
        f"{d.world} rank(s) on {dev}, {rows} rows per rank in {accum} micro-batch(es) of {micro}, "
        f"compile={opts.compile}, bf16={bf16}" + (f", packed GLA kernel {opts.packed_kernel}" if uses(cfg.kinds, "gla") else ""))

    def state_ck():
        return {"model": raw.state_dict(), "opt": opt.state_dict(), "state": dict(st), "guard": guard.state_dict(),
                "watch": watch.state_dict(), "config_json": config_json(cfg), "run_id": run_id(cfg),
                "params": pr}

    def save(name: str):
        if opts.save and out is not None and d.rank == 0:
            _atomic_save(state_ck(), out / name)
        d.barrier()

    def save_weights(path: Path, **extra):
        if opts.save and out is not None and d.rank == 0:
            _atomic_save({"model": raw.state_dict(), "config_json": config_json(cfg), "run_id": run_id(cfg),
                          "step": st["step"], "points": st["points"], "params": pr, **extra}, path)
        d.barrier()

    def rollback(reason: str):
        bp = out / "best.pt" if out is not None else None
        if bp is None or not bp.exists():
            log(f"  !! ROLLBACK wanted ({reason}) but there is no best.pt yet -- continuing")
            return
        ck = torch.load(bp, map_location=dev, weights_only=False)
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        st["peak_lr"] *= 0.5
        st["rollbacks"] += 1
        watch.bad = 0
        log(f"  !! ROLLBACK #{st['rollbacks']} ({reason}): weights and optimizer from best.pt "
            f"(step {ck['state']['step']:,}, holdout {ck['state']['best_holdout']:.4f}); peak LR halved to "
            f"{st['peak_lr']:.2e}; training continues at {human(st['points'])} points on new data")

    def validate(tag: str) -> dict:
        m = evaluate(raw, holdout, d, bf16)
        bands = "  ".join(f"{k}: {v:.4f}" for k, v in m.items() if k.startswith("ctx"))
        log(f"  [{tag}] holdout loss = {m['holdout']:.4f}   |  {bands}")
        log(f"  [{tag}] one-patch loss {m['one_patch']:.4f}  |  multi-patch loss {m['multi_patch']:.4f}")
        if opts.probes and d.rank == 0:
            from tidemark.pretrain.probes import all_probes
            p = all_probes(raw, dev)
            raw.train()
            for line in p["lines"]:
                log(f"  [{tag}] {line}")
            m.update(p["values"])
        d.barrier()
        st["history"].append({"step": st["step"], "points": st["points"], **m})
        return m

    def summary(exit_code: int, **extra) -> dict:
        s = {"exit": exit_code, "run": cfg.name, "run_id": run_id(cfg), "arm": cfg.arm, "seed": cfg.seed,
             "lr": cfg.lr, "peak_lr_now": st["peak_lr"], "step": st["step"], "points": st["points"],
             "rollbacks": st["rollbacks"], "skipped": st["n_skip"], "best_holdout": st["best_holdout"],
             "params": pr, "world": d.world, "micro_rows": micro, "compile": opts.compile,
             "packed_kernel": opts.packed_kernel if uses(cfg.kinds, "gla") else None,
             "checksum": float(sum(p.detach().double().sum() for p in raw.parameters())), **opts.meta, **extra}
        if bench["t"] is not None:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            s["bench_points_per_s"] = (st["points"] - bench["pts"]) / max(time.time() - bench["t"], 1e-9)
            s["bench_steps"] = st["step"] - bench["step"]
        if dev.type == "cuda":
            s["peak_gb"] = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        return s

    raw.train()
    bench = {"t": None, "pts": 0.0, "step": 0}
    t_save = time.time()
    win = dict(loss=0.0, n=0, pts=0.0, t=time.time(), gn=0.0, gmax=0.0)
    ms_todo = [m for m in tc.milestones if m not in st["milestones_done"]]
    while st["points"] < tc.target_points:
        if opts.max_steps is not None and st["step"] - start_step >= opts.max_steps:
            return summary(EXIT_DONE, stopped="max_steps")
        if opts.pause_points is not None and st["points"] >= opts.pause_points:
            m = validate(f"pause @ {human(st['points'])}") if opts.validate else {}
            save("last.pt")
            res = summary(EXIT_DONE, paused=True, holdout_at_pause=m.get("holdout"), metrics=m)
            if d.rank == 0 and out is not None and opts.save:
                (out / "pilot.json").write_text(json.dumps(res, indent=1, default=float))
            return res
        lr = S.sched_lr(st["peak_lr"], tc.warmup_pct, tc.target_points, st["points"])
        for g in opt.param_groups:
            g["lr"] = lr
        t0 = time.time()
        b = stream.batch(st["step"])
        if opts.batch_hook is not None:
            b = opts.batch_hook(b, d.rank, st["step"])
        torch.manual_seed(_dropout_seed(cfg.seed, d.rank, st["step"]))   # model RNG: resumes exactly
        # the loss is the mean over every loss-carrying series of the GLOBAL batch (as one
        # 512-row batch on one GPU): DDP averages gradients over ranks, hence x world
        n_glob = d.all_sum(torch.tensor([float(loss_series(b, cfg.model.patch_len))], dtype=torch.float64,
                                        device=dev))
        scale = d.world / max(1.0, float(n_glob))
        loss_sum = torch.zeros((), device=dev)
        loss_n = torch.zeros((), device=dev)
        for k in range(accum):
            mb = slice_batch(b, M, k, accum)
            sync = contextlib.nullcontext() if (not d.on or k == accum - 1) else model.no_sync()
            with sync:
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=bf16):
                    o = packed_forward(fwd, mb)
                (o["loss_sum"] * scale).backward()
            loss_sum += o["loss_sum"].detach().float()
            loss_n += o["loss_n"].detach().float()
        gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), tc.grad_clip)
        save_req = float(d.rank == 0 and time.time() - t_save >= opts.save_every_s)
        stop_req = float(stop_requested(out))          # any rank may ask (a SIGTERM may reach only some); the all-reduce below shares it
        stats = torch.tensor([float(loss_sum), float(loss_n), float(b["points"]), float(gn), save_req, stop_req],
                             dtype=torch.float64, device=dev)
        d.all_sum(stats)
        st["step"] += 1
        step = st["step"]
        g_loss = float(stats[0] / stats[1]) if stats[1] > 0 else float("nan")
        g_gn = float(stats[3]) / d.world
        if guard.is_spike(g_loss, g_gn, step):
            opt.zero_grad(set_to_none=True)
            st["n_skip"] += 1
            log(guard.last_note)
            if guard.consecutive >= tc.spike_rollback_after:
                rollback(f"{guard.consecutive} spiky batches in a row")
                guard.consecutive = 0
        else:
            opt.step()
            opt.zero_grad(set_to_none=True)
            st["points"] += float(stats[2])
            win["loss"] += g_loss
            win["n"] += 1
            win["pts"] += float(stats[2])
            win["gn"] += g_gn
            win["gmax"] = max(win["gmax"], g_gn)
        st["train_s"] += time.time() - t0
        if opts.bench_warm is not None and step - start_step == opts.bench_warm:
            if dev.type == "cuda":
                torch.cuda.synchronize(dev)
            bench.update(t=time.time(), pts=st["points"], step=step)

        if step % tc.print_every == 0 and win["n"]:
            el = time.time() - win["t"]
            pps = win["pts"] / max(el, 1e-9)
            avg = win["loss"] / win["n"]
            eta = (tc.target_points - st["points"]) / max(pps, 1e-9) / 3600
            log(f"step {step:,}  loss={avg:.4f}  lr={lr:.2e}  {human(st['points'])}/{human(tc.target_points)} "
                f"({st['points'] / tc.target_points:5.1%})  ~{pps / 1e6:.2f}M pts/s ({pps / d.world / 1e6:.2f}M per GPU)  "
                f"gen={stream.gen_seconds / max(st['train_s'], 1e-9):4.1%}  ETA {eta:.2f}h  "
                f"gnorm={win['gn'] / win['n']:.2f} (max {win['gmax']:.2f})  skipped={st['n_skip']}")
            if step - start_step == tc.print_every and dev.type == "cuda":
                log(f"  peak GPU memory so far: {torch.cuda.max_memory_allocated(dev) / 2 ** 30:.1f} GB of "
                    f"{torch.cuda.get_device_properties(dev).total_memory / 2 ** 30:.0f} GB")
            if watch.update(avg, step, start_step):
                rollback(f"training loss {avg:.4f} vs best window {watch.best:.4f}")
            win = dict(loss=0.0, n=0, pts=0.0, t=time.time(), gn=0.0, gmax=0.0)

        if opts.validate and step % tc.val_every == 0:
            m = validate(f"val @ {step:,}")
            if m["holdout"] < st["best_holdout"]:
                st["best_holdout"] = m["holdout"]
                save("best.pt")
                log(f"  new best holdout {m['holdout']:.4f} -> best.pt")
            elif S.holdout_rollback(m["holdout"], st["best_holdout"], tc.val_rollback_x):
                rollback(f"holdout {m['holdout']:.4f} vs best {st['best_holdout']:.4f}")
            if S.too_many_rollbacks(st["rollbacks"], tc.max_rollbacks):
                log(f"\n!! {st['rollbacks']} rollbacks: the run is not stable. Stopping. best.pt is the model "
                    "to use; paste the log.")
                save("last.pt")
                res = summary(EXIT_UNSTABLE, unstable=True, holdout_at_pause=None)
                if opts.pause_points is not None and d.rank == 0 and out is not None and opts.save:
                    (out / "pilot.json").write_text(json.dumps(res, indent=1, default=float))
                return res

        for msp in list(ms_todo):
            if st["points"] >= msp:
                st["milestones_done"].append(msp)
                ms_todo.remove(msp)
                save_weights(out / "milestones" / f"ckpt_{int(round(msp / 1e9))}B.pt" if out else Path("."),
                             lr_at_save=lr, note=f"milestone at {human(msp)} points")
                log(f"  MILESTONE saved: ckpt_{int(round(msp / 1e9))}B.pt (step {step:,})")
        if stats[5] > 0:
            save("last.pt")
            log(f"STOP requested (time limit): last.pt saved at step {step:,}, {human(st['points'])} points; "
                "exit for requeue")
            return summary(EXIT_REQUEUE)
        if stats[4] > 0:
            save("last.pt")
            t_save = time.time()

    m = validate("FINAL") if opts.validate else {}
    save_weights(out / "final.pt" if out else Path("."), final_metrics=m)
    res = summary(EXIT_DONE, finished=True, final=m)
    if d.rank == 0 and out is not None and opts.save:
        (out / "summary.json").write_text(json.dumps({**res, "history": st["history"]}, indent=1, default=float))
        if last is not None:
            last.unlink(missing_ok=True)            # final.pt and best.pt stay
    d.barrier()
    log(f"\nDone: {cfg.name}. {st['step']:,} steps, {human(st['points'])} points -> {out / 'final.pt' if out else '-'}")
    return res


# ================================== CLI =====================================

def pilot_rate(root, arm: str, size: str = "30m") -> float | None:
    """The arm's rate from checks/pilot_choice.json (Stage B: pilot_choice_<size>.json); None if the arm is blocked."""
    p = checks_dir(root) / ("pilot_choice.json" if size == "30m" else f"pilot_choice_{size}.json")
    if not p.exists():
        raise SystemExit(f"{p} is missing: the pilot has not finished (python -m tidemark.pretrain.pilot choose)")
    ch = json.loads(p.read_text())["arms"][arm]
    if ch["status"] == "pending":
        raise SystemExit(f"{p}: the pilot of {arm} is not finished (status pending: {ch.get('why')})")
    return float(ch["lr"]) if ch["status"] == "chosen" else None


def twin_rate(root, arm: str, size: str = "30m") -> float | None:
    """Rate of a twin from the pilot choice. The 30M gdn arm uses its OWN pilot entry (arms.gdn) once pilot_b has written
    it; before that it takes GLA's rate, as in Stage A's first version. Every other arm: unchanged."""
    if size != "30m":
        return pilot_rate(root, STAGE_B_DONOR.get(arm, arm), size)
    p = checks_dir(root) / "pilot_choice.json"
    own = arm == "gdn" and p.exists() and "gdn" in json.loads(p.read_text()).get("arms", {})
    return pilot_rate(root, arm if own else lr_donor(arm), size)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True, help="run root, e.g. $SCRATCH/tidemark")
    ap.add_argument("--arm", type=canonical, choices=ALL_ARMS, required=True,
                    help="Stage A names gla_s1 ... are accepted for glas1 ...")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--size", choices=SIZES, default="30m", help="Stage B model size (default: the 30M twin)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--lr", type=float)
    g.add_argument("--lr-from", choices=("pilot",), help="the rate chosen by the pilot (pilot_choice.json; --size other than 30m: pilot_choice_<size>.json); "
                        "Stage A / B arms without a pilot take their donor arm's rate (arms.LR_DONOR / STAGE_B_DONOR)")
    ap.add_argument("--rerun", action="store_true", help="a rerun at another start rate (needs --lr): output goes to "
                    "<root>/runs_<size>_rerun/, summary.json and rerun.json say which original run it corresponds to; "
                    "everything else (config, budget, data order, rollbacks, resume) is the same code path")
    ap.add_argument("--pause-points", type=float, default=None, help=f"pilot: pause here (e.g. {PILOT_POINTS:g})")
    ap.add_argument("--micro-rows", type=int, default=None, help="rows per micro-batch (default: bench_rows.json)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    if os.environ.get("TIDEMARK_SIGTERM_STOP") == "1":
        install_sigterm_stop()
    d = Dist(a.device)
    lr = a.lr
    if a.lr_from:
        lr = twin_rate(a.root, a.arm, a.size)
        if lr is None:
            if d.rank == 0:
                print(f"ARM {a.arm} IS BLOCKED in pilot_choice.json (both pilot rates had a rollback): "
                      "its twins do not start.", flush=True)
            d.close()
            return EXIT_BLOCKED
    if a.rerun and a.lr is None:
        raise SystemExit("--rerun needs an explicit --lr")
    cfg = sized_config(a.arm, a.seed, lr, a.size)
    kernel = gla_kernel_choice(a.root) if uses(cfg.kinds, "gla") else "chunk"
    opts = Options(micro_rows=a.micro_rows or micro_rows_choice(a.root, a.size), compile=not a.no_compile,
                   packed_kernel=kernel, pause_points=a.pause_points, log=lambda *x: print(*x, flush=True),
                   meta=rerun_meta(a.root, cfg) if a.rerun else None)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        res = train_twin(cfg, opts, run_dir(a.root, cfg, a.rerun), d)
    finally:
        d.close()
    return int(res["exit"])


if __name__ == "__main__":
    sys.exit(main())
