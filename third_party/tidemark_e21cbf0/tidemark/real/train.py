"""Real-data post-training of ft3_main: DDP on the GPUs of one node, mixed real + synthetic batches, exact resume.

  torchrun --standalone --nproc_per_node=4 -m tidemark.real.train [--smoke] [--name N] [--out-dir D] [--copy-to D2]
                                                                   [--rows 288] [--real-fraction 0.3333] [--target-points 30e9]
  (scripts/perlmutter/real_train_interactive.sh builds the line; real_*.sbatch is the batch version)

What a step is: every rank builds its OWN batch (288 rows: 96 real + 192 synthetic, config values), forward in bf16 through the
loss-masked ft3 decoder (tidemark/real/model.py), backward (DDP averages the gradients), clip 1.0, AdamW. The LR is a function of the
GLOBAL valid points seen (all-reduced every step): 5% linear warmup to 1.2e-4, cosine to 10% of the peak over the whole budget.
The next batch is built while the GPU is still running the backward pass.

Exact resume: last.pt holds model, optimizer, step, points and, for EVERY rank, the CPU / CUDA RNG states and the sampler state
captured just before the next batch was generated (synthetic pool numbers and pointers, real sampler RNG, share controller). A resumed
run regenerates that batch and carries on as if it had never stopped (same world size required). last.pt is written every
`save_every_s`, at every milestone and at a stop (STOP file in the run dir, SIGTERM, --max-hours); the process then exits with code 3.
Milestones (10B, 20B, 30B total points) are also written as weights-only files in ft3_main.pt's format (ckpt_10B.pt, ...) that the
scorer loads, and copied to --copy-to when given.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from tidemark.pretrain.train import Dist, _SIGTERM, install_sigterm_stop, stop_requested
from tidemark.real import ft3_core as C
from tidemark.real import runcfg as RC
from tidemark.real import sampler as SM
from tidemark.real.model import build_masked_model, packed_forward_masked
from tidemark.real.store import RealStore

EXIT_DONE, EXIT_REQUEUE = 0, 3


def human(n: float) -> str:
    return C.human(n)


def lr_at(points: float, rc: RC.RunConfig) -> float:
    """5% linear warmup (from peak x warmup_start_frac), then cosine to final_lr_frac x peak at the end of the budget."""
    f = min(max(points / rc.target_points, 0.0), 1.0)
    w = rc.warmup_frac
    if f < w:
        return rc.peak_lr * (rc.warmup_start_frac + (1 - rc.warmup_start_frac) * f / w)
    prog = (f - w) / (1 - w)
    return rc.peak_lr * (rc.final_lr_frac + (1 - rc.final_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def milestone_name(m: float) -> str:
    return f"ckpt_{m / 1e9:g}B.pt"                                        # 10e9 -> ckpt_10B.pt


IO_ONLY = ("io", "fetch_workers", "threads_per_worker", "prefetch", "max_fetch_threads")        # how windows are read: changes no data


def config_mismatch(saved: dict, now: dict) -> dict:
    """Fields that must be the same to resume exactly. The loader's I/O settings may differ (they change speed, not data), and fields a
    checkpoint of an older version does not have are not compared."""
    bad = {}
    for k in ("target_points", "peak_lr", "milestones", "seed", "ckpt", "shards_dir"):
        if k in saved and saved[k] != now[k]:
            bad[k] = (saved[k], now[k])
    sm, nm = saved.get("mix", {}), now["mix"]
    for k in nm:
        if k not in IO_ONLY and k in sm and sm[k] != nm[k]:
            bad[f"mix.{k}"] = (sm[k], nm[k])
    return bad


def capture(mixed: SM.MixedBatcher, device) -> dict:
    return {"sampler": mixed.state_dict(), "cpu": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None}


def restore(state: dict, mixed: SM.MixedBatcher, device) -> None:
    mixed.load_state_dict(state["sampler"])
    torch.set_rng_state(state["cpu"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def slim_checkpoint(raw, cfg: dict, rc: RC.RunConfig, step: int, points: float, note: str) -> dict:
    """Weights-only, in the format of ft3_main.pt (the scorer and load_packed_checkpoint read model + cfg)."""
    return dict(model=raw.state_dict(), cfg=cfg, step=step, points_seen=points, packed=True, run_name=rc.name,
                base_ckpt=os.path.basename(rc.ckpt), note=note, real_run=rc.to_json())


def run(rc: RC.RunConfig, dist_: Dist, max_steps: int | None = None, log=print) -> int:
    d, device = dist_, dist_.device
    rank0 = d.rank == 0
    out = Path(rc.out_dir)
    if rank0:
        out.mkdir(parents=True, exist_ok=True)
    d.barrier()
    say = (lambda *a, **k: log(*a, **k)) if rank0 else (lambda *a, **k: None)

    # ---- config, data, model
    base = RC.load_base_cfg(rc.ckpt)
    cfg = RC.model_cfg(base, rc)
    plan = RC.load_plan(rc.plan)
    targets = RC.plan_targets(plan)
    store = RealStore(Path(rc.real_root) / rc.shards_dir, sorted(targets))
    C.set_group_options(None)                                               # the holdout always uses the v5 group mix ...
    val_batches = C.build_val_batches(cfg, device, rc.val_batches)
    C.set_group_options(cfg)                                                # ... training uses this run's (ft3_main's)
    torch.manual_seed(rc.seed + 1000 * d.rank)
    mixed = SM.MixedBatcher(cfg, device, store, targets, rc.mix, seed=rc.seed, rank=d.rank)

    raw = build_masked_model(cfg).to(device)
    ck0 = torch.load(rc.ckpt, map_location=device, weights_only=False)
    raw.load_state_dict(ck0["model"], strict=True)                          # every tensor must match; the optimizer starts fresh
    base_points = float(ck0.get("points_seen", 0.0))
    del ck0
    n_par = sum(p.numel() for p in raw.parameters())
    amp_on = device.type == "cuda"
    amp_dtype = torch.bfloat16
    ddp = torch.nn.parallel.DistributedDataParallel(raw, device_ids=[d.local] if device.type == "cuda" else None,
                                                    gradient_as_bucket_view=True) if d.on else raw
    fwd = torch.compile(ddp) if rc.compile and device.type == "cuda" else ddp
    opt = torch.optim.AdamW(raw.parameters(), lr=rc.peak_lr * rc.warmup_start_frac, weight_decay=rc.weight_decay,
                            fused=device.type == "cuda")

    # ---- resume
    st = dict(step=0, points=0.0, points_syn=0.0, points_real=0.0, n_bad=0, next_ms=0, train_s=0.0, last_save=time.time())
    ms = [m for m in rc.milestones]
    last = out / "last.pt"
    pre = None
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)           # CPU: RNG and sampler states must stay CPU tensors
        if ck["world"] != d.world:
            raise SystemExit(f"{last} was written with {ck['world']} ranks; this run has {d.world} (the data stream depends on it)")
        mismatch = config_mismatch(ck["run"], rc.to_json())
        if mismatch:
            raise SystemExit(f"{last}: the run configuration changed since it was written: {mismatch}")
        raw.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        st.update(ck["state"])
        pre = ck["rank_states"][d.rank]
        restore(pre, mixed, device)
        say(f"RESUMED from {last}: step {st['step']:,}, {human(st['points'])} of {human(rc.target_points)} points")
        del ck
    else:
        say(f"fresh start from {rc.ckpt} ({human(base_points)} points of base training), strict load OK")
    st["next_ms"] = sum(1 for m in ms if st["points"] >= m)                 # index of the next milestone

    say(f"params {n_par / 1e6:.2f}M | world {d.world} x {rc.mix.rows} rows ({mixed.rows_syn} synthetic + {mixed.rows_real} real) | "
        f"target {human(rc.target_points)} points, milestones {[human(m) for m in ms]} | peak lr {rc.peak_lr:g}, warmup {rc.warmup_frac:.0%}, "
        f"cosine to {rc.final_lr_frac:.0%} | real datasets {len(targets)} | compile {rc.compile and device.type == 'cuda'}")
    if rank0:
        (out / "run_config.json").write_text(json.dumps(rc.to_json(), indent=1))

    def do_val(tag: str) -> dict:
        m = {}
        if rank0:
            vl, bands = C.evaluate_packed(raw, val_batches, amp_dtype, amp_on, device)
            say(f"  [{tag}] synthetic holdout loss = {vl:.4f}  |  " + "  ".join(f"ctx {k}: {v:.4f}" for k, v in bands.items()), flush=True)
            l_one, l_multi = C.horizon_split(raw, val_batches, amp_dtype, amp_on, device)
            g_on, g_cut = C.group_gain(raw, val_batches, amp_dtype, amp_on, device)
            say(f"  [{tag}] one-patch {l_one:.4f} multi-patch {l_multi:.4f} | grouped series ON {g_on:.4f} CUT {g_cut:.4f}", flush=True)
            m = dict(holdout=vl, one_patch=l_one, multi_patch=l_multi, group_on=g_on, group_cut=g_cut, **{f"ctx {k}": v for k, v in bands.items()})
            raw.train()
        d.barrier()
        return m

    def save_last(pre_state: dict) -> None:
        states = [None] * d.world
        if d.on:
            dist.all_gather_object(states, pre_state)
        else:
            states = [pre_state]
        if rank0:
            ck = dict(model=raw.state_dict(), opt=opt.state_dict(), state=dict(st), world=d.world, run=rc.to_json(),
                      rank_states=states, cfg=cfg)
            atomic_save(ck, last)
        st["last_save"] = time.time()
        d.barrier()

    def milestone(m: float, final: bool = False) -> None:
        if rank0:
            name = milestone_name(m)
            atomic_save(slim_checkpoint(raw, cfg, rc, st["step"], st["points"], f"{human(st['points'])} points of real-data post-training"), out / name)
            say(f"  MILESTONE {name} written (step {st['step']:,}, {human(st['points'])} points)", flush=True)
            if rc.copy_to:
                try:
                    Path(rc.copy_to).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(out / name, Path(rc.copy_to) / name)
                    say(f"  copied to {Path(rc.copy_to) / name}", flush=True)
                except Exception as e:  # noqa: BLE001
                    say(f"  !! COPY FAILED ({type(e).__name__}: {e}); the file is still in {out}", flush=True)
        d.barrier()

    # ---- the loop
    raw.train()
    t_start = time.time()
    win = dict(t=time.time(), pts=0.0, steps=0, loss=torch.zeros(3, device=device), nb=0)
    peak_logged = False
    if pre is None:
        pre = capture(mixed, device)
    b = mixed.next_batch()
    first_step = st["step"]
    while st["points"] < rc.target_points and (max_steps is None or st["step"] - first_step < max_steps):
        lr = lr_at(st["points"], rc)
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_on):
            o = packed_forward_masked(fwd, b)
        loss = o["loss"]
        loss.backward()                                                     # DDP all-reduce runs here, asynchronously
        pre_next = capture(mixed, device)
        b_next = mixed.next_batch()                                         # overlaps the GPU's backward pass
        s1 = b["parts"][0]
        with torch.no_grad():
            ps = o["per_series_loss"]
            win["loss"] += torch.nan_to_num(torch.stack([loss.detach().float(), torch.nanmean(ps[:s1]).float(), torch.nanmean(ps[s1:]).float()]), nan=0.0)
        gn = torch.nn.utils.clip_grad_norm_(raw.parameters(), rc.grad_clip)
        finite = bool(torch.isfinite(gn))                                   # the gradients are identical on all ranks after DDP
        if finite:
            opt.step()
        else:
            st["n_bad"] += 1
        opt.zero_grad(set_to_none=True)
        # global bookkeeping: points of all ranks, and the stop / timed-save requests (one small all-reduce)
        want_stop = float(stop_requested(out) or (rc.max_hours is not None and (time.time() - t_start) / 3600 >= rc.max_hours))
        want_save = float(rank0 and time.time() - st["last_save"] >= rc.save_every_s)
        vec = torch.tensor([b["points_syn"], b["points_real"], want_stop, want_save], dtype=torch.float64, device=device)
        vec = d.all_sum(vec).tolist()
        st["step"] += 1
        st["points_syn"] += vec[0]
        st["points_real"] += vec[1]
        st["points"] += vec[0] + vec[1]
        win["pts"] += b["points"]
        win["steps"] += 1
        b, pre = b_next, pre_next

        if st["step"] % rc.print_every == 0:
            el = time.time() - win["t"]
            lv = (d.all_sum(win["loss"].clone()) / (win["steps"] * d.world)).tolist()
            cnt = torch.tensor(mixed.real.cnt_valid, dtype=torch.float64, device=device)
            cnt = d.all_sum(cnt).cpu().numpy()
            real_tot = max(cnt.sum(), 1.0)
            pps = win["pts"] / max(el, 1e-9)
            say(f"step {st['step']:,} loss {lv[0]:.4f} (syn {lv[1]:.4f} real {lv[2]:.4f}) lr {lr:.2e} | {human(st['points'])}/{human(rc.target_points)} "
                f"({st['points'] / rc.target_points:5.1%}) real {st['points_real'] / max(st['points'], 1):.1%} | {pps / 1e6:.2f}M pts/s/GPU "
                f"({pps * d.world / 1e6:.1f}M/node) | bad {st['n_bad']} rej {mixed.real.rejected} forced {mixed.real.forced}", flush=True)
            if not peak_logged and device.type == "cuda":
                say(f"  peak GPU memory so far {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
                peak_logged = True
            if st["step"] % (rc.print_every * 10) == 0 or st["step"] == first_step + rc.print_every:
                top = sorted(zip(mixed.real.names, cnt / real_tot, mixed.real.targets), key=lambda t: -t[2])
                say("  real shares (realised/target): " + ", ".join(f"{n} {r:.1%}/{t:.1%}" for n, r, t in top if t > 0.005), flush=True)
            win.update(t=time.time(), pts=0.0, steps=0, loss=torch.zeros(3, device=device))

        if rc.val_every and st["step"] % rc.val_every == 0:
            do_val(f"val @ {st['step']:,}")

        crossed = [m for i, m in enumerate(ms) if i >= st["next_ms"] and st["points"] >= m]
        for m in crossed:
            save_last(pre)
            milestone(m)
            st["next_ms"] += 1
        if vec[2] > 0:
            save_last(pre)
            say(f"STOP requested: last.pt saved at step {st['step']:,}, {human(st['points'])} points; exit for requeue", flush=True)
            return EXIT_REQUEUE
        if vec[3] > 0 and not crossed:
            save_last(pre)
            say(f"  last.pt saved @ step {st['step']:,} ({human(st['points'])})", flush=True)

    if st["points"] >= rc.target_points:
        do_val("FINAL")
        if rank0:
            atomic_save(slim_checkpoint(raw, cfg, rc, st["step"], st["points"], "final"), out / "final.pt")
            r = mixed.real.realised()
            (out / "realised_shares_rank0.json").write_text(json.dumps(r, indent=1))
        d.barrier()
        say(f"DONE: {st['step']:,} steps, {human(st['points'])} points ({human(st['points_real'])} real) -> {out / 'final.pt'}")
    return EXIT_DONE


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="real30b", choices=sorted(RC.RUNS), help="the configuration: real30b (30B, 1:2, shards) or real200b "
                    "(200B, 1:1, shards_full, a checkpoint every 20B)")
    ap.add_argument("--smoke", action="store_true", help="smoke run of --run: 1B points for real30b (about 0.33B real), 2B for real200b (1B real); "
                    "schedule over the smoke budget, milestones at half and at the end")
    ap.add_argument("--name", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--real-root", default=None)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--shards-dir", default=None, help="converted datasets under --real-root (default shards)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--copy-to", default=None, help="permanent storage (CFS / home) the milestone files are copied to")
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--real-fraction", type=float, default=None)
    ap.add_argument("--target-points", type=float, default=None)
    ap.add_argument("--milestones", default=None, help="comma-separated total points, e.g. 10e9,20e9,30e9")
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="stop after this many steps of THIS process (tests)")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    base = RC.RUNS[a.run]()
    rc = RC.smoke_config(base, RC.SMOKE_POINTS[a.run]) if a.smoke else base
    over = {k: v for k, v in dict(name=a.name, ckpt=a.ckpt, real_root=a.real_root, plan=a.plan, shards_dir=a.shards_dir, out_dir=a.out_dir, copy_to=a.copy_to,
                                  target_points=a.target_points, max_hours=a.max_hours).items() if v is not None}
    rc = replace(rc, **over)
    if a.milestones:
        rc = replace(rc, milestones=tuple(float(x) for x in a.milestones.split(",")))
    mix = rc.mix
    if a.rows:
        mix = replace(mix, rows=a.rows)
    if a.real_fraction:
        mix = replace(mix, real_fraction=a.real_fraction)
    rc = replace(rc, mix=mix, compile=not a.no_compile and rc.compile).resolve()
    if os.environ.get("TIDEMARK_SIGTERM_STOP") == "1":
        install_sigterm_stop()
    d = Dist(a.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        return run(rc, d, a.max_steps)
    finally:
        d.close()


if __name__ == "__main__":
    sys.exit(main())
