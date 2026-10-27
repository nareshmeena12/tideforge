"""Throughput of the data loader on its own (no model, no training), per GPU.

  python -m tidemark.real.bench_loader [--real-root R] [--shards-dir S] [--plan P] [--ckpt C] [--rows 288] [--real-fraction 0.5]
                                       [--steps 30] [--warm 5] [--compare-legacy] [--sweep] [--sim-step-ms 0]
  torchrun --standalone --nproc_per_node=4 -m tidemark.real.bench_loader --sweep       # the 4 ranks of a node read at the same time

Default: points/s (VALID points) of the real part alone, the synthetic part alone and the mixed batch.
--compare-legacy: the real part with the first loader's reads (np.memmap, a binary search on the memory-mapped missing positions, one thread, no
  prefetch) against the configured one (preadv on persistent descriptors with random-access advice, missing positions in RAM, I/O threads, prefetch),
  on the same data. Run it on a pool that does not fit in the page cache (the full pool); on a small pool the first loader is served from RAM.
--sweep: the real part for workers per rank in 1, 4, 8, 12, 16, 24 with 1 or 2 threads per worker (a combination above 28 threads is skipped: at
  least 4 of a rank's 32 threads stay free for the training process). Every row: real-part points/s per GPU, seconds per batch, and where the time
  goes (plan = random draws and index look-ups; wait = the main thread blocked on the reads; build = group builder and placement).
--sim-step-ms N: sleep N ms after every batch, as a stand-in for the GPU step, so a prefetching loader can show how much of its reads it hides
  ("wait" then is the I/O time that is NOT hidden). With 0 the loop measures the loader's own throughput.
Under torchrun every rank runs the same sweep in step (a barrier before each row) and rank 0 prints the mean over ranks.
The owner's estimate is ~13.7M points/s per GPU for a whole node doing 55M points/s; the real part needs about as many points/s as it gets rows.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from tidemark.real import ft3_core as C
from tidemark.real import runcfg as RC
from tidemark.real import sampler as SM
from tidemark.real.store import RealStore

WORKERS = (1, 4, 8, 12, 16, 24)
THREADS_PER_WORKER = (1, 2)


def timed(fn, n: int, warm: int, device, sim_step_s: float = 0.0):
    """(valid points per second, seconds per batch, points per batch); sim_step_s is slept after every batch and not counted."""
    for _ in range(warm):
        fn()
        if sim_step_s:
            time.sleep(sim_step_s)
    if device.type == "cuda":
        torch.cuda.synchronize()
    busy, pts = 0.0, 0
    for _ in range(n):
        t0 = time.perf_counter()
        b = fn()
        if device.type == "cuda":
            torch.cuda.synchronize()
        busy += time.perf_counter() - t0
        pts += b["points"]
        if sim_step_s:
            time.sleep(sim_step_s)
    return pts / busy, busy / n, pts / n


def real_only(cfg, dev, store, targets, mix, rank, seed):
    rows = mix.split_rows(int(cfg.get("group_lanes", 8)))[1]
    return SM.RealBatcher(cfg, dev, store, targets, mix, rows=rows, seed=seed, stream=rank)


def measure_real(cfg, dev, store, targets, mix, a, rank, seed) -> dict:
    rb = real_only(cfg, dev, store, targets, mix, rank, seed)
    pps, dt, pb = timed(rb.next_batch, a.steps, a.warm, dev, a.sim_step_ms / 1000)
    n = max(rb.timers["batches"], 1)
    out = {"pps": pps, "s": dt, "pts": pb, "plan": rb.timers["plan"] / n, "wait": rb.timers["wait"] / n, "build": rb.timers["build"] / n,
           "rej": rb.rejected, "forced": rb.forced}
    rb.close()
    return out


def mean_over_ranks(vals: dict, dev) -> dict:
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return vals
    keys = sorted(vals)
    t = torch.tensor([vals[k] for k in keys], dtype=torch.float64, device=dev)
    torch.distributed.all_reduce(t)
    return {k: float(t[i]) / torch.distributed.get_world_size() for i, k in enumerate(keys)}


def barrier():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--real-root", default=None)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--shards-dir", default=None)
    ap.add_argument("--rows", type=int, default=None)
    ap.add_argument("--real-fraction", type=float, default=None)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--compare-legacy", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sim-step-ms", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=None, help="fetch_workers for the default and compare runs")
    ap.add_argument("--threads-per-worker", type=int, default=None)
    a = ap.parse_args(argv)
    rc = RC.RunConfig().resolve()
    if a.real_root:
        rc.real_root, rc.plan = a.real_root, str(Path(a.real_root) / "plan.json")
    if a.plan:
        rc.plan = a.plan
    if a.ckpt:
        rc.ckpt = a.ckpt
    if a.shards_dir:
        rc.shards_dir = a.shards_dir
    mix = rc.mix
    for k, v in (("rows", a.rows), ("real_fraction", a.real_fraction), ("fetch_workers", a.workers), ("threads_per_worker", a.threads_per_worker)):
        if v is not None:
            mix = replace(mix, **{k: v})
    rc.mix = mix
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl" if torch.cuda.is_available() and a.device != "cpu" else "gloo")
    dev = torch.device(a.device or (f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    say = print if rank == 0 else (lambda *x, **k: None)
    cfg = RC.model_cfg(RC.load_base_cfg(rc.ckpt), rc)
    targets = RC.plan_targets(RC.load_plan(rc.plan))
    store = RealStore(Path(rc.real_root) / rc.shards_dir, sorted(targets))
    C.set_group_options(cfg)
    torch.manual_seed(rc.seed + rank)
    say(f"device {dev}; world {world}; rows {mix.rows} = {mix.split_rows(int(cfg.get('group_lanes', 8)))[0]} synthetic + "
        f"{mix.split_rows(int(cfg.get('group_lanes', 8)))[1]} real; {len(targets)} real datasets, "
        f"{sum(store.stored_points().values()) / 1e9:.2f}B stored points; I/O threads per rank {mix.fetch_threads} "
        f"({mix.fetch_workers} workers x {mix.threads_per_worker}), io {mix.io}, prefetch {mix.prefetch}")

    if a.sweep:
        say(f"\nREAL PART, workers per rank sweep ({a.steps} batches after {a.warm} warm-up, simulated step {a.sim_step_ms:g} ms, "
            f"{world} rank(s) in step; at most {mix.max_fetch_threads} I/O threads per rank)")
        say(f"{'workers':>7s} {'thr/wkr':>7s} {'threads':>7s} {'real pts/s/GPU':>16s} {'s/batch':>8s} {'plan':>7s} {'wait':>7s} {'build':>7s}")
        best = None
        for w in WORKERS:
            for t in THREADS_PER_WORKER:
                if w * t > mix.max_fetch_threads:
                    say(f"{w:>7d} {t:>7d} {w * t:>7d} {'skipped: above ' + str(mix.max_fetch_threads) + ' threads':>40s}")
                    continue
                barrier()
                m = measure_real(cfg, dev, store, targets, replace(mix, fetch_workers=w, threads_per_worker=t, io="pread", prefetch=True),
                                 a, rank, rc.seed + 100 * w + t)
                m = mean_over_ranks(m, dev)
                say(f"{w:>7d} {t:>7d} {w * t:>7d} {m['pps']:>16,.0f} {m['s']:>8.3f} {m['plan']:>7.3f} {m['wait']:>7.3f} {m['build']:>7.3f}")
                if best is None or m["pps"] > best[0]:
                    best = (m["pps"], w, t)
        say(f"\nfastest: {best[1]} workers x {best[2]} thread(s) = {best[0] / 1e6:.2f}M real points/s per GPU "
            f"(set MixConfig.fetch_workers / threads_per_worker; a smaller configuration within a few percent of it leaves more CPU to training)")
        return 0

    if a.compare_legacy:
        say("\nREAL PART, first loader against the configured one (same data, same number of batches)")
        say(f"{'loader':44s} {'real pts/s/GPU':>15s} {'s/batch':>8s} {'plan':>7s} {'wait':>7s} {'build':>7s}")
        for label, mx in (("first loader (mmap, 1 thread, no prefetch)", replace(mix, io="mmap", fetch_workers=0, prefetch=False)),
                          (f"new ({mix.fetch_workers}x{mix.threads_per_worker} threads, pread, prefetch)", replace(mix, io="pread"))):
            barrier()
            m = mean_over_ranks(measure_real(cfg, dev, store, targets, mx, a, rank, rc.seed + 7), dev)
            say(f"{label:44s} {m['pps']:>15,.0f} {m['s']:>8.3f} {m['plan']:>7.3f} {m['wait']:>7.3f} {m['build']:>7.3f}")
        return 0

    mb = SM.MixedBatcher(cfg, dev, store, targets, mix, seed=rc.seed, rank=rank)
    res = {}
    res["real"] = timed(mb.real.next_batch, a.steps, a.warm, dev, a.sim_step_ms / 1000)
    res["synthetic"] = timed(mb.syn.next_batch, a.steps, a.warm, dev, a.sim_step_ms / 1000)
    res["mixed"] = timed(mb.next_batch, a.steps, a.warm, dev, a.sim_step_ms / 1000)
    say(f"{'part':10s} {'points/s':>14s} {'s / batch':>10s} {'points / batch':>15s}")
    for k, (pps, dt, pb) in res.items():
        say(f"{k:10s} {pps:>14,.0f} {dt:>10.3f} {pb:>15,.0f}")
    pps, dt, _ = res["mixed"]
    n = max(mb.real.timers["batches"], 1)
    say(f"\nmixed batch: {pps / 1e6:.1f}M points/s per GPU; a training step has to run longer than {dt:.2f} s for the loader to stay hidden behind it")
    say(f"real part, time per batch: plan {mb.real.timers['plan'] / n:.3f} s, wait for reads {mb.real.timers['wait'] / n:.3f} s, "
        f"build {mb.real.timers['build'] / n:.3f} s")
    r = mb.real.realised()
    say(f"\n{'dataset':46s} {'target':>7s} {'realised':>9s} {'missing':>8s}")
    for nme, v in sorted(r.items(), key=lambda kv: -kv[1]["target"]):
        say(f"{nme:46s} {v['target']:>7.1%} {v['realised']:>9.1%} {v['missing_share']:>8.1%}")
    say(f"windows redrawn by the horizon rule: {mb.real.rejected}, accepted after {mb.real.mix.tries} tries anyway: {mb.real.forced}, "
        f"cells with no planned dataset long enough: {mb.real.no_dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
