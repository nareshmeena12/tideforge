"""Batches for the real-data post-training: ft3's synthetic stream and real windows, mixed in every step on every rank.

  SyntheticBatcher  ft3_core.PackedBatcher with generation pools that are a pure function of (seed, stream, bucket, pool number), so a
                    resumed run regenerates the very pool it was in (state_dict / load_state_dict keep the pool number and pointer).
  RealBatcher       PackedBatcher whose cells hold real windows instead of generated series: dataset (by the plan, with a feedback
                    controller on the realised shares), then series (uniform among the series with at least c points), then window
                    (uniform start). Windows are EXACTLY the context length c of the layout cell (the usable part is never cropped
                    again). Interior gaps are forward-filled and observed; the originally missing positions go to `loss_ok` (False
                    there), so they are not scored. A fraction of the cells goes through ft3's group builder (_multivariatize).
  MixedBatcher      one real batcher (rows_real rows) + one synthetic (rows_syn rows) = one batch of fixed shape (rows_real + rows_syn,
                    W): rows tile the 8192 points exactly, so a 1:2 row split is a 1:2 split by points (up to missing points).

The batch is the dict PackedBatcher.next_batch returns plus `loss_ok` (R, W) bool, with the real part's slot ids offset by the synthetic
part's slot count (seg_patch, gid), and `parts` = (S_syn, S_real). `points` counts VALID points (originally observed) only.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
import torch

from tidemark.real import ft3_core as C


@dataclass
class MixConfig:
    rows: int = 288                    # rows per GPU per step (owner: 288 = 96 real + 192 synthetic)
    real_fraction: float = 1 / 3       # real share of the rows; rows tile exactly, so also of the points. CONFIG VALUE.
    real_group_frac: float = 0.3       # share of real cells that go through the group builder
    horizon_points: int = 128          # "horizon part" of a window: its last points
    max_missing_horizon: float = 0.5   # skip a window whose horizon part is all missing or more than this share missing
    tries: int = 20                    # resamples before the least-missing candidate is accepted (and counted)
    controller_rate: float = 0.02      # step of the share controller
    a_clip: float = 3.0                # bound of the controller's log-weights
    # ---- how the real windows are read (does not change WHICH windows are drawn or any value)
    io: str = "pread"                  # "pread": persistent fds + POSIX_FADV_RANDOM + preadv, missing positions in RAM; "mmap": the first loader
    fetch_workers: int = 8             # background I/O workers per rank (0 = read in the calling thread)
    threads_per_worker: int = 2        # I/O threads per worker (1-2)
    prefetch: bool = True              # plan and fetch batch k+1 while the GPU runs step k
    max_fetch_threads: int = 28        # of the 32 threads a rank owns, at least 4 stay free for the training process

    @property
    def fetch_threads(self) -> int:
        return self.fetch_workers * self.threads_per_worker if self.fetch_workers > 0 else 0

    def check(self) -> None:
        if self.fetch_threads > self.max_fetch_threads:
            raise ValueError(f"{self.fetch_workers} workers x {self.threads_per_worker} threads = {self.fetch_threads} I/O threads per rank; "
                             f"at most {self.max_fetch_threads} (4 of the rank's 32 threads stay free for training)")
        if self.threads_per_worker < 1:
            raise ValueError("threads_per_worker must be at least 1")

    def split_rows(self, V: int) -> tuple[int, int]:
        """(synthetic rows, real rows), both whole blocks of V rows."""
        blocks = self.rows // V
        if blocks * V != self.rows:
            raise ValueError(f"rows ({self.rows}) must be a multiple of the group lanes ({V})")
        real = int(round(blocks * self.real_fraction))
        return (blocks - real) * V, real * V


# ------------------------------------------------------------------------------------------------------- synthetic
class SyntheticBatcher(C.PackedBatcher):
    def __init__(self, cfg, device, seed: int, stream: int = 0, weights=None):
        super().__init__(cfg, device, weights=weights, seed=seed)
        self.seed, self.stream = int(seed), int(stream)
        self.pool_state: dict[int, dict] = {}

    def _pool_seed(self, bi: int, n: int) -> int:
        return (self.seed * 1_000_003 + self.stream * 10_007 + bi * 101 + n * 13) % (2 ** 62)

    def _make_pool(self, bi: int, n: int, nb: int, Lb: int):
        devs = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(self._pool_seed(bi, n))
            buf = C.generate_blocks(nb, self.V, Lb, self.device, self.weights, profile=self.profile, group_prob=self.group_prob)
        return buf

    def _take(self, bi, k):
        E = int(self.edges[bi])
        Lb = E + self.pred
        V = self.V
        st = self.pool_state.setdefault(bi, {"n": -1, "ptr": 0, "nb": 0})
        buf = self.pool.get(bi)
        if buf is None and st["n"] >= 0 and st["nb"]:                  # resumed: rebuild the pool the saved pointer refers to
            buf = self._make_pool(bi, st["n"], st["nb"], Lb)
        if buf is None or buf[0].shape[0] - st["ptr"] < k:
            nb = max(k, -(-256 // V), min(self.pool_max_rows // V, int(self.pool_pts // (Lb * V))))
            st["n"], st["nb"], st["ptr"] = st["n"] + 1, nb, 0
            buf = self._make_pool(bi, st["n"], nb, Lb)
        ptr = st["ptr"]
        out = tuple(t[ptr:ptr + k] for t in buf)
        st["ptr"] = ptr + k
        self.pool[bi] = buf
        return out

    def next_batch(self, layout=None):
        b = super().next_batch(layout)
        b["loss_ok"] = torch.ones_like(b["mask"])                 # synthetic: every observed point is scored
        return b

    def state_dict(self) -> dict:
        return {"gen": self.gen.get_state(), "pool_state": {int(b): dict(s) for b, s in self.pool_state.items()}}

    def load_state_dict(self, sd: dict) -> None:
        self.gen.set_state(sd["gen"])
        self.pool_state = {int(b): dict(s) for b, s in sd["pool_state"].items()}
        self.pool = {}


# ------------------------------------------------------------------------------------------------------- real
class RealBatcher(C.PackedBatcher):
    """Real windows in the packed layout. A batch is made in three steps, so the slow one (file reads) can run on threads and overlap the
    GPU step: PLAN (the layout, then for every cell: dataset, series, start, the missing positions: random draws and index look-ups only),
    FETCH (positioned reads of the planned windows, on `fetch_workers x threads_per_worker` threads), BUILD (group builder, placement on the
    GPU, blank spans). With prefetch the plan and the fetch of batch k+1 are made at the end of the call that returns batch k. The random
    draws happen in the same order as in the first, synchronous loader, so the batches are identical (tests/test_real_loader_golden.py)."""

    def __init__(self, cfg, device, store, targets: dict[str, float], mix: MixConfig, rows: int, seed: int, stream: int = 0):
        super().__init__(dict(cfg, pack_rows=rows, group_prob=mix.real_group_frac), device, seed=seed)
        if not self.group or not self.decoder:
            raise ValueError("the real batcher needs the causal decoder with group attention")
        mix.check()
        self.store, self.mix = store, mix
        store.configure_io(mix.io)
        self.names = store.names
        D = len(self.names)
        t = np.array([max(float(targets.get(n, 0.0)), 0.0) for n in self.names])
        if t.sum() <= 0:
            raise ValueError("no real dataset has a positive target share")
        self.targets = t / t.sum()
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(stream), 7]))
        self.a = np.zeros(D)                                   # controller: log-weight adjustment per dataset
        self.cnt_valid = np.zeros(D)                           # cumulative VALID points drawn per dataset
        self.cnt_cells = np.zeros(D)
        self.cnt_missing = np.zeros(D)                         # filled (originally missing) points drawn
        self.rejected = 0                                      # window candidates skipped by the horizon rule
        self.forced = 0                                        # windows accepted although they broke the rule (tries exhausted)
        self.no_dataset = 0                                    # cells for which no dataset had a long enough series
        self._pool: ThreadPoolExecutor | None = None
        self._pending: dict | None = None
        self.timers = {"plan": 0.0, "wait": 0.0, "build": 0.0, "batches": 0}

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None

    # ---- PLAN
    def _choose_window(self, di: int, c: int):
        s = self.store
        k0 = int(np.searchsorted(s.sorted_len[di], c, side="left"))
        j = int(self.rng.integers(k0, len(s.lens[di])))
        ser = int(s.order[di][j])
        slack = int(s.lens[di][ser]) - c
        start = int(self.rng.random() * (slack + 1))
        return ser, start, s.window_missing(di, ser, start, c)

    def _horizon_ok(self, miss: np.ndarray, c: int) -> bool:
        H = min(c, self.mix.horizon_points)
        m = len(miss) - int(np.searchsorted(miss, c - H, side="left"))
        return m < H and m / H <= self.mix.max_missing_horizon

    def _plan_bucket(self, cs: np.ndarray, E: int) -> dict:
        """Which window goes into every lane of k cells of context lengths cs (k,). No file is read."""
        s, V, D = self.store, self.V, len(self.names)
        k = len(cs)
        elig = np.stack([np.searchsorted(s.sorted_len[d], cs, side="left") < len(s.lens[d]) for d in range(D)], axis=1)   # (k, D)
        w = self.targets[None, :] * np.exp(self.a)[None, :] * elig
        none = w.sum(1) <= 0
        if none.any():                                          # no planned dataset has a series that long: any eligible one
            w[none] = elig[none].astype(float)
            self.no_dataset += int(none.sum())
        p = w / w.sum(1, keepdims=True)
        d_idx = np.minimum((self.rng.random(k)[:, None] > np.cumsum(p, axis=1)).sum(1), D - 1)
        ser = np.zeros((k, V), dtype=np.int64)
        start = np.zeros((k, V), dtype=np.int64)
        valid = np.zeros((k, V), dtype=np.int64)
        miss_l: list[list[np.ndarray]] = []
        for i in range(k):
            di, c = int(d_idx[i]), int(cs[i])
            row = []
            for lane in range(V):
                best = None
                for t in range(self.mix.tries):
                    cand = self._choose_window(di, c)
                    if self._horizon_ok(cand[2], c):
                        best = cand
                        break
                    self.rejected += 1
                    if best is None or len(cand[2]) < len(best[2]):
                        best = cand
                else:
                    self.forced += 1
                ser[i, lane], start[i, lane] = best[0], best[1]
                row.append(best[2])
                valid[i, lane] = c - len(best[2])
            miss_l.append(row)
            self.cnt_valid[di] += valid[i].sum()
            self.cnt_cells[di] += 1
            self.cnt_missing[di] += (cs[i] * V) - valid[i].sum()
        return {"E": E, "cs": cs.copy(), "d_idx": d_idx, "ser": ser, "start": start, "miss": miss_l, "valid": valid}

    def _plan_batch(self, lay: dict) -> dict:
        t0 = time.perf_counter()
        c = lay["c"]
        bucket = torch.bucketize(c, self.edges)
        buckets = []
        for bi in bucket.unique().tolist():
            sel = (bucket == bi).nonzero().squeeze(1)
            b = self._plan_bucket(c[sel].numpy(), int(self.edges[bi]))
            b["sel"] = sel
            buckets.append(b)
        self._update_controller()                              # after the draws of this batch, before those of the next (as before)
        self.timers["plan"] += time.perf_counter() - t0
        return {"lay": lay, "buckets": buckets, "valid_total": int(sum(b["valid"].sum() for b in buckets))}

    # ---- FETCH
    def _fill(self, b: dict, items: list[tuple[int, int]]) -> None:
        gx, miss_out, E = b["gx"], b["miss_out"], b["E"]
        for i, lane in items:
            di, c = int(b["d_idx"][i]), int(b["cs"][i])
            self.store.read_window(di, int(b["ser"][i, lane]), int(b["start"][i, lane]), c, gx[i, lane])
            if E > c:
                gx[i, lane] = np.resize(gx[i, lane, :c], E)      # tiled to the bucket length: only the first c points are used
            miss_out[i, lane, b["miss"][i][lane]] = True

    def _submit(self, pending: dict) -> None:
        V = self.V
        n_thr = self.mix.fetch_threads
        for b in pending["buckets"]:
            k, E = len(b["cs"]), b["E"]
            b["gx"] = np.zeros((k, V, E), dtype=np.float32)
            b["miss_out"] = np.zeros((k, V, E), dtype=bool)
            items = [(i, lane) for i in range(k) for lane in range(V)]
            if n_thr > 0:
                if self._pool is None:
                    self._pool = ThreadPoolExecutor(max_workers=n_thr, thread_name_prefix="realio")
                step = max(1, -(-len(items) // (3 * n_thr)))
                b["futs"] = [self._pool.submit(self._fill, b, items[j:j + step]) for j in range(0, len(items), step)]
            else:
                b["futs"], b["items"] = None, items

    def _wait(self, pending: dict) -> None:
        t0 = time.perf_counter()
        for b in pending["buckets"]:
            if b["futs"] is None:
                self._fill(b, b.pop("items"))
            else:
                for f in b["futs"]:
                    f.result()                                  # re-raises a read error of a worker
        self.timers["wait"] += time.perf_counter() - t0

    # ---- BUILD (PackedBatcher.next_batch for the decoder / group case, with real windows, st = 0 and a loss mask)
    @torch.no_grad()
    def _build(self, pending: dict):
        t0 = time.perf_counter()
        lay = pending["lay"]
        dev, W, V, pl, P = self.device, self.W, self.V, self.pl, self.P
        X = torch.zeros(self.R * W, device=dev)
        MK = torch.zeros(self.R * W, dtype=torch.bool, device=dev)
        LOSS = torch.zeros(self.R * W, dtype=torch.bool, device=dev)
        Y = torch.zeros(self.S, self.pred, device=dev)
        lane = torch.arange(V, device=dev)
        ROLE = torch.zeros(self.Sb, V, dtype=torch.long, device=dev)
        LG = lane.expand(self.Sb, V).clone()
        c, row, start, n, slot = lay["c"], lay["row"], lay["start"], lay["n"], lay["slot"]
        first_pt = ((row.unsqueeze(1) * V + torch.arange(V).unsqueeze(0)) * W + ((start + n) * pl - c).unsqueeze(1))
        for b in pending["buckets"]:
            sel, E = b["sel"], b["E"]
            k = sel.numel()
            gx, grole, glg = C._multivariatize(torch.from_numpy(b["gx"]).to(dev), self.group_prob, dev)
            miss = torch.from_numpy(b["miss_out"]).to(dev)
            cs = c[sel].to(dev)
            j = torch.arange(E, device=dev).view(1, 1, E)
            ok = (j < cs.view(k, 1, 1)).expand(k, V, E)
            dst = first_pt[sel].to(dev).unsqueeze(-1) + j
            X[dst[ok]] = gx[ok]
            MK[dst[ok]] = True                                   # forward-filled points are observed (as in the scorers)
            LOSS[dst[ok]] = ~miss[ok]                            # ... but a point that was missing is not scored
            sl = slot[sel].to(dev)
            ROLE[sl] = grole
            LG[sl] = glg
        sp = lay["seg_patch"].to(dev)
        seg_full = (sp.unsqueeze(1) * V + lane.view(1, V, 1)).reshape(self.R, P)
        role_p = ROLE[sp].permute(0, 2, 1).reshape(self.R, P)
        gid = (sp.unsqueeze(1) * V + LG[sp].permute(0, 2, 1)).reshape(self.R, P)
        blank = self._cpm_blank(lay).to(dev).repeat_interleave(V, dim=0)
        blank = blank & ~(role_p == C.ROLE_KNOWN).repeat_interleave(pl, dim=1)
        out = dict(blank=blank, lanes=V, gid=gid, role=role_p, slot_loss=(ROLE != C.ROLE_KNOWN).reshape(-1),
                   x=X.view(self.R, W), mask=MK.view(self.R, W), loss_ok=LOSS.view(self.R, W), y=Y, seg_patch=seg_full,
                   slot_valid=lay["slot_valid"].to(dev).repeat_interleave(V),
                   slot_ctx_len=torch.zeros(self.Sb, dtype=torch.long).index_put_((slot,), c).to(dev).repeat_interleave(V),
                   points=pending["valid_total"], n_series=int(c.numel()) * V)
        self.timers["build"] += time.perf_counter() - t0
        return out

    def next_batch(self, layout=None):
        if layout is not None:                                   # an explicit layout: plain synchronous path
            pending = self._plan_batch(layout)
            self._submit(pending)
        else:
            if self._pending is None:
                self._pending = self._plan_batch(self.sample_layout())
            pending, self._pending = self._pending, None
            if "gx" not in pending["buckets"][0]:                # fresh, or restored from a checkpoint
                self._submit(pending)
        self._wait(pending)
        out = self._build(pending)
        if layout is None and self.mix.prefetch:
            self._pending = self._plan_batch(self.sample_layout())     # the draws that the next call would make, in the same order
            self._submit(self._pending)                                # ... and its reads run while the GPU works on this batch
        self.timers["batches"] += 1
        return out

    def _update_controller(self) -> None:
        tot = self.cnt_valid.sum()
        if tot <= 0:
            return
        r = self.cnt_valid / tot
        live = self.targets > 0
        self.a[live] = np.clip(self.a[live] + self.mix.controller_rate * (np.log(self.targets[live] + 1e-12) - np.log(r[live] + 1e-6)),
                               -self.mix.a_clip, self.mix.a_clip)

    # ---- state and reporting
    def realised(self) -> dict:
        tot = max(self.cnt_valid.sum(), 1.0)
        return {n: {"target": float(self.targets[i]), "realised": float(self.cnt_valid[i] / tot), "valid_points": float(self.cnt_valid[i]),
                    "missing_share": float(self.cnt_missing[i] / max(self.cnt_valid[i] + self.cnt_missing[i], 1.0))}
                for i, n in enumerate(self.names)}

    def state_dict(self) -> dict:
        pend = None
        if self._pending is not None:                            # the plan of the next batch (no data): it is replayed after a restore
            pend = {"lay": self._pending["lay"], "valid_total": self._pending["valid_total"],
                    "buckets": [{k: v for k, v in b.items() if k not in ("gx", "miss_out", "futs", "items")} for b in self._pending["buckets"]]}
        return {"gen": self.gen.get_state(), "rng": self.rng.bit_generator.state, "a": self.a.copy(), "cnt_valid": self.cnt_valid.copy(),
                "cnt_cells": self.cnt_cells.copy(), "cnt_missing": self.cnt_missing.copy(), "rejected": self.rejected,
                "forced": self.forced, "no_dataset": self.no_dataset, "pending": pend}

    def load_state_dict(self, sd: dict) -> None:
        self.gen.set_state(sd["gen"])
        self.rng.bit_generator.state = sd["rng"]
        self.a, self.cnt_valid = sd["a"].copy(), sd["cnt_valid"].copy()
        self.cnt_cells, self.cnt_missing = sd["cnt_cells"].copy(), sd["cnt_missing"].copy()
        self.rejected, self.forced, self.no_dataset = sd["rejected"], sd["forced"], sd["no_dataset"]
        self._pending = sd.get("pending")


# ------------------------------------------------------------------------------------------------------- mixed
CAT0 = ("x", "mask", "blank", "role", "loss_ok", "y", "slot_valid", "slot_ctx_len", "slot_loss")
OFFSET = ("seg_patch", "gid")


def concat_batches(syn: dict, real: dict) -> dict:
    """Synthetic part first, real part after it: rows are stacked, slot ids of the real part are shifted by the synthetic slot count."""
    s1 = syn["slot_valid"].numel()
    out = {k: torch.cat([syn[k], real[k]], 0) for k in CAT0}
    for k in OFFSET:
        out[k] = torch.cat([syn[k], real[k] + s1], 0)
    out.update(lanes=syn["lanes"], points=syn["points"] + real["points"], n_series=syn["n_series"] + real["n_series"],
               parts=(s1, real["slot_valid"].numel()), points_syn=syn["points"], points_real=real["points"])
    return out


class MixedBatcher:
    def __init__(self, cfg: dict, device, store, targets: dict[str, float], mix: MixConfig, seed: int, rank: int = 0):
        self.cfg, self.mix, self.device = cfg, mix, device
        V = int(cfg.get("group_lanes", 8))
        rows_syn, rows_real = mix.split_rows(V)
        self.rows_syn, self.rows_real = rows_syn, rows_real
        self.syn = SyntheticBatcher(dict(cfg, pack_rows=rows_syn), device, seed=seed, stream=rank) if rows_syn else None
        self.real = RealBatcher(cfg, device, store, targets, mix, rows_real, seed=seed, stream=rank) if rows_real else None
        if self.syn is None or self.real is None:
            raise ValueError("a mixed batch needs both parts (real_fraction strictly between 0 and 1)")

    def next_batch(self) -> dict:
        return concat_batches(self.syn.next_batch(), self.real.next_batch())

    def close(self) -> None:
        self.real.close()

    def state_dict(self) -> dict:
        return {"syn": self.syn.state_dict(), "real": self.real.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.syn.load_state_dict(sd["syn"])
        self.real.load_state_dict(sd["real"])
