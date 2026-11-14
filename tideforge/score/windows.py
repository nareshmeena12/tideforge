"""Window cache: every (task, window, item) of a task list, cut by fev's own Task API (iter_windows / get_input_data /
get_ground_truth), stored once per task as .npy files (memory-mapped when read) and shared by every model, mode and seed.

Per task directory (cache/<key>/<safe task name>/), info.json written last (= done):
  tv.npy        (T, D) float64   each item's full target series, concatenated (D = target columns); stored once per item,
                                  windows are slices of it (no 20x copy for 20 windows)
  toff.npy      (n_items + 1,)   item offsets into tv
  cov_<j>.npy   (T,) float32     numeric dynamic covariate j (info["covs"]) on the same axis
  r_win, r_item, r_start, r_end  (R,) int64: row = (window, item); the window's past is tv[toff[i] + start : toff[i] + end]
  r_round       (R,) bool        fev's table for this window came back rounded through float32 (seen when a window drops an
                                  item); y_true / scale are taken from that table either way. Model inputs: past() always
                                  rounds through float32, as fev's python access (window[col][k]) returns float32 arrays
  y_true.npy    (R, H, D) float64 fev's ground truth for the row (checked equal to the series after the past)
  scale.npy     (R, D) float64   fev's abs seasonal error of the row's past (fev.metrics._abs_seasonal_error_per_item on the
                                  window's past, exactly as Task.evaluation_summary computes it)
  sc_<j>_vals / sc_<j>_off, sf_<j>  non-numeric covariate j encoded per window as score_groups.encode_column does (the code
                                  book is per window), context cut to `ctx` points, future (R, H) for known columns
Every row's past and ground truth are checked against the window data fev returns (a mismatch raises).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

FORMAT = 1


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def task_key(kw: dict, ctx: int) -> str:
    h = hashlib.sha256(json.dumps({"kw": kw, "ctx": ctx, "format": FORMAT}, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def _flat(col):
    """A list<...> arrow column -> (flat numpy values, lengths)."""
    import pyarrow.compute as pc
    return pc.list_flatten(col).to_numpy(zero_copy_only=False), pc.list_value_length(col).to_numpy(zero_copy_only=False)


def _is_numeric(col) -> bool:
    import pyarrow as pa
    t = col.type.value_type if hasattr(col.type, "value_type") else col.type
    return pa.types.is_integer(t) or pa.types.is_floating(t) or pa.types.is_boolean(t) or pa.types.is_decimal(t)


def build_task(kw: dict, out: Path, ctx: int = 8192, num_proc: int = 0) -> dict:
    """Cache one task (see the module docstring) -> its info dict."""
    import fev
    from fev.metrics import _abs_seasonal_error_per_item
    from tideforge.eval import thirdparty
    G = thirdparty.score_groups()
    out = Path(out)
    if (out / "info.json").exists():
        return json.loads((out / "info.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)
    task = fev.Task(**kw)
    full = task.load_full_dataset(num_proc=num_proc).data.table
    idc, tsc = task.id_column, task.timestamp_column
    tcols, past, known = list(task.target_columns), list(task.past_dynamic_columns), list(task.known_dynamic_columns)
    D, H, S = len(tcols), int(task.horizon), int(task.seasonality)
    ids = [str(x) for x in full.column(idc).to_pylist()]
    id2i = {x: i for i, x in enumerate(ids)}
    vals, lens = zip(*[_flat(full.column(c)) for c in tcols])
    toff = np.r_[0, np.cumsum(lens[0])].astype(np.int64)
    tv = np.stack([np.asarray(v, dtype=np.float64) for v in vals], axis=1)
    ts, _ = _flat(full.column(tsc))
    num_cov = [c for c in known + past if _is_numeric(full.column(c))]
    str_cov = [c for c in known + past if c not in num_cov]
    for j, c in enumerate(num_cov):
        np.save(out / f"cov_{j}.npy", np.asarray(_flat(full.column(c))[0], dtype=np.float32))
    np.save(out / "toff.npy", toff)
    rounded: list[np.ndarray] = []
    rw, ri, rs, re_, ytrue, scale = [], [], [], [], [], []
    sc = {c: ([], [0], []) for c in str_cov}             # (ctx arrays, offsets, future arrays)
    n_windows = 0
    for wi, w in enumerate(task.iter_windows(num_proc=num_proc)):
        n_windows += 1
        p, f = w.get_input_data()
        g = w.get_ground_truth()
        pt, ft, gt = p.data.table, f.data.table, g.data.table
        items = np.array([id2i[str(x)] for x in pt.column(idc).to_pylist()], dtype=np.int64)
        pv, plen = zip(*[_flat(pt.column(c)) for c in tcols])
        plen = np.asarray(plen[0], dtype=np.int64)
        pflat = np.stack([np.asarray(v, dtype=np.float64) for v in pv], axis=1)
        pts, _ = _flat(pt.column(tsc))
        ends_flat = np.cumsum(plen)
        end = np.empty(len(items), np.int64)
        for k, (i, e_) in enumerate(zip(items, ends_flat)):
            t_i = ts[toff[i]:toff[i + 1]]
            end[k] = int(np.searchsorted(t_i, pts[e_ - 1], side="right"))
        start = end - plen
        gidx = np.concatenate([np.arange(toff[i] + s, toff[i] + e) for i, s, e in zip(items, start, end)]) if len(items) else \
            np.zeros(0, np.int64)
        # fev returns some windows rounded through float32 (seen when the window drops an item), others not: the row
        # keeps a flag and past() rounds the same way, so a model sees exactly fev's numbers
        rnd = False
        if (start < 0).any():
            raise RuntimeError(f"{task.task_name} window {wi}: fev's past starts before the series")
        if not np.array_equal(tv[gidx], pflat, equal_nan=True):
            rnd = np.array_equal(tv[gidx].astype(np.float32).astype(np.float64), pflat, equal_nan=True)
            if not rnd:
                raise RuntimeError(f"{task.task_name} window {wi}: fev's past is not a slice of the full series")
        rounded.append(np.full(len(items), rnd))
        if len(items):                                   # the model side: python access as the old scorer read it
            a = np.asarray(p[tcols[0]][0], dtype=np.float64)
            if not np.array_equal(tv[toff[items[0]] + start[0]:toff[items[0]] + end[0], 0].astype(np.float32)
                                  .astype(np.float64), a, equal_nan=True):
                raise RuntimeError(f"{task.task_name} window {wi}: python access does not return the float32 past")
        gv = np.stack([np.asarray(_flat(gt.column(c))[0], dtype=np.float64) for c in tcols], axis=1).reshape(len(items), H, D)
        fidx = np.stack([toff[i] + e + np.arange(H) for i, e in zip(items, end)]) if len(items) else np.zeros((0, H), np.int64)
        tf = tv[fidx].astype(np.float32).astype(np.float64) if rnd else tv[fidx]
        if (fidx >= toff[items + 1][:, None]).any() or not (np.array_equal(tf, gv, equal_nan=True) or np.array_equal(
                tv[fidx].astype(np.float32).astype(np.float64), gv, equal_nan=True) or np.array_equal(tv[fidx], gv, equal_nan=True)):
            raise RuntimeError(f"{task.task_name} window {wi}: fev's ground truth is not the series after the past")
        scale.append(_abs_seasonal_error_per_item(y_past=pflat, y_past_lengths=plen, seasonality=S).reshape(len(items), D))
        ytrue.append(gv)
        rw.append(np.full(len(items), wi, np.int64)); ri.append(items); rs.append(start); re_.append(end)
        for c in str_cov:                                # per-window code book, as score_groups.window_groups
            arrs = [np.asarray(x) for x in pt.column(c).to_pylist()]
            fut = [np.asarray(x) for x in ft.column(c).to_pylist()] if c in known else []
            enc = G.encode_column(arrs + fut)
            ctx_l, off, fu = sc[c]
            for k in range(len(items)):
                a = enc[k][-ctx:]
                ctx_l.append(a); off.append(off[-1] + len(a))
                fu.append(enc[len(items) + k][:H] if fut else np.zeros(H, np.float32))
    np.save(out / "tv.npy", tv)
    np.save(out / "r_round.npy", np.concatenate(rounded) if rounded else np.zeros(0, bool))
    R = int(sum(len(x) for x in rw))
    for name, arrs in (("r_win", rw), ("r_item", ri), ("r_start", rs), ("r_end", re_)):
        np.save(out / f"{name}.npy", np.concatenate(arrs) if arrs else np.zeros(0, np.int64))
    np.save(out / "y_true.npy", np.concatenate(ytrue) if ytrue else np.zeros((0, H, D)))
    np.save(out / "scale.npy", np.concatenate(scale) if scale else np.zeros((0, D)))
    for j, c in enumerate(str_cov):
        ctx_l, off, fu = sc[c]
        np.save(out / f"sc_{j}_vals.npy", np.concatenate(ctx_l).astype(np.float32) if ctx_l else np.zeros(0, np.float32))
        np.save(out / f"sc_{j}_off.npy", np.asarray(off, np.int64))
        np.save(out / f"sf_{j}.npy", np.stack(fu).astype(np.float32) if fu else np.zeros((0, H), np.float32))
    info = dict(task_name=task.task_name, kw=kw, horizon=H, seasonality=S, freq=str(getattr(task, "freq", "") or ""),
                quantile_levels=list(task.quantile_levels), eval_metric=task.eval_metric, extra_metrics=list(task.extra_metrics),
                target_columns=tcols, past=past, known=known, num_cov=num_cov, str_cov=str_cov, n_windows=n_windows,
                n_rows=R, n_items=len(ids), ctx=ctx, rounded_windows=int(sum(bool(x.any()) for x in rounded)), format=FORMAT,
                bytes=int(sum(p_.stat().st_size for p_ in out.glob("*.npy"))))
    tmp = out / "info.json.tmp"
    tmp.write_text(json.dumps(info, default=str), encoding="utf-8")
    os.replace(tmp, out / "info.json")
    return info


class TaskCache:
    """One cached task, memory-mapped."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.info = json.loads((self.path / "info.json").read_text(encoding="utf-8"))
        ld = lambda n: np.load(self.path / f"{n}.npy", mmap_mode="r")      # noqa: E731
        self.tv, self.toff = ld("tv"), np.asarray(ld("toff"))
        self.r_win, self.r_item = np.asarray(ld("r_win")), np.asarray(ld("r_item"))
        self.r_start, self.r_end = np.asarray(ld("r_start")), np.asarray(ld("r_end"))
        self.y_true, self.scale = ld("y_true"), ld("scale")
        self.r_round = np.asarray(ld("r_round"))
        self.cov = {c: ld(f"cov_{j}") for j, c in enumerate(self.info["num_cov"])}
        self.sc = {c: (ld(f"sc_{j}_vals"), np.asarray(ld(f"sc_{j}_off")), ld(f"sf_{j}"))
                   for j, c in enumerate(self.info["str_cov"])}

    @property
    def name(self) -> str:
        return self.info["task_name"]

    @property
    def has_groups(self) -> bool:
        """score_groups.has_groups: several targets or any dynamic covariate."""
        i = self.info
        return len(i["target_columns"]) > 1 or bool(i["known"]) or bool(i["past"])

    def past(self, r: int, d: int) -> np.ndarray:
        """The row's full past of target d as a model receives it from fev (window[col][k] is float32; returned as float64,
        NaN kept). The metric arrays (y_true, scale) keep fev's table values instead."""
        o = self.toff[self.r_item[r]]
        return np.asarray(self.tv[o + self.r_start[r]:o + self.r_end[r], d]).astype(np.float32).astype(np.float64)

    def cov_past(self, r: int, c: str, ctx: int) -> np.ndarray:
        if c in self.cov:
            o = self.toff[self.r_item[r]]
            return np.asarray(self.cov[c][o + self.r_start[r]:o + self.r_end[r]], dtype=np.float32)[-ctx:]
        v, off, _ = self.sc[c]
        return np.asarray(v[off[r]:off[r + 1]])[-ctx:]

    def cov_future(self, r: int, c: str) -> np.ndarray:
        H = self.info["horizon"]
        if c in self.cov:
            o = self.toff[self.r_item[r]] + self.r_end[r]
            return np.asarray(self.cov[c][o:o + H], dtype=np.float32)
        return np.asarray(self.sc[c][2][r])


def _build_one(kw, out, ctx, num_proc):
    return build_task(kw, out, ctx, num_proc)


def build(specs: list[dict], root: Path, ctx: int = 8192, workers: int = 8, num_proc: int = 0, log=print) -> list[TaskCache]:
    """Cache every task of `specs` under root (resumable: done tasks are skipped), tasks in parallel processes."""
    root = Path(root)
    dirs = [root / f"{safe(s['kw']['task_name'])}-{task_key(s['kw'], ctx)}" for s in specs]
    todo = [(s, d) for s, d in zip(specs, dirs) if not (d / "info.json").exists()]
    if todo:
        log(f"window cache: building {len(todo)} of {len(specs)} tasks in {root} ({workers} workers)")
        if workers <= 1:
            for s, d in todo:
                build_task(s["kw"], d, ctx, num_proc)
        else:
            with ProcessPoolExecutor(workers) as ex:
                futs = {ex.submit(_build_one, s["kw"], d, ctx, num_proc): s["kw"]["task_name"] for s, d in todo}
                for f in as_completed(futs):
                    f.result()                           # a failed task stops the build with its error
    caches = [TaskCache(d) for d in dirs]
    size = sum(c.info["bytes"] for c in caches)
    log(f"window cache: {len(caches)} tasks, {sum(c.info['n_rows'] for c in caches)} rows (window x item), "
        f"{size / 2 ** 30:.2f} GiB on disk")
    return caches
