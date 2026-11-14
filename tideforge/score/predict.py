"""The batching engine: every (task, window, item, target) of a task list goes to the model in large pooled batches, one call
per horizon and kind (not per task or window), with resumable shards on disk.

Units, per task:
  plain  (task, row, target): the univariate path. Used for every task in mode OFF, for tasks without groups
         (one target, no dynamic covariate) in mode ON, for univariate models, and when the model cannot take a group at
         this horizon (ft3_core.forecast_groups falls back to forecasting the targets one by one there as well).
  group  (task, row): mode ON with a group model: the row's targets (sibling variates; a joint task's columns), its past
         covariates and its known covariates with their future values, as score_groups.window_groups builds them.
Units are sorted (task, row, target) and cut into shards of at most `chunk_points` context points; each shard is one call
to the predictor and one file preds/<mode>/<fingerprint>/H<h>_<kind>_<k>.npz (task, row, target, q). A crashed run
continues from the shards that exist.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def fingerprint(caches, model_id: str, mode: str) -> str:
    key = json.dumps({"tasks": [(c.name, c.info["n_rows"]) for c in caches], "model": model_id, "mode": mode}, sort_keys=True)
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def use_groups(cache, pred, mode: str) -> bool:
    return mode == "on" and pred.groups and cache.has_groups and pred.groups_ok(cache.info["horizon"])


def _plain_inputs(caches, units, pred):
    return [pred.prepare(caches[t].past(r, d), caches[t].info["seasonality"]) for t, r, d in units]


def _group_inputs(caches, units, pred):
    tg, ps, kn = [], [], []
    for t, r in units:
        c = caches[t]
        S, C = c.info["seasonality"], pred.max_context or 10 ** 9
        tg.append([pred.prepare(c.past(r, d), S) for d in range(len(c.info["target_columns"]))])
        ps.append([c.cov_past(r, col, C) for col in c.info["past"]])
        kn.append([(c.cov_past(r, col, C), c.cov_future(r, col)) for col in c.info["known"]])
    return tg, ps, kn


def _shards(units, sizes, chunk_points):
    out, cur, pts = [], [], 0
    for u, s in zip(units, sizes):
        if cur and pts + s > chunk_points:
            out.append(cur)
            cur, pts = [], 0
        cur.append(u)
        pts += s
    if cur:
        out.append(cur)
    return out


def run(caches, pred, mode: str, out_dir: Path, model_id: str, chunk_points: int = 300_000_000, log=print) -> tuple[dict, dict]:
    """-> ({task name: q (R, D, Q, H) float32}, stats)."""
    d = Path(out_dir) / "preds" / mode / fingerprint(caches, model_id, mode)
    d.mkdir(parents=True, exist_ok=True)
    Q = len(pred.quantile_levels)
    by_h: dict[tuple[int, str], list] = {}
    for t, c in enumerate(caches):
        H = c.info["horizon"]
        if use_groups(c, pred, mode):
            by_h.setdefault((H, "group"), []).extend((t, r) for r in range(c.info["n_rows"]))
        else:
            by_h.setdefault((H, "plain"), []).extend((t, r, k) for r in range(c.info["n_rows"])
                                                    for k in range(len(c.info["target_columns"])))
    stats = dict(series=0, model_s=0.0, prep_s=0.0, shards=0, reused=0)
    for (H, kind), units in sorted(by_h.items()):
        sizes = [min(int(caches[u[0]].r_end[u[1]] - caches[u[0]].r_start[u[1]]), pred.max_context or 10 ** 12)
                 * (len(caches[u[0]].info["target_columns"]) if kind == "group" else 1) for u in units]
        for k, part in enumerate(_shards(units, sizes, chunk_points)):
            f = d / f"H{H}_{kind}_{k:05d}.npz"
            n_series = len(part) if kind == "plain" else sum(len(caches[t].info["target_columns"]) for t, _ in part)
            stats["series"] += n_series
            if f.exists():
                stats["reused"] += 1
                continue
            t0 = time.perf_counter()
            if kind == "plain":
                x = _plain_inputs(caches, part, pred)
                t1 = time.perf_counter()
                S = caches[part[0][0]].info["seasonality"]
                if getattr(pred, "needs_season", False):
                    q = _by_season(pred, caches, part, x, H)
                else:
                    q = pred.predict(x, H, season=S)
                tt, rr, dd = (np.array(v, np.int64) for v in zip(*part))
            else:
                tg, ps, kn = _group_inputs(caches, part, pred)
                t1 = time.perf_counter()
                res = pred.predict(tg, H, past_covariates=ps, known_covariates=kn)
                tt = np.concatenate([np.full(len(a), t, np.int64) for (t, _), a in zip(part, res)])
                rr = np.concatenate([np.full(len(a), r, np.int64) for (_, r), a in zip(part, res)])
                dd = np.concatenate([np.arange(len(a)) for a in res])
                q = np.concatenate(res)
            _sync(pred)
            t2 = time.perf_counter()
            stats["prep_s"] += t1 - t0
            stats["model_s"] += t2 - t1
            stats["shards"] += 1
            tmp = f.with_suffix(".tmp.npz")
            np.savez(tmp, t=tt, r=rr, d=dd, q=np.asarray(q, np.float32))
            os.replace(tmp, f)
            log(f"  {mode} H={H} {kind} shard {k}: {n_series} series, prep {t1 - t0:.1f}s, model {t2 - t1:.1f}s "
                f"({n_series / max(t2 - t1, 1e-9):.0f} series/s)")
    q_all = {c.name: np.full((c.info["n_rows"], len(c.info["target_columns"]), Q, c.info["horizon"]), np.nan, np.float32)
             for c in caches}
    for f in sorted(d.glob("H*.npz")):
        z = np.load(f)
        for t in np.unique(z["t"]):
            m = z["t"] == t
            q_all[caches[t].name][z["r"][m], z["d"][m]] = z["q"][m]
    missing = [n for n, q in q_all.items() if np.isnan(q).all(axis=(2, 3)).any()]
    if missing:
        raise RuntimeError(f"predictions missing for {len(missing)} tasks: {missing[:5]}")
    return q_all, stats


def _by_season(pred, caches, part, x, H):
    seas = np.array([caches[t].info["seasonality"] for t, _, _ in part])
    q = np.zeros((len(part), len(pred.quantile_levels), H))
    for s in np.unique(seas):
        i = np.flatnonzero(seas == s)
        q[i] = pred.predict([x[j] for j in i], H, season=int(s))
    return q


def _sync(pred):
    dev = getattr(pred, "device", None)
    if dev is not None and getattr(dev, "type", "") == "cuda":
        import torch
        torch.cuda.synchronize(dev)
