"""Milestone 4 scoring on ONE interactive Perlmutter node (4 GPUs, 128 CPU cores).

Wired to the owner's pipeline in reference/time_fev_eval.py. Its scoring is NOT rewritten:
fill_season, naive_preds, the model_preds logic, the TIME loop (timebench Dataset /
save_window_predictions), the TIME leaderboard and head-to-head code, task.evaluation_summary
and fev.leaderboard are ported as written (the test compares them with the reference source
when reference/ is present). What changed is IO only: paths under $ROOT/eval, no network on
the node (everything is fetched first), one file per (model, task) so the four GPUs write
independently and every step resumes, and models loaded from our checkpoints with
tidemark.pretrain.inference.load_twin / forecast_left_padded (identical to the reference's).

  python -m tidemark.eval.score_twins fetch               LOGIN node (network): TIME repo + data,
                                                          official TIME results, fev tasks + data
                                                          + reference results, into $ROOT/eval
  python -m tidemark.eval.score_twins naive [--workers N] seasonal naive on every fev task, CPU pool,
                                                          cached in $ROOT/eval/fev_naive.csv
  python -m tidemark.eval.score_twins twins --gpu G       the 2 twins of GPU G (0 softmax, 1 GLA,
                                                          2 SWA16, 3 SWA8; seeds 0 and 1)
  python -m tidemark.eval.score_twins board               merge and print the boards
  (--smoke on naive / twins / board: 3 fev tasks and 2 TIME datasets, one twin, outputs in
   $ROOT/eval_smoke, the real caches are not touched)

TIME seasonal naive comes from the official TIME-Output results (not recomputed). fev-bench's
seasonal naive is computed here, as in the reference. scripts/perlmutter/score_interactive.sh
runs everything in one session.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

QL = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MAX_CTX = 8192
BATCH = 512
# the reference's TIME reference models (picked if the official results have them)
REFERENCE = ["Chronos-2", "Timestral-5M", "Timestral-28M", "Toto-2.0-4m", "Toto-2.0-22m", "TiRex"]
FEV_RAW = "https://raw.githubusercontent.com/autogluon/fev/main/benchmarks/fev_bench/"
FEV_YAML = FEV_RAW + "tasks.yaml"
FEV_REFERENCE_FILES = {"Chronos-2": "chronos-2.csv", "Timestral-5M": "timestral-5m.csv",
                       "Timestral-28M": "timestral-28m.csv", "Toto-2.0-4m": "toto-2_0-4m.csv",
                       "Toto-2.0-22m": "toto-2_0-22m.csv", "TiRex": "tirex.csv"}
TIME_REPO_URL = "https://github.com/zqiao11/TIME.git"
GPU_ARM = {0: "softmax", 1: "gla", 2: "swa16", 3: "swa8"}
SEEDS = (0, 1)
SMOKE_FEV, SMOKE_TIME, SMOKE_TWIN = 3, 2, "gla_s0"
NEEDED = ["fev", "datasets", "gluonts", "toolz", "dotenv", "yaml", "scipy", "huggingface_hub", "pandas"]
PIP_NAMES = {"dotenv": "python-dotenv", "yaml": "pyyaml", "fev": "fev"}
# arm = [a-z0-9]+ , or <arm>_lr<k> for a rerun at another start rate (stage B reruns: softmax_lr24_s0)
TAG_RE = re.compile(r"^(?P<arm>[a-z0-9]+(?:_lr\d+)?)_s(?P<seed>\d+)(?:_local(?P<w>\d+))?$")


# ================================== paths ===================================

def default_root() -> str:
    return os.environ.get("ROOT") or str(Path(os.environ.get("SCRATCH", ".")) / "tidemark")


class Paths:
    """Everything under $ROOT/eval; --smoke writes its outputs to $ROOT/eval_smoke and shares
    the downloaded data."""

    def __init__(self, root, smoke: bool = False):
        self.root = Path(root)
        base = self.root / "eval"
        self.data, self.hf = base / "data", base / "hf"
        self.out = self.root / "eval_smoke" if smoke else base
        self.time_repo, self.time_data = self.data / "TIME", self.data / "TIME_data"
        self.time_official = self.data / "TIME-Output"
        self.fev_yaml, self.fev_ref = self.data / "fev" / "tasks.yaml", self.data / "fev_reference"
        self.manifest = self.data / "manifest.json"
        self.time_out, self.fev_out = self.out / "time", self.out / "fev"
        self.naive_csv, self.naive_parts = self.out / "fev_naive.csv", self.out / "fev_naive_parts"
        self.board = self.out / "board"
        self.runs = self.root / "runs"


def safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name))


def setup_env(root, offline: bool) -> None:
    """Before anything imports datasets / huggingface_hub: caches under $ROOT/eval/hf; on the
    compute node nothing may touch the network."""
    os.environ.setdefault("HF_HOME", str(Paths(root).hf))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    os.replace(tmp, path)


def read_manifest(paths: Paths) -> dict:
    return json.loads(paths.manifest.read_text()) if paths.manifest.exists() else {}


# ========================= ported from the reference ========================

def fill_season(a, S):
    """Missing history values -> the value one season earlier (else the last value); leading NaNs dropped."""
    y = np.asarray(a, dtype=np.float64)
    ok = np.isfinite(y)
    if ok.all() or not ok.any():
        return y.astype(np.float32)
    s0 = int(np.argmax(ok)); y = y[s0:].copy(); ok = ok[s0:]
    idx = np.where(ok, np.arange(len(y)), 0); np.maximum.accumulate(idx, out=idx)
    ff = y[idx]
    out = y.copy()
    for t in np.flatnonzero(~ok):
        out[t] = out[t - S] if (S > 1 and t >= S and np.isfinite(out[t - S])) else ff[t]
    return out.astype(np.float32)


def naive_preds(task):
    q_keys = [str(q) for q in task.quantile_levels]
    out = []
    for w in task.iter_windows():
        past, _ = w.get_input_data()
        d = {}
        for col in task.target_columns:
            fs = []
            for y in past[col]:
                y = np.asarray(y, dtype=np.float64); y = y[np.isfinite(y)]
                if len(y) == 0:
                    y = np.zeros(1)
                sl = max(1, min(int(task.seasonality), len(y)))
                f = np.tile(y[-sl:], int(np.ceil(task.horizon / sl)))[:task.horizon]
                fs.append({"predictions": f.tolist(), **{k: f.tolist() for k in q_keys}})
            d[col] = fs
        out.append(d)
    return out


def fev_inputs(task):
    """First half of the reference's model_preds: the univariate series of every window and
    target column (fill_season, last MAX_CTX points). Built once and shared by both twins."""
    S = int(task.seasonality)
    series, index = [], []
    for wi, w in enumerate(task.iter_windows()):
        past, _ = w.get_input_data()
        for col in task.target_columns:
            ys = [fill_season(np.asarray(y, dtype=np.float64)[-MAX_CTX:], S) for y in past[col]]
            index.append((wi, col, len(ys))); series += ys
    return series, index


def fev_predictions(model, task, series, index, device, forecast=None):
    """Second half of the reference's model_preds: forecast and format for evaluation_summary."""
    if forecast is None:
        from tidemark.pretrain.inference import forecast_left_padded as forecast
    q_keys = [str(q) for q in task.quantile_levels]
    assert len(q_keys) == 9, "task quantiles != the model's 9"
    pt, qt = forecast(model, series, task.horizon, device, max_ctx=MAX_CTX, batch_size=BATCH)
    qt = np.sort(qt, axis=1); pt = qt[:, 4]
    out, k = [], 0
    for wi, col, n in index:
        if len(out) <= wi:
            out.append({})
        out[wi][col] = [{"predictions": pt[i].tolist(), **{q: qt[i, j].tolist() for j, q in enumerate(q_keys)}}
                        for i in range(k, k + n)]
        k += n
    return out


# ============================== task handling ===============================

def load_fev_tasks(paths: Paths):
    import fev
    return fev.Benchmark.from_yaml(paths.fev_yaml).tasks


def prepare(task, num_proc: int):
    """Load and cache the task's dataset with a bounded number of processes: fev's default is
    cpu_count() (128 here) per call, which several processes at once would multiply. num_proc=0
    keeps everything in the calling process (datasets 4.x starts a sub-pool even for 1; the 100
    naive workers must not each start one)."""
    task.load_full_dataset(num_proc=num_proc)
    return task


def smoke_fev(tasks, manifest: dict):
    """The SMOKE_FEV tasks with the fewest series (sizes from `fetch`); the first ones if unknown."""
    size = {r["task_name"]: r["items"] for r in manifest.get("fev", []) if r.get("items") is not None}
    if not size:
        return list(tasks)[:SMOKE_FEV]
    pick = sorted(tasks, key=lambda t: size.get(t.task_name, float("inf")))[:SMOKE_FEV]
    return [t for t in tasks if t in pick]


def smoke_time(names: list[str], manifest: dict) -> list[str]:
    size = manifest.get("time_sizes", {})
    if not size:
        return list(names)[:SMOKE_TIME]
    return sorted(sorted(names, key=lambda n: size.get(n, float("inf")))[:SMOKE_TIME], key=names.index)


# ============================= timebench (TIME) =============================

class TB:
    """The timebench names the reference imports, from the TIME repo clone under $ROOT/eval/data."""

    def __init__(self, paths: Paths):
        for p in (paths.time_repo / "src", paths.time_repo / "scripts"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))
        hits = sorted(paths.time_data.rglob("dataset_info.json"), key=lambda p: len(p.parts))
        if not hits:
            raise SystemExit(f"no TIME datasets under {paths.time_data}: run `fetch` on the login node")
        os.environ["TIME_DATASET"] = str(hits[0].parents[2])
        from gluonts.time_feature import get_seasonality
        from timebench.evaluation.data import Dataset, get_dataset_settings, load_dataset_config
        from timebench.evaluation.saver import save_window_predictions
        from timebench.evaluation.utils import get_available_terms
        self.get_seasonality, self.Dataset = get_seasonality, Dataset
        self.get_dataset_settings, self.load_dataset_config = get_dataset_settings, load_dataset_config
        self.save_window_predictions, self.get_available_terms = save_window_predictions, get_available_terms

    def leaderboard_module(self):
        import compute_local_leaderboard as LB
        return LB


def time_done(paths: Paths, tag: str, ds_name: str, term: str) -> bool:
    return (paths.time_out / tag / ds_name / term / "metrics.npz").exists()


def run_time(models: dict, paths: Paths, device, tb=None, smoke: bool = False, forecast=None, log=print) -> None:
    """The reference's RUN_TIME loop for every model in `models` ({tag: model}); the series are
    extracted once per (dataset, term) and shared; resumable per (tag, dataset, term)."""
    if forecast is None:
        from tidemark.pretrain.inference import forecast_left_padded as forecast
    tb = tb or TB(paths)
    config = tb.load_dataset_config()
    names = list(config["datasets"])
    if smoke:
        names = smoke_time(names, read_manifest(paths))
    t_all = time.time()
    for ds_name in names:
        for term in tb.get_available_terms(ds_name, config):
            todo = [t for t in models if not time_done(paths, t, ds_name, term)]
            if not todo:
                continue
            t0 = time.time()
            try:
                st = tb.get_dataset_settings(ds_name, term, config)
                ds = tb.Dataset(name=ds_name, term=term, to_univariate=False, prediction_length=st["prediction_length"],
                                test_length=st["test_length"], val_length=st["val_length"])
                H, D, S = ds.prediction_length, ds.target_dim, tb.get_seasonality(ds.freq)
                inputs = list(ds.test_data.input)
                ys = []
                for d in inputs:
                    a = np.asarray(d["target"])
                    for v in (a[None] if a.ndim == 1 else a):
                        ys.append(fill_season(v[-MAX_CTX:], S))
            except Exception as e:
                log(f"{ds_name:32s} {term:6s} FAILED (data): {type(e).__name__}: {str(e)[:160]}")
                continue
            for tag in todo:
                try:
                    _, q = forecast(models[tag], ys, H, device, max_ctx=MAX_CTX, batch_size=BATCH)   # (N*D, 9, H)
                    q = np.sort(q.astype(np.float64), axis=1)
                    q = q if D == 1 else q.reshape(len(inputs), D, len(QL), H).transpose(0, 2, 1, 3)
                    tb.save_window_predictions(dataset=ds, fc_quantiles=q, ds_config=f"{ds_name}/{term}",
                                               output_base_dir=str(paths.time_out / tag), seasonality=S,
                                               quantile_levels=QL,
                                               model_hyperparams={"model": tag, "context_length": MAX_CTX,
                                                                  "mode": "univariate"})
                except Exception as e:
                    log(f"{ds_name:32s} {term:6s} {tag} FAILED: {type(e).__name__}: {str(e)[:160]}")
                    continue
            log(f"{ds_name:32s} {term:6s} H={H:<4d} D={D:<3d} n={len(inputs):<6d} {len(todo)} model(s) "
                f"{time.time() - t0:6.1f}s")
    log(f"TIME: {(time.time() - t_all) / 60:.1f} min")


# ================================== fev =====================================

def fev_done(paths: Paths, tag: str, task_name: str) -> bool:
    return (paths.fev_out / tag / f"{safe(task_name)}.json").exists()


def run_fev(models: dict, tasks, paths: Paths, device, num_proc: int = 8, forecast=None, log=print) -> None:
    t_all = time.time()
    for ti, task in enumerate(tasks):
        tn = task.task_name
        todo = [t for t in models if not fev_done(paths, t, tn)]
        if not todo:
            continue
        t0 = time.time()
        try:
            prepare(task, num_proc)
            series, index = fev_inputs(task)
        except Exception as e:
            log(f"   {tn}: FAILED (data) {type(e).__name__}: {str(e)[:150]}")
            continue
        for tag in todo:
            try:
                pr = fev_predictions(models[tag], task, series, index, device, forecast)
                s_ = task.evaluation_summary(pr, model_name=tag, trained_on_this_dataset=False)
                write_json(paths.fev_out / tag / f"{safe(tn)}.json", s_)
            except Exception as e:
                log(f"   {tn} {tag}: FAILED {type(e).__name__}: {str(e)[:150]}")
        log(f"[{ti + 1}/{len(tasks)}] {tn:45s} {len(todo)} model(s) {time.time() - t0:6.1f}s")
    log(f"fev-bench: {(time.time() - t_all) / 60:.1f} min")


# ============================ seasonal naive (CPU) ==========================

_W: dict = {}


def _naive_init(yaml_path: str, num_proc: int):
    import fev
    _W["tasks"] = fev.Benchmark.from_yaml(yaml_path).tasks
    _W["num_proc"] = num_proc            # 0: in-process (see prepare)


def _naive_one(i: int):
    t0 = time.time()
    task = _W["tasks"][i]
    try:
        prepare(task, _W["num_proc"])
        s_ = task.evaluation_summary(naive_preds(task), model_name="seasonal_naive", trained_on_this_dataset=False)
        return i, task.task_name, s_, None, time.time() - t0
    except Exception as e:  # noqa: BLE001
        return i, task.task_name, None, f"{type(e).__name__}: {str(e)[:200]}", time.time() - t0


def run_naive(paths: Paths, workers: int, smoke: bool = False, log=print) -> bool:
    """Seasonal naive on every fev-bench task, in a pool of `workers` processes; one file per task
    (resumes), then the cache CSV. The step is skipped if the CSV exists."""
    import pandas as pd
    if paths.naive_csv.exists():
        log(f"naive: {paths.naive_csv} exists, skipped (delete it to recompute)")
        return True
    tasks = load_fev_tasks(paths)
    manifest = read_manifest(paths)
    if smoke:
        keep = {t.task_name for t in smoke_fev(tasks, manifest)}
        order = [i for i, t in enumerate(tasks) if t.task_name in keep]
    else:
        items = {r["task_name"]: r.get("items") or 0 for r in manifest.get("fev", [])}
        order = sorted(range(len(tasks)), key=lambda i: -items.get(tasks[i].task_name, 0))   # big tasks first
    todo = [i for i in order if not (paths.naive_parts / f"{safe(tasks[i].task_name)}.json").exists()]
    log(f"naive: {len(order)} tasks, {len(order) - len(todo)} already done, {len(todo)} to do, "
        f"{min(workers, max(1, len(todo)))} workers")
    failed, t_all = [], time.time()

    def took(res, k):
        _, tn, s_, err, dt = res
        if err:
            failed.append(tn)
            log(f"[{k}/{len(todo)}] {tn:45s} FAILED {err}")
        else:
            write_json(paths.naive_parts / f"{safe(tn)}.json", s_)
            log(f"[{k}/{len(todo)}] {tn:45s} {dt:6.1f}s")

    if workers <= 1 or len(todo) <= 1:
        _naive_init(str(paths.fev_yaml), 0)
        for k, i in enumerate(todo, 1):
            took(_naive_one(i), k)
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(workers, len(todo)), initializer=_naive_init,
                                          initargs=(str(paths.fev_yaml), 0)) as pool:
            for k, res in enumerate(pool.imap_unordered(_naive_one, todo, chunksize=1), 1):
                took(res, k)
    names = {tasks[i].task_name for i in order}
    rows = [json.loads(p.read_text()) for p in sorted(paths.naive_parts.glob("*.json"))]
    rows = [r for r in rows if r["task_name"] in names]
    if failed or len(rows) < len(names):
        log(f"naive: {len(names) - len(rows)} task(s) without a result ({failed}); the CSV is NOT written, "
            "fix and rerun (finished tasks are kept)")
        return False
    paths.naive_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(paths.naive_csv, index=False)
    log(f"naive: {len(rows)} tasks in {(time.time() - t_all) / 60:.1f} min -> {paths.naive_csv}")
    return True


# ================================== twins ===================================

def parse_tag(tag: str) -> tuple[str, int, int | None]:
    m = TAG_RE.match(tag)
    if not m:
        raise SystemExit(f"bad twin tag {tag!r} (expected e.g. gla_s0 or softmax_s1_local8)")
    return m["arm"], int(m["seed"]), int(m["w"]) if m["w"] else None


def twin_checkpoint(runs: Path, arm: str, seed: int) -> Path:
    hits = sorted(runs.glob(f"{arm}_s{seed}_lr*/final.pt"))
    if len(hits) != 1:
        raise SystemExit(f"expected exactly one final.pt for {arm} seed {seed} under {runs}, found {len(hits)}: "
                         f"{[str(h) for h in hits]}")
    return hits[0]


def gpu_tags(gpu: int, local_windows=()) -> list[str]:
    arm = GPU_ARM[gpu]
    tags = [f"{arm}_s{s}" for s in SEEDS]
    if arm == "softmax":
        tags += [f"softmax_s{s}_local{w}" for w in local_windows for s in SEEDS]
    return tags


def load_models(tags, runs: Path, device, gla_backend: str = "fla") -> dict:
    from tidemark.pretrain.inference import load_twin
    from tidemark.pretrain.model import restrict_window
    models = {}
    for tag in tags:
        arm, seed, w = parse_tag(tag)
        model, cfg, ck = load_twin(twin_checkpoint(runs, arm, seed), device, gla_backend)
        if w is not None:
            restrict_window(model, w)
        models[tag] = model
        print(f"loaded {tag}: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
              f"{float(ck.get('points', 0)) / 1e9:.0f}B points", flush=True)
    return models


# ================================== board ===================================

def seed_means(P, ours: list[str]):
    """Adds one column per arm = the mean over its seeds (a task missing a seed is dropped)."""
    arms: dict[str, list[str]] = {}
    for t in ours:
        if t in P:
            a, s, w = parse_tag(t)
            arms.setdefault(a if w is None else f"{a}_local{w}", []).append(t)
    for arm, cols in arms.items():
        if len(cols) == len(SEEDS):
            P[arm] = P[cols].mean(axis=1, skipna=False)
    return P


def ratio_stats(P, a: str, b: str, rng, n_boot: int = 2000, mask=None):
    """Geometric-mean ratio of a to b over tasks with a bootstrap 95% CI over tasks (the reference's
    h2h estimator); None if a column is missing or nothing is left."""
    if a not in P or b not in P:
        return None
    r = (P[a] / P[b]).replace([np.inf, -np.inf], np.nan).dropna()
    r = r[r > 0]
    if mask is not None:
        r = r[[i for i in r.index if mask(i)]]
    if len(r) == 0:
        return None
    lg = np.log(r.values)
    boots = np.exp(lg[rng.integers(0, len(lg), (n_boot, len(lg)))].mean(1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": len(r), "gmean": float(np.exp(lg.mean())), "lo": float(lo), "hi": float(hi),
            "a_better": float((r < 1).mean())}


def h2h_lines(benchmark: str, Ps: dict, a: str, b: str, splits: dict, kind: str, rng, records: list) -> list[str]:
    """One head-to-head block (per metric) with the split by horizon; appends rows to `records`."""
    out = [f"\n{a}  vs  {b}   (ratio < 1 = {a} better; {kind})"]
    for k, P in Ps.items():
        s = ratio_stats(P, a, b, rng)
        if s is None:
            out.append(f"   {k}: not available (missing {a} or {b})")
            continue
        parts = []
        records.append({"benchmark": benchmark, "metric": k, "a": a, "b": b, "kind": kind, "split": "all", **s})
        for name, mask in splits.items():
            g = ratio_stats(P, a, b, rng, mask=mask)
            if g is not None:
                records.append({"benchmark": benchmark, "metric": k, "a": a, "b": b, "kind": kind, "split": name, **g})
                parts.append(f"{name} {g['gmean']:.4f} [{g['lo']:.4f}, {g['hi']:.4f}] (n={g['n']})")
        out.append(f"   {k}: gmean ratio {s['gmean']:.4f} [95% CI {s['lo']:.4f}, {s['hi']:.4f}]  {a} better on "
                   f"{s['a_better']:.0%} of {s['n']} tasks  | " + "  ".join(parts))
    return out


def comparisons(arms_present: list[str]) -> list[tuple[str, str, str]]:
    """The head-to-heads asked for: each arm vs softmax (seeds averaged), seed 0 vs seed 1 per arm
    (the noise floor), and the local-window softmax vs the full one if they were scored."""
    pairs = [(a, "softmax", "seeds averaged") for a in ("gla", "swa16", "swa8")]
    pairs += [(f"{a}_s0", f"{a}_s1", "seed noise floor: seed 0 vs seed 1") for a in ("softmax", "gla", "swa16", "swa8")]
    pairs += [(a, "softmax", "seeds averaged, inference-only window") for a in arms_present if a.startswith("softmax_local")]
    return pairs


def single_seed_comparisons(tags) -> list[tuple[str, str, str]]:
    """Paired head-to-heads for arms that have ONE seed (Stage B), no seed averaging: each tag vs the softmax of the same seed;
    a rerun (<arm>_lr<k>_s<seed>) also vs the softmax rerun of the same rate (if scored) and vs its own original (start-rate
    effect). Arms with both seeds are covered by comparisons()."""
    ok = {}
    for t in tags:
        m = TAG_RE.match(str(t))
        if m and m["w"] is None:
            ok[t] = (m["arm"], int(m["seed"]))
    seeds_of: dict[str, set] = {}
    for arm, sd in ok.values():
        seeds_of.setdefault(arm, set()).add(sd)
    pairs = []
    for t, (arm, sd) in sorted(ok.items(), key=lambda kv: (kv[1][0], kv[1][1])):
        if len(seeds_of[arm]) >= len(SEEDS):
            continue
        base, _, rate = arm.partition("_lr")
        if arm != "softmax" and f"softmax_s{sd}" in ok:
            pairs.append((t, f"softmax_s{sd}", "single seed, paired over tasks" + (" (rerun vs original softmax)" if rate else "")))
        if rate and base != "softmax" and f"softmax_lr{rate}_s{sd}" in ok:
            pairs.append((t, f"softmax_lr{rate}_s{sd}", "single seed, paired over tasks (rerun vs softmax rerun at the same rate)"))
        if rate and f"{base}_s{sd}" in ok:
            pairs.append((t, f"{base}_s{sd}", "single seed, paired over tasks (start-rate effect: rerun vs its own original)"))
    return pairs


def time_board(df, ours: list[str], pick: list[str]):
    """The reference's TIME leaderboard on a frame of per-(model, dataset, horizon) metrics. The
    official models are first restricted to the tasks our models cover (identical to the
    reference when ours cover all of them; makes --smoke meaningful). Returns (board, M, tasks, cov)."""
    import pandas as pd
    from scipy.stats import gmean
    df = df.copy()
    df["task"] = df.dataset_id + "/" + df.horizon
    df = df[df.model.isin(pick + ours)]
    df = df[df.task.isin(set(df[df.model.isin(ours)].task))]
    cov = df.groupby("model").task.nunique()
    full = cov[cov == cov.max()].index.tolist()
    tasks = sorted(set.intersection(*[set(df[df.model == m].task) for m in full]))
    d = df[df.model.isin(full) & df.task.isin(tasks)]
    M = {k: d.pivot_table(index="task", columns="model", values=k) for k in ("MASE", "CRPS")}
    rows = {}
    for k, P in M.items():
        norm = P.div(P["seasonal_naive"], axis=0).replace([np.inf, -np.inf], np.nan)
        rows[f"{k} (norm.)"] = norm.apply(lambda c: gmean(c.dropna()))
        rows[f"{k} rank"] = P.rank(axis=1, method="average").mean()
    board = pd.DataFrame(rows).sort_values("MASE (norm.)")
    return board, M, tasks, cov, full


def fev_summaries(paths: Paths, tasks_filter=None):
    """naive cache + our per-task files + the reference results (restricted to the tasks present),
    as the reference assembles `sums`."""
    import pandas as pd
    if not paths.naive_csv.exists():
        raise SystemExit(f"{paths.naive_csv} is missing: run the `naive` step first")
    rows = pd.read_csv(paths.naive_csv).to_dict("records")
    ours = [json.loads(p.read_text()) for p in sorted(paths.fev_out.glob("*/*.json"))]
    sums = pd.DataFrame(rows + ours)
    if tasks_filter is not None:
        sums = sums[sums.task_name.isin(tasks_filter)]
    notes = []
    for ref in sorted(paths.fev_ref.glob("*.csv")):
        try:
            r = pd.read_csv(ref)
            r = r[r.task_name.isin(sums.task_name.unique())]
            sums = pd.concat([sums, r], ignore_index=True)
            notes.append(f"reference loaded: {ref.name} ({r.model_name.unique().tolist()})")
        except Exception as e:  # noqa: BLE001
            notes.append(f"(reference {ref.name} not loaded: {type(e).__name__})")
    return sums, notes


def horizon_splits(sums, ours: list[str]):
    """Terciles of the task horizons of the tasks all our models share: short / medium / long."""
    tasks = None
    for t in ours:
        s = set(sums[sums.model_name == t].task_name)
        tasks = s if tasks is None else tasks & s
    h = sums[sums.model_name == ours[0]].drop_duplicates("task_name").set_index("task_name").horizon
    h = h[h.index.isin(tasks or set())]
    if len(h) < 3:
        return {}, "too few tasks to split by horizon"
    q1, q2 = np.quantile(h.values, [1 / 3, 2 / 3])
    hz = h.to_dict()
    sp = {f"short(h<={q1:g})": lambda i: hz.get(i, np.inf) <= q1,
          f"medium({q1:g}<h<={q2:g})": lambda i: q1 < hz.get(i, np.nan) <= q2,
          f"long(h>{q2:g})": lambda i: hz.get(i, -np.inf) > q2}
    return sp, f"fev-bench horizon terciles over {len(hz)} tasks: short h<={q1:g}, medium h<={q2:g}, long above"


def long_table(fev_sums, time_df, ours: list[str]):
    """One row per (model, benchmark, metric, task): the table eval/gaps.py reads. window = 'all'
    (fev-bench summaries are averaged over windows already)."""
    import pandas as pd
    rows = []
    for t in ours:
        arm, seed, w = parse_tag(t)
        a = arm if w is None else f"{arm}_local{w}"
        if fev_sums is not None:
            for r in fev_sums[fev_sums.model_name == t].itertuples():
                rows.append((t, a, seed, "fev", "test_error", r.task_name, "all", r.test_error))
        if time_df is not None:
            for r in time_df[time_df.model == t].itertuples():
                for k in ("MASE", "CRPS"):
                    rows.append((t, a, seed, "time", k, f"{r.dataset_id}/{r.horizon}", "all", getattr(r, k)))
    return pd.DataFrame(rows, columns=["model", "arm", "seed", "benchmark", "metric", "task", "window", "err"])


def cmd_board(paths: Paths, smoke: bool = False, tb=None, log=print) -> int:
    import pandas as pd
    lines: list[str] = []
    out = lambda s="": (lines.append(s), log(s))   # noqa: E731
    paths.board.mkdir(parents=True, exist_ok=True)
    rng, records = np.random.default_rng(0), []
    ours_fev = sorted(p.name for p in paths.fev_out.glob("*") if p.is_dir()) if paths.fev_out.exists() else []
    ours_time = sorted(p.name for p in paths.time_out.glob("*") if p.is_dir()) if paths.time_out.exists() else []
    time_df, fev_sums = None, None

    # ------------------------------------------------------------------ TIME
    if ours_time:
        tb = tb or TB(paths)
        LB = tb.leaderboard_module()
        official = json.loads(paths.manifest.read_text())["time_official"]
        pick = ["seasonal_naive"] + [m for m in REFERENCE if m in official]
        theirs = LB.get_all_datasets_results(paths.time_official / "results")
        ours = LB.get_all_datasets_results(paths.time_out)
        ours = ours[ours.model.isin(ours_time)]
        time_df = pd.concat([theirs[theirs.model.isin(pick)], ours], ignore_index=True)
        board, M, tasks, cov, full = time_board(time_df, ours_time, pick)
        out("=" * 100 + f"\nTIME, {len(tasks)} tasks (lower = better; ranks among these {len(full)} models)\n" + "=" * 100)
        out(board.round(3).to_string())
        if len(cov[cov < cov.max()]):
            out("not ranked (fewer tasks): " + ", ".join(f"{m} ({n})" for m, n in cov[cov < cov.max()].items()))
        board.to_csv(paths.board / "time_board.csv")
        seed_means(M["MASE"], ours_time); seed_means(M["CRPS"], ours_time)
        arms = [c for c in M["MASE"].columns if c.startswith("softmax_local")]
        term_splits = {t: (lambda i, t=t: i.endswith("/" + t)) for t in ("short", "medium", "long")}
        out("\n--- TIME head-to-heads (bootstrap over tasks; split by term) ---")
        for a, b, kind in comparisons(arms):
            for line in h2h_lines("time", M, a, b, term_splits, kind, rng, records):
                out(line)
        for a, b, kind in single_seed_comparisons(list(M["MASE"].columns)):
            for line in h2h_lines("time", M, a, b, term_splits, kind, rng, records):
                out(line)
    else:
        out(f"(no TIME results under {paths.time_out})")

    # ------------------------------------------------------------------ fev-bench
    if ours_fev:
        import fev
        sums, notes = fev_summaries(paths)
        for n in notes:
            out(n)
        fev_sums = sums
        common = set.intersection(*[set(sums[sums.model_name == m].task_name) for m in sums.model_name.unique()])
        board = fev.leaderboard(sums, baseline_model="seasonal_naive", n_resamples=1000, missing_strategy="drop")
        out("\n" + "=" * 100 + f"\nfev-bench ({len(common)} tasks all models share; SQL skill score, higher = better; "
            "univariate)\n" + "=" * 100)
        out(board.to_string())
        board.to_csv(paths.board / "fev_board.csv")
        P = sums.pivot_table(index="task_name", columns="model_name", values="test_error")
        seed_means(P, ours_fev)
        splits, note = horizon_splits(sums, ours_fev)
        out("\n--- fev-bench head-to-heads on the official metric (bootstrap over tasks; " + note + ") ---")
        arms = [c for c in P.columns if str(c).startswith("softmax_local")]
        for a, b, kind in comparisons(arms):
            for line in h2h_lines("fev", {"test_error": P}, a, b, splits, kind, rng, records):
                out(line)
        for a, b, kind in single_seed_comparisons(list(P.columns)):
            for line in h2h_lines("fev", {"test_error": P}, a, b, splits, kind, rng, records):
                out(line)
    else:
        out(f"(no fev-bench results under {paths.fev_out})")

    if records:
        pd.DataFrame(records).to_csv(paths.board / "head_to_head.csv", index=False)
    long = long_table(fev_sums, time_df, sorted(set(ours_fev) | set(ours_time)))
    long.to_csv(paths.board / "scores_long.csv", index=False)
    (paths.board / "board.txt").write_text("\n".join(lines))
    out(f"\nwritten to {paths.board}: time_board.csv, fev_board.csv, head_to_head.csv, scores_long.csv "
        f"({len(long)} rows, the table for tidemark.eval.gaps), board.txt")
    return 0


# ================================== fetch ===================================

def check_imports() -> list[str]:
    import importlib
    missing = []
    for m in NEEDED:
        try:
            importlib.import_module(m)
        except ImportError:
            missing.append(m)
    return missing


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as r:
        dest.write_bytes(r.read())


def cmd_fetch(paths: Paths, procs: int = 8, log=print) -> int:
    missing = check_imports()
    if missing:
        pipn = " ".join(PIP_NAMES.get(m, m) for m in missing)
        log(f"missing Python packages: {missing}\nrun on the login node:  pip install --user {pipn}")
        return 1
    from huggingface_hub import HfApi, snapshot_download
    manifest = read_manifest(paths)
    # 1. the TIME repo (timebench + the leaderboard script)
    if not (paths.time_repo / "src").exists():
        subprocess.run(["git", "clone", "-q", "--depth", "1", TIME_REPO_URL, str(paths.time_repo)], check=True)
    log(f"TIME repo: {paths.time_repo}")
    # 2. the TIME datasets
    if not any(paths.time_data.rglob("dataset_info.json")):
        log("downloading the TIME datasets (Real-TSF/TIME) ...")
        snapshot_download("Real-TSF/TIME", repo_type="dataset", local_dir=str(paths.time_data))
    sizes = {}
    for ds_info in paths.time_data.rglob("dataset_info.json"):
        sizes[ds_info.parent.name] = sum(f.stat().st_size for f in ds_info.parent.rglob("*") if f.is_file())
    manifest["time_sizes"] = sizes
    # 3. the official TIME results of seasonal naive and the reference models
    official = sorted({f.path.split("/")[1] for f in HfApi().list_repo_tree("Real-TSF/TIME-Output", repo_type="dataset",
                                                                           path_in_repo="results")})
    pick = ["seasonal_naive"] + [m for m in REFERENCE if m in official]
    snapshot_download("Real-TSF/TIME-Output", repo_type="dataset", local_dir=str(paths.time_official),
                      allow_patterns=[f"results/{m}/**/{f}" for m in pick for f in ("metrics.npz", "config.json")])
    manifest["time_official"] = official
    log(f"TIME official results: {pick}")
    # 4. the fev-bench tasks, their data, and the reference results
    _download(FEV_YAML, paths.fev_yaml)
    tasks = load_fev_tasks(paths)
    rec, failed = [], []
    for i, task in enumerate(tasks, 1):
        try:
            ds = task.load_full_dataset(num_proc=procs)
            rec.append({"task_name": task.task_name, "horizon": task.horizon, "num_windows": task.num_windows,
                        "items": len(ds)})
            log(f"[{i}/{len(tasks)}] fev task {task.task_name:45s} {len(ds)} series")
        except Exception as e:  # noqa: BLE001
            failed.append(task.task_name)
            log(f"[{i}/{len(tasks)}] fev task {task.task_name} FAILED: {type(e).__name__}: {str(e)[:150]}")
    manifest["fev"] = rec
    for name, f in FEV_REFERENCE_FILES.items():
        try:
            _download(FEV_RAW + "results/" + f, paths.fev_ref / f)
        except Exception as e:  # noqa: BLE001
            log(f"fev reference {name} ({f}) not downloaded: {type(e).__name__}")
    write_json(paths.manifest, manifest)
    log(f"fetched: {len(rec)} fev tasks ({len(failed)} failed), {len(sizes)} TIME datasets, caches under {paths.hf}")
    log("on the compute node nothing touches the network (HF_HUB_OFFLINE / HF_DATASETS_OFFLINE are set)")
    return 1 if failed else 0


# =================================== CLI ====================================

def cmd_twins(a, paths: Paths) -> int:
    import torch
    from tidemark.pretrain.inference import forecast_left_padded
    if a.only:
        tags = [t.strip() for t in a.only.split(",") if t.strip()]
    elif a.gpu is not None:
        tags = gpu_tags(a.gpu, a.local_window)
    else:
        raise SystemExit("give --gpu G (0..3) or --only TAG[,TAG]")
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    models = load_models(tags, Path(a.runs) if a.runs else paths.runs, device, a.gla_backend)
    if not a.skip_time:
        run_time(models, paths, device, smoke=a.smoke, forecast=forecast_left_padded)
    if not a.skip_fev:
        tasks = load_fev_tasks(paths)
        tasks = smoke_fev(tasks, read_manifest(paths)) if a.smoke else tasks
        run_fev(models, tasks, paths, device, 4 if a.num_proc is None else a.num_proc, forecast=forecast_left_padded)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("fetch", "naive", "twins", "board"))
    ap.add_argument("--root", default=default_root(), help="run root (default $ROOT or $SCRATCH/tidemark)")
    ap.add_argument("--smoke", action="store_true", help="3 fev tasks, 2 TIME datasets, outputs in $ROOT/eval_smoke")
    ap.add_argument("--workers", type=int, default=100, help="naive: processes (default 100)")
    ap.add_argument("--gpu", type=int, choices=sorted(GPU_ARM), help="twins: 0 softmax, 1 GLA, 2 SWA16, 3 SWA8")
    ap.add_argument("--only", default=None, help="twins: tags instead of --gpu, e.g. gla_s0 (smoke: gla_s0)")
    ap.add_argument("--runs", default=None, help="twins: folder with <arm>_s<seed>_lr*/final.pt (default $ROOT/runs)")
    ap.add_argument("--local-window", type=int, nargs="*", default=[],
                    help="twins on --gpu 0: also score the softmax twins limited to this many patches at inference")
    ap.add_argument("--skip-time", action="store_true")
    ap.add_argument("--skip-fev", action="store_true")
    ap.add_argument("--num-proc", type=int, default=None, help="processes for dataset preprocessing per task "
                    "(fetch: default 8; twins: default 4; 0 = in-process)")
    ap.add_argument("--gla-backend", default="fla", choices=("fla", "reference"))
    ap.add_argument("--cpu", action="store_true")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    setup_env(a.root, offline=a.cmd != "fetch")
    paths = Paths(a.root, smoke=a.smoke)
    if a.cmd == "fetch":
        return cmd_fetch(paths, 8 if a.num_proc is None else a.num_proc)
    if a.cmd == "naive":
        return 0 if run_naive(paths, a.workers, a.smoke) else 1
    if a.cmd == "twins":
        if a.smoke and not a.only:
            a.only = SMOKE_TWIN
        return cmd_twins(a, paths)
    return cmd_board(paths, a.smoke)


if __name__ == "__main__":
    sys.exit(main())
