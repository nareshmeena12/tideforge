"""tfbench scorer CLI (one GPU; also fev-bench through the same pipeline).

  python -m tideforge.score.run fetch --version v1.1.0                    # login node: release -> $TF_ROOT/tfbench_release/v1.1.0
  python -m tideforge.score.run fetch --fev-bench --eval-root R           # login node: fev-bench data into R's HF cache
  python -m tideforge.score.run score --model <ckpt.pt|seasonal_naive> --tier fast|core|full [--locked] [--version v1.1.0]
                                      [--groups auto|on|off|both] [--batch-size N] [--model-id ID] [--out DIR]
  python -m tideforge.score.run score --model <ckpt.pt> --fev-bench --eval-root R [--groups both]
  python -m tideforge.score.run compare DIR [DIR ...] [--out compare.md]
  python -m tideforge.score.run profile ...                                 # = python -m tideforge.score.profile_fev

Tuning split by default. --locked scores the locked split (final reporting only, never for selection): the run is added to
results/tfbench/locked_ledger.csv (commit it) and to $TF_ROOT/looks_pending.md. fev-bench runs also add a looks row.
--groups auto: ON for models with group / lane attention (ft3_main, cov_r3, R1 runs), OFF for univariate ones; both = ON and
OFF in one run (one window cache). Output: <out>/<model_id>/<version>/<tier>_<split>/groups_<on|off>/: tasks.csv (every
metric, seasonal naive, relative scores, all tags), failure_map.csv, summary.md, stats.json; predictions under preds/
(resumable).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from tideforge.score import release as RL

TF = os.environ.get("TF_ROOT")


def default_out() -> Path:
    return Path(TF) / "results" / "tfbench" if TF else RL.REPO / "results" / "tfbench"


def model_id_of(model: str) -> str:
    p = Path(model)
    if p.suffix in (".pt", ".pth"):
        return p.stem if p.parent.name in ("", ".") else f"{p.parent.name}_{p.stem}"
    return model


def peak_rss_gb() -> float | None:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20           # Linux: KiB
    except ImportError:
        return None


def looks_row(a, mid: str, bench: str, result: str) -> None:
    from tideforge.eval import looks
    row = looks.format_row(experiment=a.experiment, checkpoint=mid, benchmark=bench, why=a.why, result=result)
    try:
        looks.append_looks([row])
    except SystemExit as e:                     # no TF_ROOT (a local test run): say so, do not fail the scoring
        print(f"looks row not written ({e}): {row}", flush=True)


def fevbench_board(eval_root: Path, caches, model_scores: dict, naive_scores: dict, name: str) -> tuple[str, dict]:
    """fev.leaderboard on our summaries, as the old board computed the fev skill: naive + model (+ the reference CSVs the
    old board loaded, restricted to these tasks; missing_strategy drop)."""
    import fev
    from tideforge.score.metrics import fev_summary
    rows = [fev_summary(c, naive_scores[c.name], "seasonal_naive") for c in caches]
    rows += [fev_summary(c, model_scores[c.name], name) for c in caches]
    sums = pd.DataFrame(rows)
    out, res = [], {}
    b = fev.leaderboard(sums, baseline_model="seasonal_naive", missing_strategy="drop")
    res["skill (model + naive)"] = float(b.loc[name, "skill_score"])
    refs = sorted((Path(eval_root) / "eval" / "data" / "fev_reference").glob("*.csv"))
    if refs:
        r = pd.concat([pd.read_csv(f) for f in refs], ignore_index=True)
        r = r[r["task_name"].isin(sums["task_name"].unique())]
        b2 = fev.leaderboard(pd.concat([sums, r], ignore_index=True), baseline_model="seasonal_naive", missing_strategy="drop")
        res["skill (with the reference models, as the old board)"] = float(b2.loc[name, "skill_score"])
        out += ["## fev.leaderboard with the reference models (the old board's setting)", b2.round(4).to_string(), ""]
    out = ["## fev.leaderboard (model + seasonal naive)", b.round(4).to_string(), ""] + out
    return "\n".join(out), res


def cmd_score(a) -> int:
    t_start = time.time()
    from tideforge.score import adapters as AD
    from tideforge.score import aggregate as AG
    from tideforge.score import metrics as MT
    from tideforge.score import predict as PR
    from tideforge.score import windows as W
    if a.fev_bench:
        if not a.eval_root:
            raise SystemExit("--fev-bench needs --eval-root (the old scorer's evaluation root with the fev-bench data)")
        from tideforge.eval import thirdparty
        thirdparty.score_twins().setup_env(a.eval_root, offline=True)
        specs = RL.fevbench_tasks(Path(a.eval_root))
        version, tier, split = "fev-bench", "fev-bench", "fev-bench"
        cache_root = Path(a.eval_root) / "tfbench_cache" / "fevbench"
        naive_kind = "fev"
    else:
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        split = "locked" if a.locked else "tuning"
        rel = RL.Release(Path(a.release_root or RL.default_release_root()) / a.version)
        specs = rel.tasks(a.tier, split, locked=a.locked)
        version, tier = a.version, a.tier
        cache_root = rel.path / "_cache" / f"{tier}_{split}"
        naive_kind = "tfbench"
    if a.limit:
        specs = specs[:a.limit]
    print(f"{len(specs)} tasks ({version}, {tier}, {split})", flush=True)
    t0 = time.time()
    caches = W.build(specs, cache_root, ctx=a.ctx, workers=a.workers, log=lambda s: print(s, flush=True))
    t_cache = time.time() - t0
    cache_bytes = sum(c.info["bytes"] for c in caches)
    mid = a.model_id or model_id_of(a.model)
    pred = AD.get(a.model, device=a.device, batch_size=a.batch_size, naive_kind=naive_kind)
    pred.name = mid
    modes = {"auto": ["on" if pred.groups else "off"], "on": ["on"], "off": ["off"], "both": ["on", "off"]}[a.groups]
    if not pred.groups:
        modes = ["off"]                       # a univariate model: ON is the same forecast
    out_root = Path(a.out or default_out())
    base = out_root / mid / version / f"{tier}_{split}"
    from tideforge.score.adapters.naive import SeasonalNaive
    nv = SeasonalNaive(naive_kind)
    qn, _ = PR.run(caches, nv, "off", out_root / "_naive" / version / f"{tier}_{split}_{naive_kind}", f"naive_{naive_kind}",
                   log=lambda s: None)
    naive = {c.name: MT.score_task(c, qn[c.name]) for c in caches}
    summaries = {}
    for mode in modes:
        d = base / f"groups_{mode}"
        d.mkdir(parents=True, exist_ok=True)
        t1 = time.time()
        q, st = PR.run(caches, pred, mode, d, mid, log=lambda s: print(s, flush=True))
        t_pred = time.time() - t1
        sc = {c.name: MT.score_task(c, q[c.name]) for c in caches}
        df = AG.task_table(specs, caches, sc, naive, mode)
        df.to_csv(d / "tasks.csv", index=False)
        AG.failure_map(df).to_csv(d / "failure_map.csv", index=False)
        n_win = int(sum(c.info["n_rows"] for c in caches))
        stats = dict(model=a.model, model_id=mid, version=version, tier=tier, split=split, groups=mode, tasks=len(caches),
                     windows=n_win, series=st["series"], model_s=round(st["model_s"], 1), prep_s=round(st["prep_s"], 1),
                     predict_wall_s=round(t_pred, 1), cache_s=round(t_cache, 1), cache_gib=round(cache_bytes / 2 ** 30, 3),
                     windows_per_s=round(n_win / max(t_pred, 1e-9), 1), series_per_s_model=round(
                         st["series"] / max(st["model_s"], 1e-9), 1), shards_reused=st["reused"], peak_rss_gib=peak_rss_gb(),
                     batch_tokens=getattr(pred, "_tok", None), total_s=round(time.time() - t_start, 1))
        extra = ""
        if a.fev_bench:
            extra, res = fevbench_board(Path(a.eval_root), caches, sc, naive, mid)
            stats.update(res)
        (d / "stats.json").write_text(json.dumps(stats, indent=1, default=str), encoding="utf-8")
        md = AG.summary_md(df, f"tfbench {version} {tier} {split}: {mid}, groups {mode.upper()}", stats, a.n_boot, a.seed, extra)
        (d / "summary.md").write_text(md, encoding="utf-8")
        s, used, dropped = AG.skill(df)
        summaries[mode] = (s, df)
        print(f"{mid} groups {mode.upper()}: skill {100 * s:.2f} on {used} tasks ({dropped} not finite), "
              f"{stats['windows_per_s']} windows/s -> {d}", flush=True)
        if a.fev_bench:
            print("  " + ", ".join(f"{k}: {100 * v:.2f}" for k, v in res.items()), flush=True)
    if a.locked or a.fev_bench:
        for mode, (s, df) in summaries.items():
            bench = "fev-bench" if a.fev_bench else f"tfbench {version} locked {tier}"
            looks_row(a, f"{mid} (groups {mode.upper()})", bench, f"skill {100 * s:.2f} on {len(df)} tasks")
            if a.locked:
                RL.ledger_append(dict(model_id=mid, model=a.model, version=version, tier=tier, groups=mode, n_tasks=len(df),
                                      skill_MASE=round(100 * AG.skill(df, "MASE")[0], 3),
                                      skill_WQL=round(100 * AG.skill(df, "WQL")[0], 3), out=str(base)))
    return 0


def cmd_fetch(a) -> int:
    if a.fev_bench:
        if not a.eval_root:
            raise SystemExit("--fev-bench needs --eval-root")
        from tideforge.eval import thirdparty
        thirdparty.score_twins().setup_env(a.eval_root, offline=False)
        import fev
        specs = RL.fevbench_tasks(Path(a.eval_root))
        for i, s in enumerate(specs):
            fev.Task(**s["kw"]).load_full_dataset(num_proc=4)
            print(f"[{i + 1}/{len(specs)}] {s['kw']['task_name']} cached", flush=True)
        return 0
    p = RL.fetch(a.version, Path(a.release_root) if a.release_root else None)
    print(f"release {a.version} -> {p}", flush=True)
    return 0


def cmd_compare(a) -> int:
    from tideforge.score import aggregate as AG
    md = AG.compare([Path(d) for d in a.dirs], a.n_boot, a.seed)
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8")
    print(md)
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["profile"]:
        from tideforge.score import profile_fev
        return profile_fev.main(argv[1:])
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--version", default="v1.1.0")
    f.add_argument("--release-root")
    f.add_argument("--fev-bench", action="store_true")
    f.add_argument("--eval-root")
    s = sub.add_parser("score")
    s.add_argument("--model", required=True, help="checkpoint path, or seasonal_naive")
    s.add_argument("--model-id")
    s.add_argument("--tier", default="fast", choices=RL.TIERS)
    s.add_argument("--locked", action="store_true", help="score the LOCKED split (final reporting only; ledger + looks row)")
    s.add_argument("--version", default="v1.1.0")
    s.add_argument("--release-root")
    s.add_argument("--fev-bench", action="store_true", help="score fev-bench (any fev task list) instead of tfbench")
    s.add_argument("--eval-root", help="--fev-bench: the old scorer's evaluation root (data, HF cache, reference CSVs)")
    s.add_argument("--groups", default="auto", choices=["auto", "on", "off", "both"])
    s.add_argument("--batch-size", type=int, help="series (univariate) / rows (groups) per forward; default from free GPU memory")
    s.add_argument("--device")
    s.add_argument("--ctx", type=int, default=8192)
    s.add_argument("--workers", type=int, default=8, help="processes building the window cache")
    s.add_argument("--limit", type=int, default=0, help="first N tasks only (smoke)")
    s.add_argument("--out")
    s.add_argument("--n-boot", type=int, default=1000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--experiment", default="tfbench")
    s.add_argument("--why", default="tfbench scorer run")
    c = sub.add_parser("compare")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--out")
    c.add_argument("--n-boot", type=int, default=1000)
    c.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return {"fetch": cmd_fetch, "score": cmd_score, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
