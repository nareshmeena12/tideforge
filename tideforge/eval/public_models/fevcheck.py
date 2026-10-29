"""Wrapper check (E0.5): each adapter reproduces its model's published fev-bench per-task numbers on 5 fixed tasks within 2%.

    python -m tideforge.eval.public_models.fevcheck run   --model chronos2 --root $TF_ROOT/eval_t0 --fev $TF_ROOT/ext/fev
                                                          [--out $TF_ROOT/public/fevcheck] [--device cuda]
    python -m tideforge.eval.public_models.fevcheck table --out $TF_ROOT/public/fevcheck [--looks]

run: the 5 TASKS through fev itself (tidemark's task yaml and data cache under --root, as the ft3 scorer uses; offline), our
adapter's 9 quantiles per window -> task.evaluation_summary -> test_error (SQL, fev-bench's metric) and MASE per task, against
the same task's row in fev's results CSV (--fev = the fev git clone, benchmarks/fev_bench/results/<model>.csv). Pass: every
task |ours / published - 1| <= 2% on test_error. Writes <out>/<model>.json. Covariates: past and known columns are passed to
models that take them (categoricals coded as in score_groups.encode_column); others get the targets only (as their published
runs did). Contexts are not filled: NaN reaches the model, which handles it.
table: one line per model; --looks adds one LOOKS.md row per model (a counted fev look, EXPERIMENTS.md rule 5).

The published rows were made with fev 0.8.0 (Chronos-2, TimesFM-2.5, TiRex, TiRex-2), 0.9.0 (Toto-2.0) and 0.11.0.dev0
(Timestral). The json records the installed fev version next to the CSV's; a version difference is reported, never ignored.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.eval import looks, thirdparty
from tideforge.eval.public_models import SPECS, make
from tideforge.probes.common import Cell

# fixed 2026-10-29, before any run: covariates univariate (epf_de: known), covariates multi-target (uci_air_quality_1D),
# multivariate without covariates (ETT_1W), univariate short (australian_tourism), seasonal daily with H = 28 (ercot_1D)
TASKS = ("epf_de", "uci_air_quality_1D", "ETT_1W", "australian_tourism", "ercot_1D")
TOL = 0.02
EXPERIMENT = "E0.5"


def published(fev_repo: str | Path, model: str) -> pd.DataFrame:
    p = Path(fev_repo) / "benchmarks" / "fev_bench" / "results" / f"{SPECS[model].fev_csv}.csv"
    if not p.exists():
        raise SystemExit(f"{p} missing: clone fev into --fev (git clone https://github.com/autogluon/fev $TF_ROOT/ext/fev)")
    return pd.read_csv(p).set_index("task_name")


def window_items(window, use_cov: bool) -> list[dict]:
    """One item per series of a fev window: targets (k, L) over the target columns; past and known lanes when use_cov."""
    G = thirdparty.score_groups()
    past, future = window.get_input_data()
    tg = {c: [np.asarray(y, np.float64) for y in past[c]] for c in window.target_columns}
    n = len(tg[window.target_columns[0]])
    ps = {c: G.encode_column(list(past[c])) for c in window.past_dynamic_columns} if use_cov else {}
    kc = {c: G.encode_column(list(past[c]) + list(future[c])) for c in window.known_dynamic_columns} if use_cov else {}
    out = []
    for i in range(n):
        out.append(dict(targets=[tg[c][i] for c in window.target_columns],
                        past=[np.asarray(ps[c][i], np.float64) for c in ps],
                        known=[(np.asarray(kc[c][i], np.float64), np.asarray(kc[c][n + i], np.float64)) for c in kc]))
    return out


def cells(items: list[dict], H: int, S: int, cap: int, mode: str, batch: int = 256):
    """Items of equal (capped) context length as Cells; yields (indices, cell)."""
    by: dict = {}
    for j, it in enumerate(items):
        by.setdefault(min(cap, len(it["targets"][0])), []).append(j)
    for L, idx in by.items():
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            a = {"targets": np.stack([np.stack([t[-L:] for t in items[j]["targets"]]) for j in b]).astype(np.float32),
                 "future": np.zeros((len(b), len(items[b[0]]["targets"]), H))}
            if mode == "on" and items[b[0]]["past"]:
                a["past"] = np.stack([np.stack([p[-L:] for p in items[j]["past"]]) for j in b]).astype(np.float32)
            if mode == "on" and items[b[0]]["known"]:
                a["known_ctx"] = np.stack([np.stack([c[-L:] for c, _ in items[j]["known"]]) for j in b]).astype(np.float32)
                a["known_fut"] = np.stack([np.stack([f[:H] for _, f in items[j]["known"]]) for j in b]).astype(np.float32)
            yield b, Cell("fev", a, S, modes=(mode,))


def task_predictions(model, task) -> list[dict]:
    """fev predictions per window: {target column: [{"predictions": median, "0.1": ..., ...} per series]}."""
    q_keys = [str(q) for q in task.quantile_levels]
    if len(q_keys) != 9:
        raise ValueError(f"{task.task_name}: {len(q_keys)} quantile levels, the adapters give 9")
    cov = getattr(model, "covariates", "none") != "none"
    out = []
    for w in task.iter_windows():
        mode = "on" if cov and (w.past_dynamic_columns or w.known_dynamic_columns) else "hidden"
        items = window_items(w, mode == "on")
        q = [None] * len(items)
        for idx, cell in cells(items, task.horizon, int(task.seasonality), model.max_context, mode):
            for j, qj in zip(idx, model.predict_cell(cell, mode)):
                q[j] = qj                                                    # (k, 9, H)
        d = {}
        for ci, col in enumerate(w.target_columns):
            d[col] = [{"predictions": q[j][ci, 4].tolist(), **{k: q[j][ci, i].tolist() for i, k in enumerate(q_keys)}}
                      for j in range(len(items))]
        out.append(d)
    return out


def compare(ours: dict, pub: pd.DataFrame) -> dict:
    rows, ok = [], True
    for t, r in ours.items():
        p = pub.loc[t]
        dev = r["test_error"] / float(p["test_error"]) - 1
        ok &= abs(dev) <= TOL
        rows.append(dict(task=t, ours=r["test_error"], published=float(p["test_error"]), rel=dev,
                         MASE=r.get("MASE"), MASE_published=float(p["MASE"]) if "MASE" in p else None, pass_=abs(dev) <= TOL))
    return dict(rows=rows, passed=bool(ok and len(rows) == len(TASKS)))


def cmd_run(a) -> int:
    import fev
    S_ = thirdparty.score_twins()
    S_.setup_env(a.root, offline=True)
    pub = published(a.fev, a.model)
    tasks = {t.task_name: t for t in fev.Benchmark.from_yaml(S_.Paths(a.root).fev_yaml).tasks}
    missing = [t for t in TASKS if t not in tasks or t not in pub.index]
    if missing:
        raise SystemExit(f"tasks missing from the yaml or {SPECS[a.model].fev_csv}.csv: {missing}")
    model = make(a.model, a.device)
    ours = {}
    for name in TASKS:
        task = tasks[name]
        S_.prepare(task, 0)
        s = task.evaluation_summary(task_predictions(model, task), model_name=a.model, trained_on_this_dataset=False)
        ours[name] = {k: float(s[k]) for k in ("test_error", "MASE", "SQL") if k in s}
        print(f"{a.model} {name}: test_error {ours[name]['test_error']:.4f} (published {float(pub.loc[name, 'test_error']):.4f})",
              flush=True)
    res = compare(ours, pub)
    res.update(model=a.model, spec={k: v for k, v in SPECS[a.model].__dict__.items() if k != "kwargs"},
               fev_installed=fev.__version__, fev_published=str(pub["fev_version"].iloc[0]), tol=TOL, tasks=list(TASKS))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.model}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(summary_line(res))
    return 0 if res["passed"] else 1


def summary_line(res: dict) -> str:
    worst = max(res["rows"], key=lambda r: abs(r["rel"]))
    ver = "" if res["fev_installed"] == res["fev_published"] else f"; fev {res['fev_installed']} vs published {res['fev_published']}"
    return (f"{res['model']}: {'PASS' if res['passed'] else 'FAIL'} ({sum(r['pass_'] for r in res['rows'])}/{len(TASKS)} tasks "
            f"within {100 * res['tol']:.0f}%, worst {worst['task']} {100 * worst['rel']:+.2f}%{ver})")


def cmd_table(a) -> int:
    res = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(a.out).glob("*.json"))]
    if not res:
        raise SystemExit(f"no results in {a.out}")
    for r in res:
        print(summary_line(r))
        for row in r["rows"]:
            print(f"    {row['task']:22s} ours {row['ours']:.4f}  published {row['published']:.4f}  {100 * row['rel']:+.2f}%")
    if a.looks:
        rows = [looks.format_row(experiment=EXPERIMENT, checkpoint=r["model"], benchmark=f"fev-bench ({len(TASKS)} tasks)",
                                 why="wrapper check: published per-task numbers within 2%", result=summary_line(r))
                for r in res]
        added = looks.append_looks(rows, a.looks_file)
        print(f"{len(added)} LOOKS row(s) added to {a.looks_file}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("run", "table"))
    ap.add_argument("--model", choices=sorted(SPECS))
    ap.add_argument("--root")
    ap.add_argument("--fev")
    ap.add_argument("--out", default="runs/public/fevcheck")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--looks", action="store_true")
    ap.add_argument("--looks-file", default=str(looks.LOOKS))
    a = ap.parse_args(argv)
    if a.cmd == "run" and not (a.model and a.root and a.fev):
        raise SystemExit("run needs --model, --root and --fev")
    return {"run": cmd_run, "table": cmd_table}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
