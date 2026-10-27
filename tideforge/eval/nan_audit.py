"""E0.3 audit: how many context values are NaN, per fev-bench task and TIME dataset / term, before and after fill_season, i.e.
what still reaches ft3_core.forecast_left_padded (which deletes NaNs). No model, CPU only.

  python -m tideforge.eval.nan_audit --root R [--smoke] [--out R/eval/nan_audit.csv] [--looks]

Series are cut exactly as the scorers cut them: the last MAX_CTX (8192) points of each target, then score_twins.fill_season with
the task's seasonality. Groups ON (fev only): the past / known dynamic covariates after score_groups.encode_column (non-numbers
coded, missing -> NaN), last MAX_CTX points; forecast_groups forward-fills those NaNs (_lane_array, known future).
Columns per row:
  n_series, n_values                     target series and their context values
  nan_leading, nan_interior              NaNs before fill: before the first finite value / after it
  series_interior_nan                    series with at least one interior NaN (the phase-shift risk without fill)
  nan_after_fill, all_nan_series         NaNs still present after fill_season (= deleted by forecast_left_padded), all-NaN series
  past_cov_values, past_cov_nan          groups ON: past covariates
  known_ctx_values, known_ctx_nan        groups ON: known covariates, context
  known_fut_values, known_fut_nan        groups ON: known covariates, future (forecast horizon)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from tideforge.eval import looks, thirdparty

TARGET_COLS = ("n_series", "n_values", "nan_leading", "nan_interior", "series_interior_nan", "nan_after_fill", "all_nan_series")
COV_COLS = ("past_cov_values", "past_cov_nan", "known_ctx_values", "known_ctx_nan", "known_fut_values", "known_fut_nan")


def target_counts(series, S: int, max_ctx: int, fill) -> dict:
    """Counts over a list of raw 1-D target histories."""
    c = dict.fromkeys(TARGET_COLS, 0)
    for a in series:
        y = np.asarray(a, dtype=np.float64)[-max_ctx:]
        ok = np.isfinite(y)
        lead = int(np.argmax(ok)) if ok.any() else len(y)
        c["n_series"] += 1
        c["n_values"] += len(y)
        c["nan_leading"] += lead
        inner = int((~ok[lead:]).sum())
        c["nan_interior"] += inner
        c["series_interior_nan"] += int(inner > 0)
        f = np.asarray(fill(y, S))
        c["nan_after_fill"] += int((~np.isfinite(f)).sum())
        c["all_nan_series"] += int(not ok.any())
    return c


def _nan(arrs) -> tuple[int, int]:
    arrs = [np.asarray(a, dtype=np.float64) for a in arrs]
    return sum(a.size for a in arrs), sum(int((~np.isfinite(a)).sum()) for a in arrs)


def fev_task_counts(task, max_ctx: int, fill, encode) -> dict:
    S = int(task.seasonality)
    c = dict.fromkeys(TARGET_COLS + COV_COLS, 0)
    n_win = 0
    for w in task.iter_windows():
        n_win += 1
        past, future = w.get_input_data()
        for col in task.target_columns:
            for k, v in target_counts(list(past[col]), S, max_ctx, fill).items():
                c[k] += v
        for col in w.past_dynamic_columns:
            n, m = _nan([a[-max_ctx:] for a in encode(list(past[col]))])
            c["past_cov_values"] += n; c["past_cov_nan"] += m
        for col in w.known_dynamic_columns:
            enc = encode(list(past[col]) + list(future[col]))
            k = len(past[col])
            n, m = _nan([a[-max_ctx:] for a in enc[:k]])
            c["known_ctx_values"] += n; c["known_ctx_nan"] += m
            n, m = _nan(enc[k:])
            c["known_fut_values"] += n; c["known_fut_nan"] += m
    return dict(n_windows=n_win, **c)


def time_term_counts(inputs, S: int, max_ctx: int, fill) -> dict:
    series = []
    for d in inputs:
        a = np.asarray(d["target"])
        series += list(a[None] if a.ndim == 1 else a)
    return dict(n_windows=len(inputs), **target_counts(series, S, max_ctx, fill), **dict.fromkeys(COV_COLS, 0))


def run(root, smoke: bool, out: Path, skip_fev: bool, skip_time: bool, num_proc: int, log=print):
    import pandas as pd
    S_ = thirdparty.score_twins()
    G = thirdparty.score_groups()
    S_.setup_env(root, offline=True)
    paths = S_.Paths(root, smoke=smoke)
    rows = []
    if not skip_time:
        tb = S_.TB(paths)
        config = tb.load_dataset_config()
        names = list(config["datasets"])
        names = S_.smoke_time(names, S_.read_manifest(paths)) if smoke else names
        for ds_name in names:
            for term in tb.get_available_terms(ds_name, config):
                t0 = time.time()
                try:
                    st = tb.get_dataset_settings(ds_name, term, config)
                    ds = tb.Dataset(name=ds_name, term=term, to_univariate=False, prediction_length=st["prediction_length"],
                                    test_length=st["test_length"], val_length=st["val_length"])
                    Sx = tb.get_seasonality(ds.freq)
                    r = time_term_counts(list(ds.test_data.input), Sx, S_.MAX_CTX, S_.fill_season)
                    rows.append(dict(benchmark="TIME", task=f"{ds_name}/{term}", seasonality=Sx, D=ds.target_dim, **r))
                    log(f"TIME {ds_name:32s} {term:6s} interior NaN {r['nan_interior']:>9d}  after fill {r['nan_after_fill']:>7d}"
                        f"  ({time.time() - t0:.1f}s)")
                except Exception as e:  # noqa: BLE001
                    log(f"TIME {ds_name} {term} FAILED: {type(e).__name__}: {str(e)[:160]}")
                    rows.append(dict(benchmark="TIME", task=f"{ds_name}/{term}", error=f"{type(e).__name__}: {e}"[:300]))
    if not skip_fev:
        tasks = S_.load_fev_tasks(paths)
        tasks = S_.smoke_fev(tasks, S_.read_manifest(paths)) if smoke else tasks
        for task in tasks:
            t0 = time.time()
            try:
                S_.prepare(task, num_proc)
                r = fev_task_counts(task, S_.MAX_CTX, S_.fill_season, G.encode_column)
                rows.append(dict(benchmark="fev", task=task.task_name, seasonality=int(task.seasonality), **r))
                log(f"fev  {task.task_name:45s} interior NaN {r['nan_interior']:>9d}  after fill {r['nan_after_fill']:>7d}"
                    f"  cov NaN {r['past_cov_nan'] + r['known_ctx_nan'] + r['known_fut_nan']:>8d}  ({time.time() - t0:.1f}s)")
            except Exception as e:  # noqa: BLE001
                log(f"fev  {task.task_name} FAILED: {type(e).__name__}: {str(e)[:160]}")
                rows.append(dict(benchmark="fev", task=task.task_name, error=f"{type(e).__name__}: {e}"[:300]))
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return df


def summary(df) -> list[str]:
    out = []
    for b, g in df.groupby("benchmark"):
        ok = g[g["error"].isna()] if "error" in g else g
        failed = len(g) - len(ok)
        out.append(f"{b}: {len(ok)} tasks ok, {failed} failed; tasks with interior NaN {int((ok.nan_interior > 0).sum())}; "
                   f"interior NaN {int(ok.nan_interior.sum())} of {int(ok.n_values.sum())} values; "
                   f"NaN after fill {int(ok.nan_after_fill.sum())} in {int((ok.nan_after_fill > 0).sum())} tasks; "
                   f"all-NaN series {int(ok.all_nan_series.sum())}; covariate NaN (ON) "
                   f"{int(ok.past_cov_nan.sum() + ok.known_ctx_nan.sum() + ok.known_fut_nan.sum())}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None, help="default: ROOT/eval/nan_audit.csv (ROOT/eval_smoke/... with --smoke)")
    ap.add_argument("--skip-fev", action="store_true")
    ap.add_argument("--skip-time", action="store_true")
    ap.add_argument("--num-proc", type=int, default=4)
    ap.add_argument("--looks", action="store_true", help="add the 'data only, no model scored' row to LOOKS.md")
    ap.add_argument("--looks-file", default=str(looks.LOOKS))
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else Path(a.root) / ("eval_smoke" if a.smoke else "eval") / "nan_audit.csv"
    df = run(a.root, a.smoke, out, a.skip_fev, a.skip_time, a.num_proc)
    lines = summary(df)
    print("\n".join(lines))
    print(f"written {out}")
    if a.looks and not a.smoke:
        bench = " + ".join(x for x, s in (("fev-bench", a.skip_fev), ("TIME", a.skip_time)) if not s)
        row = looks.format_row(experiment="E0.3", checkpoint="none (data only, no model scored)", benchmark=bench,
                               why="NaN audit of inference contexts", result="; ".join(lines))
        added = looks.append_looks([row], a.looks_file)
        print(f"{len(added)} row(s) added to {a.looks_file}:", *added, sep="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
