"""Profile the current fev scoring path (tidemark score_groups / score_twins, as tideforge.eval.score runs it for ft3_main and
R1) without changing it: the same functions are called in the same order, each wrapped in a timer (CUDA synchronised around
model calls).

    python -m tideforge.score.profile_fev --root $TF_ROOT/eval_t0 --ckpt $FT3_MAIN [--modes on,off] [--tasks 20] [--out DIR]

Per task and mode it records: data (task.load_full_dataset), windows (iter_windows + get_input_data + fill_season, i.e.
fev_inputs / window_groups), model (forecast_left_padded / forecast_groups), metrics (task.evaluation_summary), the number
of windows, series and model calls, and the rest (per-task overhead). Writes profile.csv and profile.md to --out. The
data must already be in the eval root's HF cache (the old scorer's fetch); it runs offline.
"""
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import pandas as pd

from tideforge.eval import thirdparty
from tideforge.tfbench.report import _md

PARTS = ("data", "windows", "model", "metrics")


class Clock:
    def __init__(self, cuda: bool):
        self.cuda, self.t, self.calls = cuda, dict.fromkeys(PARTS, 0.0), 0

    def sync(self):
        if self.cuda:
            import torch
            torch.cuda.synchronize()

    @contextlib.contextmanager
    def part(self, name):
        self.sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.t[name] += time.perf_counter() - t0


def timed_model(fn, clock: Clock):
    """A forecast function that charges its time to 'model' (and counts calls)."""
    def wrapped(*a, **k):
        clock.calls += 1
        with clock.part("model"):
            return fn(*a, **k)
    return wrapped


def profile_task(task, model, device, mode: str, num_proc: int) -> dict:
    """One task, one mode: the exact steps of score_groups.run_fev_modes, timed."""
    S_, G, C = thirdparty.score_twins(), thirdparty.score_groups(), thirdparty.ft3_core()
    clock = Clock(device.type == "cuda")
    t0 = time.perf_counter()
    with clock.part("data"):
        S_.prepare(task, num_proc)
    fl, fg = timed_model(C.forecast_left_padded, clock), timed_model(C.forecast_groups, clock)
    n_series = 0
    if mode == "off":
        with clock.part("windows"):
            series, index = S_.fev_inputs(task)
        n_series = len(series)
        pr = S_.fev_predictions(model, task, series, index, device, fl)
    else:
        # fev_groups_predictions with window building timed apart from the model calls it makes
        orig = G.window_groups

        def wg(*a, **k):
            with clock.part("windows"):
                return orig(*a, **k)
        G.window_groups = wg
        try:
            pr = G.fev_groups_predictions(model, task, device, fg, fl, S_.fill_season, S_.MAX_CTX, S_.BATCH)
        finally:
            G.window_groups = orig
        n_series = sum(len(next(iter(w.values()))) for w in pr)
    with clock.part("metrics"):
        s = task.evaluation_summary(pr, model_name="profile", trained_on_this_dataset=False)
    total = time.perf_counter() - t0
    # 'other': iter_windows in ON (lazy, inside fev_groups_predictions) and formatting the predictions for fev
    return dict(task=task.task_name, mode=mode, windows_n=task.num_windows, series=n_series, model_calls=clock.calls,
                **{f"{k}_s": v for k, v in clock.t.items()}, other_s=total - sum(clock.t.values()), total_s=total,
                test_error=s["test_error"])


def summarise(df: pd.DataFrame) -> str:
    L = ["# Profile of the current fev scoring path", ""]
    for mode, g in df.groupby("mode"):
        tot = g["total_s"].sum()
        L += [f"## mode {mode}: {len(g)} tasks, {tot / 60:.1f} min, {g['series'].sum()} series-windows, "
              f"{g['model_calls'].sum()} model calls", "", "| part | seconds | share |", "|---|---|---|"]
        for p in (*PARTS, "other"):
            L.append(f"| {p} | {g[f'{p}_s'].sum():.1f} | {100 * g[f'{p}_s'].sum() / tot:.1f}% |")
        L += ["", "Slowest tasks:", "", _md(g.sort_values("total_s", ascending=False).head(10)[
            ["task", "series", "model_calls", *[f"{p}_s" for p in (*PARTS, "other")], "total_s"]].round(2)), ""]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the old scorer's evaluation root (fev data cached under it)")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--modes", default="on,off")
    ap.add_argument("--tasks", type=int, default=0, help="first N fev-bench tasks only (0 = all)")
    ap.add_argument("--num-proc", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args(argv)
    S_ = thirdparty.score_twins()
    S_.setup_env(a.root, offline=True)
    import torch
    from tideforge import model as TM
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    model, _, _ = TM.load(a.ckpt, device)
    tasks = S_.load_fev_tasks(S_.Paths(a.root))
    tasks = tasks[:a.tasks] if a.tasks else tasks
    out = Path(a.out or Path(a.root) / "profile_fev")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, t in enumerate(tasks):
        for mode in [m.strip() for m in a.modes.split(",") if m.strip()]:
            try:
                r = profile_task(t, model, device, mode, a.num_proc)
            except Exception as e:  # noqa: BLE001
                print(f"[{i + 1}/{len(tasks)}] {t.task_name} {mode}: FAILED {type(e).__name__}: {str(e)[:150]}", flush=True)
                continue
            rows.append(r)
            print(f"[{i + 1}/{len(tasks)}] {t.task_name:40s} {mode:3s} " + " ".join(
                f"{p} {r[f'{p}_s']:6.2f}" for p in (*PARTS, "other")) + f"  total {r['total_s']:6.2f}s", flush=True)
            pd.DataFrame(rows).to_csv(out / "profile.csv", index=False)
    df = pd.DataFrame(rows)
    (out / "profile.md").write_text(summarise(df), encoding="utf-8")
    print(summarise(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
