"""Groups-mode scoring of ft3-architecture checkpoints on TIME and fev-bench, next to the univariate (groups OFF) mode.

  python -m tidemark.real.score_groups score --root R --ckpt ft3main=/path/ft3_main.pt [--ckpt ...] [--modes on,off] [--smoke]
  python -m tidemark.real.score_groups table --root R [--base ft3main] [--gate]
  python -m tidemark.real.score_groups compare-log --root R --log reference/fev_log_ft3.txt [--name ft3main]

Two modes per checkpoint NAME (tags in the evaluation root R, both with the same univariate scoring code underneath):
  OFF  tag NAME_s0     every variate forecast alone (score_ft3 / score_twins: forecast_left_padded). Unchanged.
  ON   tag NAMEon_s0   groups ON.
         TIME: as reference/time_bench_colab.py mode "groups": a multivariate series (D > 1) is ONE group, its variates are the group's
               targets (ft3_core.forecast_groups, rows 256); univariate datasets (D == 1) are forecast exactly as in OFF.
         fev:  one group per item and window: the item's target columns (sibling variates), its past dynamic covariates (role PAST)
               and its known dynamic covariates with their future values (role KNOWN); static covariates are not used. A task with one
               target and no covariate is forecast as in OFF (the reference log shows identical ON and HIDDEN numbers on those tasks).
History gaps are filled one season back (fill_season) in both modes, as in the owner's Colab runs.

NOT recoverable from the files at hand, so assumed (the likeliest sources of a mismatch with the owner's 43.78): the reference fev cell's
`predict_task` is not in reference/. (1) How covariates that are not numbers (strings such as StateHoliday) are encoded: here the
sorted distinct values of the column in the window become 0, 1, 2 ..., missing -> NaN. (2) The order in which covariates are kept when
a group has more than the 8 lanes: the task's column order (forecast_groups keeps known before past). (3) forecast_groups stands in for
the Colab file's forecast_groups_multi (score_tide100m_colab.py, not available): same algorithm per the repo's copy of the reference.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

COVARIATE_VOCAB_MISSING = ("", "nan", "None")


def tag_off(name: str) -> str:
    return f"{name}_s0"


def tag_on(name: str) -> str:
    return f"{name}on_s0"


# ----------------------------------------------------------------------------------------------- covariate encoding
def encode_column(values: list) -> list[np.ndarray]:
    """One covariate column of a window (list of per-item arrays) -> float32 arrays. Numbers stay numbers; anything else is coded by the
    sorted distinct values of the column (0, 1, 2, ...), missing values -> NaN."""
    arrs = [np.asarray(v) for v in values]
    first = next((a for a in arrs if a.size), None)
    if first is None or first.dtype.kind in "fiub":
        return [a.astype(np.float32) for a in arrs]
    vocab = sorted({str(x) for a in arrs for x in a.tolist() if x is not None and str(x) not in COVARIATE_VOCAB_MISSING})
    code = {v: float(i) for i, v in enumerate(vocab)}
    return [np.array([code.get(str(x), np.nan) if x is not None else np.nan for x in a.tolist()], dtype=np.float32) for a in arrs]


def has_groups(window) -> bool:
    """Does groups ON differ from groups OFF on this task: several targets, or any dynamic covariate."""
    return len(window.target_columns) > 1 or bool(window.known_dynamic_columns) or bool(window.past_dynamic_columns)


def window_groups(window, max_ctx: int, season: int, fill) -> list[dict]:
    """The groups of one fev window: one per item (see the module docstring)."""
    past, future = window.get_input_data()
    tg = {c: [fill(np.asarray(y, dtype=np.float64)[-max_ctx:], season) for y in past[c]] for c in window.target_columns}
    ps = {c: encode_column(list(past[c])) for c in window.past_dynamic_columns}
    kc = {c: encode_column(list(past[c]) + list(future[c])) for c in window.known_dynamic_columns}
    n = len(tg[window.target_columns[0]])
    groups = []
    for i in range(n):
        known = [(kc[c][i][-max_ctx:], kc[c][n + i]) for c in window.known_dynamic_columns]       # (context, future values)
        groups.append(dict(targets=[tg[c][i] for c in window.target_columns],
                           past=[ps[c][i][-max_ctx:] for c in window.past_dynamic_columns], known=known))
    return groups


def fev_groups_predictions(model, task, device, forecast_groups, forecast_plain, fill, max_ctx: int, batch: int, rows: int = 256):
    """Predictions in fev's format for groups ON: [window] -> {target column: [per item {predictions, quantiles}]}."""
    q_keys = [str(q) for q in task.quantile_levels]
    assert len(q_keys) == 9, "task quantiles != the model's 9"
    S = int(task.seasonality)
    out = []
    for w in task.iter_windows():
        cols = w.target_columns
        groups = window_groups(w, max_ctx, S, fill)
        if not has_groups(w):
            ser = [g["targets"][0] for g in groups]
            _, qt = forecast_plain(model, ser, task.horizon, device, max_ctx=max_ctx, batch_size=batch)      # (N, 9, H), as OFF
            res = [(None, qt[i][None]) for i in range(len(groups))]
        else:
            res = forecast_groups(model, groups, task.horizon, device, max_ctx=max_ctx, rows=rows)
        d = {c: [] for c in cols}
        for gi, (_, qt) in enumerate(res):
            qt = np.sort(np.asarray(qt, dtype=np.float64), axis=1)                              # (n_targets, 9, H)
            for ci, c in enumerate(cols):
                d[c].append({"predictions": qt[ci, 4].tolist(), **{q: qt[ci, j].tolist() for j, q in enumerate(q_keys)}})
        out.append(d)
    return out


# ----------------------------------------------------------------------------------------------- TIME
def time_quantiles(model, inputs, H: int, S: int, D: int, device, mode: str, fill, forecast_groups, forecast_plain, max_ctx: int,
                   batch: int, chunk: int = 4096, rows: int = 256, QL=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    """(n, 9, D, H) quantiles (or (n, 9, H) if D == 1), sorted: the reference's forecast_task."""
    out = []
    for c0 in range(0, len(inputs), chunk):
        part = inputs[c0:c0 + chunk]
        if D == 1 or mode == "off":
            ys = []
            for d in part:
                a = np.asarray(d["target"])
                for v in (a[None] if a.ndim == 1 else a):
                    ys.append(fill(v[-max_ctx:], S))
            _, q = forecast_plain(model, ys, H, device, max_ctx=max_ctx, batch_size=batch)           # (n*D, 9, H)
            out.append(q if D == 1 else q.reshape(len(part), D, len(QL), H).transpose(0, 2, 1, 3))
        else:
            groups = [dict(targets=[fill(v[-max_ctx:], S) for v in np.asarray(d["target"])]) for d in part]
            res = forecast_groups(model, groups, H, device, max_ctx=max_ctx, rows=rows)
            out.append(np.stack([q.transpose(1, 0, 2) for _, q in res]))                             # (n, 9, D, H)
    return np.sort(np.concatenate(out).astype(np.float64), axis=1)


def run_time_modes(models: dict, modes, paths, device, tb=None, smoke: bool = False, log=print) -> None:
    from tidemark.real import ft3_core as C
    from tidemark.eval import score_twins as S_
    tb = tb or S_.TB(paths)
    config = tb.load_dataset_config()
    names = list(config["datasets"])
    if smoke:
        names = S_.smoke_time(names, S_.read_manifest(paths))
    t_all = time.time()
    for ds_name in names:
        for term in tb.get_available_terms(ds_name, config):
            todo = {mode: {n: m for n, m in models.items()
                           if not S_.time_done(paths, tag_on(n) if mode == "on" else tag_off(n), ds_name, term)} for mode in modes}
            if not any(todo.values()):
                continue
            t0 = time.time()
            try:
                st = tb.get_dataset_settings(ds_name, term, config)
                ds = tb.Dataset(name=ds_name, term=term, to_univariate=False, prediction_length=st["prediction_length"],
                                test_length=st["test_length"], val_length=st["val_length"])
                H, D, Sx = ds.prediction_length, ds.target_dim, tb.get_seasonality(ds.freq)
                inputs = list(ds.test_data.input)
            except Exception as e:  # noqa: BLE001
                log(f"{ds_name:32s} {term:6s} FAILED (data): {type(e).__name__}: {str(e)[:160]}")
                continue
            done: dict = {}
            for mode in modes:
                for name, model in todo[mode].items():
                    tag = tag_on(name) if mode == "on" else tag_off(name)
                    try:
                        key = (name, "off" if D == 1 else mode)                 # D == 1: one forecast serves both modes
                        if key not in done:
                            done[key] = time_quantiles(model, inputs, H, Sx, D, device, mode, S_.fill_season, C.forecast_groups,
                                                       C.forecast_left_padded, S_.MAX_CTX, S_.BATCH)
                        tb.save_window_predictions(dataset=ds, fc_quantiles=done[key], ds_config=f"{ds_name}/{term}",
                                                   output_base_dir=str(paths.time_out / tag), seasonality=Sx, quantile_levels=S_.QL,
                                                   model_hyperparams={"model": tag, "context_length": S_.MAX_CTX,
                                                                      "mode": "groups" if mode == "on" else "univariate"})
                    except Exception as e:  # noqa: BLE001
                        log(f"{ds_name:32s} {term:6s} {tag} FAILED: {type(e).__name__}: {str(e)[:160]}")
            log(f"{ds_name:32s} {term:6s} H={H:<4d} D={D:<3d} n={len(inputs):<6d} {time.time() - t0:6.1f}s")
    log(f"TIME (modes {','.join(modes)}): {(time.time() - t_all) / 60:.1f} min")


# ----------------------------------------------------------------------------------------------- fev
def run_fev_modes(models: dict, modes, tasks, paths, device, num_proc: int = 4, log=print) -> None:
    from tidemark.real import ft3_core as C
    from tidemark.eval import score_twins as S_
    t_all = time.time()
    for ti, task in enumerate(tasks):
        tn = task.task_name
        todo = {mode: {t: m for t, m in ({tag_on(n): m for n, m in models.items()} if mode == "on"
                                        else {tag_off(n): m for n, m in models.items()}).items() if not S_.fev_done(paths, t, tn)}
                for mode in modes}
        if not any(todo.values()):
            continue
        t0 = time.time()
        try:
            S_.prepare(task, num_proc)
            series, index = S_.fev_inputs(task) if todo.get("off") else (None, None)
        except Exception as e:  # noqa: BLE001
            log(f"   {tn}: FAILED (data) {type(e).__name__}: {str(e)[:150]}")
            continue
        for mode in modes:
            for tag, model in todo[mode].items():
                try:
                    if mode == "off":
                        pr = S_.fev_predictions(model, task, series, index, device, C.forecast_left_padded)
                    else:
                        pr = fev_groups_predictions(model, task, device, C.forecast_groups, C.forecast_left_padded, S_.fill_season,
                                                    S_.MAX_CTX, S_.BATCH)
                    s_ = task.evaluation_summary(pr, model_name=tag, trained_on_this_dataset=False)
                    S_.write_json(paths.fev_out / tag / f"{S_.safe(tn)}.json", s_)
                except Exception as e:  # noqa: BLE001
                    log(f"   {tn} {tag}: FAILED {type(e).__name__}: {str(e)[:150]}")
        log(f"[{ti + 1}/{len(tasks)}] {tn:45s} {sum(len(v) for v in todo.values())} run(s) {time.time() - t0:6.1f}s")
    log(f"fev-bench (modes {','.join(modes)}): {(time.time() - t_all) / 60:.1f} min")


# ----------------------------------------------------------------------------------------------- the table
def mode_table(root, names: list[str], base: str):
    """One row per checkpoint, both modes side by side: TIME MASE / CRPS, fev skill (x100) and win rate (x100), and the change vs `base`
    for each (TIME in %, negative = better; fev in points, positive = better)."""
    import pandas as pd
    board = Path(root) / "eval" / "board"
    tb = pd.read_csv(board / "time_board.csv", index_col=0) if (board / "time_board.csv").exists() else pd.DataFrame()
    fb = pd.read_csv(board / "fev_board.csv", index_col=0) if (board / "fev_board.csv").exists() else pd.DataFrame()
    cols = {}
    rows = []
    for n in names:
        r = {"checkpoint": n}
        for mode, tag in (("ON", tag_on(n)), ("OFF", tag_off(n))):
            if tag in tb.index:
                r[f"TIME MASE {mode}"] = float(tb.loc[tag, "MASE (norm.)"])
                r[f"TIME CRPS {mode}"] = float(tb.loc[tag, "CRPS (norm.)"])
            if tag in fb.index:
                r[f"fev skill {mode}"] = 100 * float(fb.loc[tag, "skill_score"])
                if "win_rate" in fb.columns:
                    r[f"fev win {mode}"] = 100 * float(fb.loc[tag, "win_rate"])
        rows.append(r)
    df = pd.DataFrame(rows)
    b = df[df.checkpoint == base]
    if len(b):
        b = b.iloc[0]
        for c in [c for c in df.columns if c != "checkpoint"]:
            if c in b and pd.notna(b[c]):
                df[f"{c} chg"] = 100 * (df[c] / b[c] - 1) if c.startswith("TIME") else df[c] - b[c]
    return df


GATE = {"ON": {"TIME MASE": (0.680, 0.002), "TIME CRPS": (0.571, 0.002), "fev skill": (43.78, 0.2)},
        "OFF": {"TIME MASE": (0.687, 0.002), "TIME CRPS": (0.575, 0.002), "fev skill": (42.09, 0.2)}}


def gate_lines(df, base: str) -> list[str]:
    b = df[df.checkpoint == base]
    if not len(b):
        return [f"baseline {base} is not in the table"]
    b = b.iloc[0]
    out, ok_all = [], True
    for mode, specs in GATE.items():
        for metric, (want, tol) in specs.items():
            key = f"{metric} {mode}"
            if key not in b or b[key] != b[key]:
                out.append(f"{key}: no result"); ok_all = False
                continue
            d = float(b[key]) - want
            ok = abs(d) <= tol
            ok_all &= ok
            out.append(f"{key} = {b[key]:.4f} vs {want} (tolerance {tol}): diff {d:+.4f} -> {'PASS' if ok else 'FAIL'}")
    out.append("GATE: " + ("PASS" if ok_all else "FAIL (stop: report what differs; do not tune)"))
    return out


# ----------------------------------------------------------------------------------------------- per-task comparison with the Colab log
LOG_LINE = re.compile(r"\[\s*(\d+)/\d+\]\s+(\S+)\s+h=(\d+)\s+ON:(.*?)\|\s*HIDDEN:(.*?)\|")


def parse_fev_log(path, model: str = "ft3_main_season") -> dict[str, tuple[float, float]]:
    """{task: (ratio ON, ratio HIDDEN)} of `model` from the Colab side-by-side lines `[ n/100] task h=.. ON: m v ... | HIDDEN: m v ...`."""
    def values(s):
        return {m.group(1): float(m.group(2)) for m in re.finditer(r"(\S+)\s+(-?[0-9]*\.?[0-9]+)\*?", s)}
    out = {}
    for ln in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = LOG_LINE.match(ln)
        if m:
            on, hid = values(m.group(4)), values(m.group(5))
            if model in on and model in hid:
                out[m.group(2)] = (on[model], hid[model])
    return out


def our_ratios(root, tag: str) -> dict[str, float]:
    """{task: model error / seasonal-naive error} from the per-task fev summaries of `tag` and the cached naive."""
    import pandas as pd
    root = Path(root)
    naive = pd.read_csv(root / "eval" / "fev_naive.csv")
    naive = naive[naive.model_name == "seasonal_naive"].drop_duplicates("task_name").set_index("task_name").test_error
    out = {}
    for p in sorted((root / "eval" / "fev" / tag).glob("*.json")):
        s = json.loads(p.read_text())
        t = s["task_name"]
        if t in naive.index and naive[t] > 0 and s["test_error"] > 0:
            out[t] = float(s["test_error"] / naive[t])
    return out


def compare_with_log(root, name: str, log_path, log_model: str = "ft3_main_season", thr: float = 0.005, top: int = 10) -> list[str]:
    ref = parse_fev_log(log_path, log_model)
    L = []
    for mode, tag, idx in (("ON", tag_on(name), 0), ("OFF (HIDDEN)", tag_off(name), 1)):
        ours = our_ratios(root, tag)
        common = sorted(set(ours) & set(ref))
        if not common:
            L.append(f"{mode}: no common task between {tag} and the log")
            continue
        diffs = sorted(((abs(ours[t] - ref[t][idx]), t, ours[t], ref[t][idx]) for t in common), reverse=True)
        n_big = sum(1 for d, *_ in diffs if d > thr)
        L.append(f"{mode}: {len(common)} tasks compared with the log ({len(ref)} parsed); {n_big} differ by more than {thr}; "
                 f"largest {diffs[0][0]:.4f}")
        L += [f"   {t:36s} ours {o:.4f}  log {r:.4f}  diff {o - r:+.4f}" for d, t, o, r in diffs[:top]]
    return L


# ----------------------------------------------------------------------------------------------- CLI
def cmd_score(a) -> int:
    import torch
    from tidemark.eval.score_twins import Paths, load_fev_tasks, read_manifest, setup_env, smoke_fev
    from tidemark.real.score_ft3 import load_ft3, parse_ckpts
    setup_env(a.root, offline=True)
    paths = Paths(a.root, smoke=a.smoke)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    modes = tuple(m.strip() for m in a.modes.split(",") if m.strip())
    if not set(modes) <= {"on", "off"}:
        raise SystemExit("--modes: on, off or on,off")
    models = {}
    for tag, path in parse_ckpts(a.ckpt).items():
        name = tag.removesuffix("_s0")
        m, cfg, ck = load_ft3(path, device)
        print(f"loaded {name}: strict state dict OK, step {ck.get('step')}, {float(ck.get('points_seen', 0)) / 1e9:.1f}B points seen", flush=True)
        models[name] = m
    if not a.skip_time:
        run_time_modes(models, modes, paths, device, smoke=a.smoke)
    if not a.skip_fev:
        tasks = load_fev_tasks(paths)
        tasks = smoke_fev(tasks, read_manifest(paths)) if a.smoke else tasks
        run_fev_modes(models, modes, tasks, paths, device, 4 if a.num_proc is None else a.num_proc)
    return 0


def cmd_table(a) -> int:
    import pandas as pd
    pd.set_option("display.width", 250)
    names = a.names.split(",") if a.names else sorted({p.name.replace("on_s0", "").replace("_s0", "") for p in (Path(a.root) / "eval" / "time").glob("*_s0")
                                                       if not p.name.startswith(("softmax", "gla", "swa", "gdn"))})
    df = mode_table(a.root, names, a.base)
    print(df.round(4).to_string(index=False))
    out = Path(a.root) / "eval" / "board" / "real_table_modes.csv"
    df.to_csv(out, index=False)
    print(f"written {out}")
    if a.gate:
        print("\n".join(gate_lines(df, a.base)))
    return 0


def cmd_compare(a) -> int:
    print("\n".join(compare_with_log(a.root, a.name, a.log, a.log_model)))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("score", "table", "compare-log"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--modes", default="on,off")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--skip-time", action="store_true")
    ap.add_argument("--skip-fev", action="store_true")
    ap.add_argument("--num-proc", type=int, default=None)
    ap.add_argument("--names", default=None, help="table: comma-separated checkpoint names (default: those found under eval/time)")
    ap.add_argument("--base", default="ft3main")
    ap.add_argument("--gate", action="store_true", help="table: check the baseline against the owner's numbers (ON 43.78 / 0.680 / 0.571, OFF 42.09 / 0.687 / 0.575)")
    ap.add_argument("--name", default="ft3main", help="compare-log: checkpoint name")
    ap.add_argument("--log", default="reference/fev_log_ft3.txt")
    ap.add_argument("--log-model", default="ft3_main_season")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        if not a.ckpt:
            raise SystemExit("score needs at least one --ckpt NAME=PATH")
        return cmd_score(a)
    return cmd_table(a) if a.cmd == "table" else cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
