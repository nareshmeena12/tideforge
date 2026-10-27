"""Score ft3-architecture checkpoints (ft3_main and the post-trained ones) with the repo's TIME / fev-bench code, unchanged.

  python -m tidemark.real.score_ft3 score --root R --ckpt ft3main=/path/ft3_main.pt [--ckpt p10b=/path/ckpt_10B.pt ...] [--smoke]
  python -m tidemark.real.score_ft3 board --root R                    # the leaderboards + head-to-heads (score_twins.cmd_board)
  python -m tidemark.real.score_ft3 table --root R [--base ft3main] [--expect-time 0.680 --expect-fev 43.78]

R is an evaluation root with the same layout as the 30M one ($ROOT/eval/{data,hf,fev_naive.csv,...}; scripts/perlmutter/
real_eval.sh builds $ROOT/size_real with links to $ROOT/eval). The checkpoint NAME must match [a-z0-9]+; its tag is NAME_s0.

The model is loaded exactly like ft3_core.load_packed_checkpoint (config from the checkpoint, dropout 0) but with strict state-dict
loading checked explicitly. Inference is ft3_core.forecast_left_padded (the reference's, univariate: every variate forecast alone,
context 8192, 9 quantiles, history gaps filled one season back, horizons read in one pass), plugged into score_twins.run_time /
run_fev through their `forecast` argument. The groups inference mode (forecast_groups with covariates) is NOT wired here.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

NAME_RE = re.compile(r"^[a-z0-9]+$")


def parse_ckpts(items: list[str]) -> dict[str, str]:
    out = {}
    for it in items:
        name, sep, path = it.partition("=")
        if not sep or not NAME_RE.match(name):
            raise SystemExit(f"--ckpt {it!r}: expected NAME=PATH with NAME in [a-z0-9]+, e.g. ft3main=/path/ft3_main.pt")
        if name in out:
            raise SystemExit(f"--ckpt {name} given twice")
        out[f"{name}_s0"] = path
    return out


def load_ft3(path, device):
    """(model in eval mode, cfg, checkpoint). Strict: every key of the checkpoint must match the model, and vice versa."""
    import torch
    from tidemark.real import ft3_core as C
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ck["cfg"])
    model = C.build_model(dict(cfg, dropout=0.0)).to(device)
    res = model.load_state_dict(ck["model"], strict=True)
    if res.missing_keys or res.unexpected_keys:
        raise SystemExit(f"{path}: state dict does not match the model: missing {res.missing_keys[:5]} unexpected {res.unexpected_keys[:5]}")
    return model.eval(), cfg, ck


def cmd_score(a) -> int:
    import torch
    from tidemark.eval.score_twins import (Paths, load_fev_tasks, read_manifest, run_fev, run_time, setup_env, smoke_fev)
    from tidemark.real import ft3_core as C
    setup_env(a.root, offline=True)
    paths = Paths(a.root, smoke=a.smoke)
    device = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    ckpts = parse_ckpts(a.ckpt)
    models = {}
    for tag, path in ckpts.items():
        m, cfg, ck = load_ft3(path, device)
        n = sum(p.numel() for p in m.parameters())
        print(f"loaded {tag}: {n / 1e6:.2f}M params, strict state dict OK, step {ck.get('step')}, "
              f"{float(ck.get('points_seen', 0)) / 1e9:.1f}B points seen", flush=True)
        models[tag] = m
    if not a.skip_time:
        run_time(models, paths, device, smoke=a.smoke, forecast=C.forecast_left_padded)
    if not a.skip_fev:
        tasks = load_fev_tasks(paths)
        tasks = smoke_fev(tasks, read_manifest(paths)) if a.smoke else tasks
        run_fev(models, tasks, paths, device, 4 if a.num_proc is None else a.num_proc, forecast=C.forecast_left_padded)
    return 0


def cmd_board(a) -> int:
    from tidemark.eval.score_twins import Paths, cmd_board, setup_env
    setup_env(a.root, offline=True)
    return cmd_board(Paths(a.root, smoke=a.smoke), a.smoke)


# ---------------------------------------------------------------------------------------------- the table
def results_table(root, tags: list[str], base: str) -> "pd.DataFrame":
    """One row per checkpoint: TIME MASE / CRPS (normalised to seasonal naive, as the board prints), fev skill score and the other
    fev-leaderboard columns that exist, and the change against `base`."""
    import pandas as pd
    board = Path(root) / "eval" / "board"
    tb = pd.read_csv(board / "time_board.csv", index_col=0) if (board / "time_board.csv").exists() else pd.DataFrame()
    fb = pd.read_csv(board / "fev_board.csv", index_col=0) if (board / "fev_board.csv").exists() else pd.DataFrame()
    rows = []
    for t in tags:
        r = {"checkpoint": t.removesuffix("_s0")}
        if t in tb.index:
            r["TIME MASE"] = float(tb.loc[t, "MASE (norm.)"])
            r["TIME CRPS"] = float(tb.loc[t, "CRPS (norm.)"])
        if t in fb.index:
            for c in ("skill_score", "win_rate"):
                if c in fb.columns:
                    r[f"fev {c} (x100)"] = 100 * float(fb.loc[t, c])
        rows.append(r)
    df = pd.DataFrame(rows)
    b = df[df.checkpoint == base.removesuffix("_s0")]
    if len(b):
        b = b.iloc[0]
        for c in [c for c in df.columns if c != "checkpoint"]:
            if c.startswith("TIME"):
                df[f"{c} change"] = 100 * (df[c] / b[c] - 1)                 # % ; negative = better
            else:
                df[f"{c} change"] = df[c] - b[c]                              # points ; positive = better
    return df


def check_expectations(df, base: str, time_mase=None, fev=None, tol_time=0.002, tol_fev=0.2, fev_alt=None, time_crps=None) -> list[str]:
    """PASS / FAIL lines for the baseline against the owner's UNIVARIATE numbers (TIME MASE 0.687, CRPS 0.575, fev 42.09; the
    groups-mode fev 43.78 is not reproducible with the univariate scorer)."""
    b = df[df.checkpoint == base.removesuffix("_s0")]
    if not len(b):
        return [f"baseline {base} is not in the table"]
    b = b.iloc[0]
    out = []
    if time_mase is not None and "TIME MASE" in b:
        d = float(b["TIME MASE"]) - time_mase
        out.append(f"TIME MASE {b['TIME MASE']:.4f} vs {time_mase:.3f}: diff {d:+.4f} (tolerance {tol_time}) -> {'PASS' if abs(d) <= tol_time else 'FAIL'}")
    if time_crps is not None and "TIME CRPS" in b:
        d = float(b["TIME CRPS"]) - time_crps
        out.append(f"TIME CRPS {b['TIME CRPS']:.4f} vs {time_crps:.3f}: diff {d:+.4f} (tolerance {tol_time}) -> {'PASS' if abs(d) <= tol_time else 'FAIL'}")
    if fev is not None:
        cols = [c for c in df.columns if c.startswith("fev skill_score") and not c.endswith("change")]      # the skill score only
        for c in cols:
            v = float(b[c])
            hits = [f"{x:g}" for x in [fev, fev_alt] if x is not None and abs(v - x) <= tol_fev]
            out.append(f"{c} = {v:.2f}: " + (f"matches {', '.join(hits)} within {tol_fev}" if hits else
                                              f"no match for {fev:g}" + (f" / {fev_alt:g}" if fev_alt else "") + f" (tolerance {tol_fev})"))
        if not cols:
            out.append("no fev result for the baseline")
    return out


def cmd_table(a) -> int:
    import pandas as pd
    pd.set_option("display.width", 200)
    tags = [t for t in (a.tags.split(",") if a.tags else sorted(p.name for p in (Path(a.root) / "eval" / "time").glob("*")
                                                                if (Path(a.root) / "eval" / "time").exists()))]
    tags = [t for t in tags if re.match(r"^[a-z0-9]+_s0$", t) and not t.startswith(("softmax", "gla", "swa", "gdn"))] or tags
    df = results_table(a.root, tags, a.base)
    print(df.round(4).to_string(index=False))
    out = Path(a.root) / "eval" / "board" / "real_table.csv"
    df.to_csv(out, index=False)
    print(f"written {out}")
    for line in check_expectations(df, a.base, a.expect_time, a.expect_fev, a.tol_time, a.tol_fev, a.expect_fev_alt, a.expect_crps):
        print(line)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("score", "board", "table"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--skip-time", action="store_true")
    ap.add_argument("--skip-fev", action="store_true")
    ap.add_argument("--num-proc", type=int, default=None)
    ap.add_argument("--tags", default=None, help="table: comma-separated tags (default: the NAME_s0 folders under eval/time)")
    ap.add_argument("--base", default="ft3main", help="table: the checkpoint the change is measured against")
    ap.add_argument("--expect-time", type=float, default=None)
    ap.add_argument("--expect-crps", type=float, default=None)
    ap.add_argument("--expect-fev", type=float, default=None)
    ap.add_argument("--expect-fev-alt", type=float, default=None, help="the owner's second fev number (42.09)")
    ap.add_argument("--tol-time", type=float, default=0.002)
    ap.add_argument("--tol-fev", type=float, default=0.2)
    a = ap.parse_args(argv)
    if a.cmd == "score":
        if not a.ckpt:
            raise SystemExit("score needs at least one --ckpt NAME=PATH")
        return cmd_score(a)
    return cmd_board(a) if a.cmd == "board" else cmd_table(a)


if __name__ == "__main__":
    sys.exit(main())
