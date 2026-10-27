"""Score ft3-architecture checkpoints on fev-bench and TIME, groups ON and OFF, with tidemark's scorers (not rewritten).

  python -m tideforge.eval.score score --root R --ckpt ft3main=/path/ft3_main.pt [--variant baseline] [--modes on,off] [--smoke]
  python -m tideforge.eval.score board --root R [--smoke]
  python -m tideforge.eval.score table --root R --names ft3mainbase[,...] --base ft3mainbase [--gate]
                                       [--looks --experiment E0.2 --why "..."]
  python -m tideforge.eval.score compare-log --root R --name ft3mainbase --log reference/fev_log_ft3.txt
  python -m tideforge.eval.score compare-board --root R [--smoke] --ref <tidemark board>/scores_long.csv

`score` runs third_party's score_groups.main (score_ft3 board for `board`) inside nan_variants.use_variant(variant). The variant code
is appended to the checkpoint name, so it is part of every result tag: ft3main + baseline -> ft3mainbase_s0 (OFF) and
ft3mainbaseon_s0 (ON); nofill_delete -> ft3mainnfdel; nofill_mask -> ft3mainnfmask. score_groups resumes per (tag, task): use a
fresh evaluation root for a new reproduction, never one with old results under the same tags.
`table --gate` checks the base checkpoint against score_groups.GATE (ON fev 43.78, TIME 0.680 / 0.571; OFF fev 42.09,
TIME 0.687 / 0.575; tolerance 0.2 fev, 0.002 TIME). `--looks` adds one LOOKS.md row per checkpoint, benchmark and mode.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from tideforge.eval import looks, thirdparty
from tideforge.eval.nan_variants import CODE, VARIANTS, use_variant, variant_of


def variant_name(name: str, variant: str) -> str:
    return f"{name}{CODE[variant]}"


def cmd_score(a) -> int:
    G = thirdparty.score_groups()
    argv = ["score", "--root", a.root, "--modes", a.modes]
    names = []
    for item in a.ckpt:
        name, sep, path = item.partition("=")
        if not sep:
            raise SystemExit(f"--ckpt {item!r}: expected NAME=PATH")
        names.append(variant_name(name, a.variant))
        argv += ["--ckpt", f"{names[-1]}={path}"]
    for flag in ("smoke", "cpu", "skip_time", "skip_fev"):
        if getattr(a, flag):
            argv.append("--" + flag.replace("_", "-"))
    if a.num_proc is not None:
        argv += ["--num-proc", str(a.num_proc)]
    print(f"variant {a.variant}: checkpoint names {', '.join(names)} (tags NAME_s0 = OFF, NAMEon_s0 = ON)", flush=True)
    with use_variant(a.variant):
        return G.main(argv)


def cmd_board(a) -> int:
    thirdparty.add_path()
    from tidemark.real import score_ft3
    return score_ft3.main(["board", "--root", a.root] + (["--smoke"] if a.smoke else []))


def look_rows(df, names: list[str], experiment: str, why: str) -> list[str]:
    """One LOOKS.md row per checkpoint name, benchmark (fev-bench, TIME) and mode (ON, OFF) present in the mode table."""
    G = thirdparty.score_groups()
    rows = []
    for n in names:
        r = df[df.checkpoint == n]
        if not len(r):
            continue
        r = r.iloc[0]
        var = variant_of(n) or "unknown"
        for mode, tag in (("ON", G.tag_on(n)), ("OFF", G.tag_off(n))):
            ck = f"{tag} (variant {var}, groups {mode})"
            v = lambda c: r.get(c, float("nan"))   # noqa: E731
            if not math.isnan(v(f"fev skill {mode}")):
                res = f"skill {v(f'fev skill {mode}'):.2f}" + (f", win {v(f'fev win {mode}'):.2f}" if f"fev win {mode}" in r else "")
                rows.append(looks.format_row(experiment=experiment, checkpoint=ck, benchmark="fev-bench", why=why, result=res))
            if not math.isnan(v(f"TIME MASE {mode}")):
                res = f"MASE {v(f'TIME MASE {mode}'):.4f}, CRPS {v(f'TIME CRPS {mode}'):.4f}"
                rows.append(looks.format_row(experiment=experiment, checkpoint=ck, benchmark="TIME", why=why, result=res))
    return rows


def cmd_table(a) -> int:
    import pandas as pd
    G = thirdparty.score_groups()
    argv = ["table", "--root", a.root, "--base", a.base] + (["--names", a.names] if a.names else []) + (["--gate"] if a.gate else [])
    rc = G.main(argv)
    if a.looks:
        if not (a.experiment and a.why):
            raise SystemExit("--looks needs --experiment and --why")
        df = pd.read_csv(Path(a.root) / "eval" / "board" / "real_table_modes.csv")
        names = a.names.split(",") if a.names else list(df.checkpoint)
        added = looks.append_looks(look_rows(df, names, a.experiment, a.why), a.looks_file)
        print(f"\n{len(added)} row(s) added to {a.looks_file} (paste them back if this checkout is not the one you commit from):")
        print("\n".join(added))
    return rc


def cmd_compare(a) -> int:
    G = thirdparty.score_groups()
    return G.main(["compare-log", "--root", a.root, "--name", a.name, "--log", a.log])


def compare_board(ours, ref, name: str = "ft3mainbase", ref_name: str = "ft3main"):
    """Per-task differences between our scores_long.csv and a reference board's (same columns: model, arm, seed, benchmark,
    metric, task, window, err). Our arms <name> / <name>on map to the reference's <ref_name> / <ref_name>on (hidden / ON).
    Returns the joined rows of the tasks both have (err_ours, err_ref, abs_diff, rel_diff)."""
    import pandas as pd
    o, r = pd.read_csv(ours), pd.read_csv(ref)
    amap = {name: ref_name, name + "on": ref_name + "on"}
    o = o[o["arm"].isin(amap)].assign(arm=lambda d: d["arm"].map(amap))
    r = r[r["arm"].isin(amap.values())]
    key = ["arm", "benchmark", "metric", "task", "window"]
    j = o[key + ["err"]].merge(r[key + ["err"]], on=key, suffixes=("_ours", "_ref"))
    j["abs_diff"] = (j["err_ours"] - j["err_ref"]).abs()
    j["rel_diff"] = j["abs_diff"] / j["err_ref"].abs().clip(lower=1e-12)
    return j.sort_values("rel_diff", ascending=False)


def cmd_compare_board(a) -> int:
    from pathlib import Path
    ours = Path(a.root) / ("eval_smoke" if a.smoke else "eval") / "board" / "scores_long.csv"
    j = compare_board(ours, a.ref, a.name, a.ref_name)
    if j.empty:
        print(f"no common (arm, benchmark, metric, task, window) between {ours} and {a.ref}")
        return 1
    print(j.to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    print(f"\n{len(j)} task rows compared; max relative difference {j['rel_diff'].max():.3g}; "
          f"rows with |rel diff| > 1e-4: {int((j['rel_diff'] > 1e-4).sum())}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("score", "board", "table", "compare-log", "compare-board"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--ckpt", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--variant", choices=VARIANTS, default="baseline")
    ap.add_argument("--modes", default="on,off")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--skip-time", action="store_true")
    ap.add_argument("--skip-fev", action="store_true")
    ap.add_argument("--num-proc", type=int, default=None)
    ap.add_argument("--names", default=None, help="table: comma-separated checkpoint names incl. the variant code")
    ap.add_argument("--base", default="ft3mainbase")
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--looks", action="store_true", help="table: add the LOOKS.md rows")
    ap.add_argument("--looks-file", default=str(looks.LOOKS))
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--why", default=None)
    ap.add_argument("--name", default="ft3mainbase", help="compare-log: checkpoint name")
    ap.add_argument("--log", default="reference/fev_log_ft3.txt")
    ap.add_argument("--ref", default=None, help="compare-board: the reference scores_long.csv")
    ap.add_argument("--ref-name", default="ft3main", help="compare-board: the reference's checkpoint name")
    a = ap.parse_args(argv)
    if a.cmd == "score":
        if not a.ckpt:
            raise SystemExit("score needs at least one --ckpt NAME=PATH")
        return cmd_score(a)
    return {"board": cmd_board, "table": cmd_table, "compare-log": cmd_compare, "compare-board": cmd_compare_board}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
