"""Rescore finished runs on probes only, no training (owner 2026-11-01): e.g. P6 under nofill_mask, which is how a model trained
with gaps meets them.

    python -m tideforge.campaign.rescore --results $TF_ROOT/results/results.csv --root $TF_DATA/probes \\
                                         --exps E0.7,G6.1,G6.2,G6.3,G6.4 --probes P6 --nan-variant nofill_mask [--device cuda]

Every FINAL row of the listed experiments (all seeds): the rows are appended to the run's own probes/results.csv under their
nan_variant, which the board reads as a separate line and keeps out of the win rule, the SQL column and e07.json. A run that
already has the probe under this variant is skipped, so the command can be repeated.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tideforge.campaign import board as BD
from tideforge.probes import run as PR
from tideforge.probes.models import make


def rescore(results, root, exps: list[str], probes: list[str], variant: str, device: str = "cuda", log=print, make_model=make) -> int:
    finals, _ = BD.read(results, "dev-v1")
    by = {r.row["run_id"]: r for r in finals if r.exp in exps}              # the last row of a run
    missing = sorted(set(exps) - {r.exp for r in by.values()})
    if missing:
        raise SystemExit(f"no final rows for {missing}")
    done = 0
    for rid, r in by.items():
        d = BD.run_dir_of(r.row)
        csv = d / "probes" / "results.csv"
        have = set()
        if csv.exists():
            f = pd.read_csv(csv)
            have = set(f.loc[f["nan_variant"] == variant, "probe"])
        todo = [p for p in probes if p not in have]
        if not todo:
            log(f"{rid}: already scored")
            continue
        PR.run(make_model("m5", r.row["ckpt"], device, variant), todo, Path(root), d / "probes", variant)
        done += 1
        log(f"{rid}: {','.join(todo)} under {variant}")
    log(f"rescored {done} of {len(by)} runs")
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--exps", required=True)
    ap.add_argument("--probes", default="P6")
    ap.add_argument("--nan-variant", default="nofill_mask", choices=["nofill_delete", "nofill_mask"])
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    rescore(a.results, a.root, a.exps.split(","), PR.probe_ids(a.probes), a.nan_variant, a.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
