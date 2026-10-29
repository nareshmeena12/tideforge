"""Campaign runner (Phase 3 tools, owner OK 2026-10-29): a list of (experiment, seeds) -> 5M node jobs, 4 runs per node.

    python -m tideforge.campaign.runner plan phase2_screen [--results $TF_ROOT/results/results.csv] [--strict-code] [--write]

configs/campaign/<name>.yaml:
    name: phase2_screen
    seeds: [0]                    # default seeds for every run (Phase 2's screen: 1 seed)
    runs:
      - G1.1                      # an experiment ID (configs/exp/<ID>.yaml) with the default seeds
      - {exp: G4.3, seeds: [0, 1, 2]}

plan: expands the runs into (experiment, seed) pairs in the campaign's order, drops the ones already done, packs the rest 4 to
a regular node (one run per GPU at 5M; tideforge.slurm's node template and QOS checks), prints the GPU-h estimate (the
training rate of tideforge.slurm: an estimate until E0.9's measured rate is filled in) and the submit order. --write writes
scripts/perlmutter/generated/<name>/<name>_NN.sbatch (not committed: it depends on the results file). Same inputs, same
bytes: nothing in the files depends on the date or the machine.
Done: a run is done when results.csv has a FINAL row (ckpt final.pt) whose run id, minus its code version, is the run's
(experiment + generator + model fragment, train hash, seed). --strict-code also requires the code version of HEAD.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

from tideforge import slurm
from tideforge.train import recipe as RC

REPO = Path(__file__).resolve().parents[2]
CAMPAIGNS = REPO / "configs" / "campaign"
GENERATED = slurm.OUT / "generated"
PER_NODE = 4


@dataclass(frozen=True)
class Run:
    exp: str
    seed: int
    key: str                  # the run id without the code version
    points: float

    def hours(self) -> float:
        return self.points / slurm.train_rate()[0] / 3600 + slurm.FINISH_H


def load_campaign(name_or_path: str | Path) -> dict:
    p = Path(name_or_path)
    if p.suffix != ".yaml":
        p = CAMPAIGNS / f"{name_or_path}.yaml"
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    if d.get("name") != p.stem:
        raise ValueError(f"{p}: name {d.get('name')!r} must equal the file name {p.stem!r}")
    return d


def expand(campaign: dict) -> list[Run]:
    seeds0 = [int(s) for s in campaign.get("seeds", [0])]
    out, seen = [], set()
    for item in campaign["runs"]:
        exp, seeds = (item, seeds0) if isinstance(item, str) else (item["exp"], [int(s) for s in item.get("seeds", seeds0)])
        rc = RC.load(exp)
        for s in seeds:
            if (exp, s) in seen:
                raise ValueError(f"{campaign['name']}: {exp} seed {s} listed twice")
            seen.add((exp, s))
            out.append(Run(exp, s, run_key(rc, s), float(rc.sections["optim"]["target_points"])))
    return out


def run_key(rc: RC.Recipe, seed: int) -> str:
    return rc.run_id(seed, "CODE").rsplit("-CODE", 1)[0]


def done_rows(results: str | Path | None) -> list[dict]:
    p = Path(results) if results else None
    if not p or not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if Path(r.get("ckpt", "")).name == "final.pt"]


def is_done(run: Run, rows: list[dict], code: str | None = None) -> bool:
    for r in rows:
        rid = r.get("run_id", "")
        if code is None and rid.startswith(run.key + "-"):
            return True
        if code is not None and rid == f"{run.key}-{code.replace('+', '')}":
            return True
    return False


def node_time(runs: list[Run]) -> str:
    """Wall-time limit of a node: the longest run x 1.15 (validation, checkpoints) with E0.9's measured rate, x 3 while the
    rate is the estimate (a slower real rate would otherwise requeue every run); whole hours, at least 2."""
    margin = 1.15 if slurm.TRAIN_PTS_PER_S else 3.0
    h = max(2, math.ceil(max(r.hours() for r in runs) * margin))
    return f"{h:02d}:00:00"


def jobs(name: str, runs: list[Run]) -> list[tuple[str, dict]]:
    out = []
    for k in range(0, len(runs), PER_NODE):
        part = runs[k:k + PER_NODE]
        jn = f"{name}_{k // PER_NODE + 1:02d}"
        j = dict(qos="regular", gpus=slurm.NODE_GPUS, cpus=slurm.NODE_CPUS, time=node_time(part),
                 runs=[(r.exp, r.seed) for r in part],
                 what=f"campaign {name}: " + ", ".join(f"{r.exp} s{r.seed}" for r in part) + "; one run per GPU, then finish")
        slurm.check(j)
        out.append((jn, j))
    return out


def plan(name: str, results: str | None = None, strict_code: bool = False, write: bool = False, log=print) -> dict:
    c = load_campaign(name)
    runs = expand(c)
    rows = done_rows(results)
    code = RC.code_version() if strict_code else None
    todo = [r for r in runs if not is_done(r, rows, code)]
    js = jobs(c["name"], todo)
    rate, kind = slurm.train_rate()
    busy = sum(r.hours() for r in todo)
    alloc = sum(j["gpus"] * slurm.hours(j["time"]) for _, j in js)
    log(f"campaign {c['name']}: {len(runs)} runs, {len(runs) - len(todo)} done, {len(todo)} to run in {len(js)} node jobs "
        f"({PER_NODE} per regular node)")
    log(f"GPU-h: {busy:.1f} busy ({len(todo)} x training at {rate:.2e} pts/s per GPU, {kind}, + finish {slurm.FINISH_H} h "
        f"ESTIMATE each); allocated at most {alloc:.1f} ({len(js)} nodes x 4 GPUs x their limits)")
    texts = {jn: slurm.render_train(jn, j) for jn, j in js}
    d = GENERATED / c["name"]
    if write:
        d.mkdir(parents=True, exist_ok=True)
        for jn, t in texts.items():
            (d / f"{jn}.sbatch").write_bytes(t.encode("utf-8"))
    log("submit order (from the log folder):" + ("" if write else " [not written: add --write]"))
    for jn, j in js:
        log(f"  cd $TF_ROOT/logs && sbatch $SCRATCH/tideforge/repo/scripts/perlmutter/generated/{c['name']}/{jn}.sbatch"
            f"   # {', '.join(f'{e} s{s}' for e, s in j['runs'])} ({j['time']})")
    return dict(runs=runs, todo=todo, jobs=js, texts=texts, busy_h=busy, alloc_h=alloc)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("plan",))
    ap.add_argument("campaign")
    ap.add_argument("--results", default=None)
    ap.add_argument("--strict-code", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    plan(a.campaign, a.results, a.strict_code, a.write)
    return 0


if __name__ == "__main__":
    sys.exit(main())
