"""Results board (Phase 3 tools, owner OK 2026-10-29): every finished run of results.csv against B0, with E0.7's range.

    python -m tideforge.campaign.board --results $TF_ROOT/results/results.csv --noise $TF_ROOT/noise/e07.json
                                       [--version dev-v1] [--b0 E0.7] [--out $TF_ROOT/board] [--write]

Reads per FINAL row (ckpt final.pt) the run's own files under its run_dir (finish writes them): dev/<version>/tuning_fast/m5/run.json
(main dev), dev/<version>/tuning_cov/m5/run.json (dev covariate, ON and hidden) and probes/results.csv. Milestone rows (ckpt_<N>B.pt)
go to a separate table (X3's rank check).
Per experiment, over its seeds: the geometric mean of each overall score (the dev scores are weighted geometric means of
ratios to seasonal naive, so A / B of two such means is the paired ratio devscore compare reports), the mean of each probe
value. B0 = the --b0 experiment (E0.7, 3 seeds) the same way. gain = 1 - A / B (MASE; > 0 better), the units of E0.7's range
(tideforge.eval.noise: dev for the main score, dev_cov for the covariate score).
Win rule: the experiment config's win: block (expcfg.check_win), else rule 5's default (dev beats B0 by more than the range,
no domain slice worse by more than 1%). Each criterion PASS / FAIL / UNKNOWN (missing data or no noise file); the verdict is
mechanical: "win" (all PASS, >= 3 seeds), "win, 1-seed screen: confirm with 3 seeds", "no win" (any FAIL), "unknown".
Runs scored on another dev version or set than B0 are "not comparable".
Probe verdict (owner 2026-11-01): each probe's designed main metric (MAIN), A against B0's per-seed min..max of that same number,
computed from the B0 runs' probe CSVs (the files e07.json is built from): improves / worse beyond the range, else no effect. It
is a reading aid; the win rule above does not use it. SQL stays as a second column. Controls (E0.8) are reported on every probe.
Only nan_variant baseline rows enter the win rule, SQL and the means; other variants (campaign.rescore) only their own MAIN line.
--cells P1:G1.13a also writes one experiment's per-cell table against B0.
Outputs: <out>/board.md, board.csv (per experiment), runs.csv (per run), milestones.csv. --write also replaces, from the same
numbers, each experiment's Part B table in experiments/<ID>.md (between <!-- board:begin --> and <!-- board:end -->; inserted
in place of the template's empty table the first time) and its index row in EXPERIMENTS.md section 9 (rows the board owns end
with <!-- board -->; a hand-written row for the same ID is left alone and reported). Nothing else in those files is touched.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.eval import noise as NZ

REPO = Path(__file__).resolve().parents[2]
BEGIN, END, ROWTAG = "<!-- board:begin -->", "<!-- board:end -->", "<!-- board -->"
INDEX_HEAD = "| ID | Date | Status | Result (one line) | Log |"
METRIC = "MASE"


# ------------------------------------------------------------------------------------------------- reading

def _json(p: Path) -> dict | None:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


@dataclass
class RunData:
    row: dict
    exp: str
    seed: int
    dev: dict | None = None          # run.json of the main fast set
    cov: dict | None = None          # run.json of the covariate set
    probes: pd.DataFrame | None = None
    points: float = math.nan


def run_dir_of(row: dict) -> Path:
    return Path(row.get("run_dir") or Path(row["ckpt"]).parent)


def read(results: str | Path, version: str) -> tuple[list[RunData], list[dict]]:
    with open(results, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    finals, miles = [], []
    for r in rows:
        name = Path(r.get("ckpt", "")).name
        if name == "final.pt":
            d = run_dir_of(r)
            base = d / "dev" / version
            pr = d / "probes" / "results.csv"
            finals.append(RunData(r, r["exp"], int(r["seed"]), _json(base / "tuning_fast" / "m5" / "run.json"),
                                  _json(base / "tuning_cov" / "m5" / "run.json"), pd.read_csv(pr) if pr.exists() else None))
        elif re.fullmatch(r"ckpt_\d+B\.pt", name):
            miles.append(dict(exp=r["exp"], seed=r["seed"], run_id=r["run_id"], points_B=int(name[5:-4]),
                              dev_MASE=r.get("dev_MASE", ""), ckpt=r["ckpt"]))
    return finals, miles


# ------------------------------------------------------------------------------------------------- aggregation

def gmean(v) -> float:
    v = [float(x) for x in v]
    return float(np.exp(np.mean(np.log(v)))) if v else math.nan


@dataclass
class Agg:
    exp: str
    seeds: list
    dev: dict = field(default_factory=dict)        # MASE, CRPS
    cov_on: dict = field(default_factory=dict)
    cov_gain: float = math.nan                     # 1 - ON / hidden (MASE), mean over seeds
    domains: dict = field(default_factory=dict)    # domain -> MASE
    slices: dict = field(default_factory=dict)     # "key=value" -> MASE
    probes: pd.DataFrame | None = None             # mean value per (probe, cell, mode, view, metric, control)
    keys: set = field(default_factory=set)         # (version, set sha) of the dev and cov runs
    seed_probes: list = field(default_factory=list)  # each seed's probe rows


def _variant(f: pd.DataFrame, name: str = "baseline") -> pd.DataFrame:
    return f[f["nan_variant"] == name] if "nan_variant" in f else (f if name == "baseline" else f.iloc[:0])


def aggregate(exp: str, runs: list[RunData]) -> Agg:
    a = Agg(exp, sorted(r.seed for r in runs))
    devs = [r.dev for r in runs if r.dev]
    covs = [r.cov for r in runs if r.cov]
    if len(devs) == len(runs):
        for m in ("MASE", "CRPS"):
            a.dev[m] = gmean(d["scores"]["main"]["overall"][m][0] for d in devs)
        sl: dict = {}
        for d in devs:
            for s in d["scores"]["main"]["slices"]:
                sl.setdefault(f"{s['slice']}={s['value']}", []).append(s[METRIC])
        a.slices = {k: gmean(v) for k, v in sl.items() if len(v) == len(devs)}
        a.domains = {k.split("=", 1)[1]: v for k, v in a.slices.items() if k.startswith("domain=")}
        a.keys |= {("dev", d["version"], d["set_sha256"]) for d in devs}
    if len(covs) == len(runs):
        for m in ("MASE", "CRPS"):
            a.cov_on[m] = gmean(c["scores"]["on"]["overall"][m][0] for c in covs)
        a.cov_gain = float(np.mean([1 - c["on_vs_hidden"]["ratio"] for c in covs]))
        a.keys |= {("cov", c["version"], c["set_sha256"]) for c in covs}
    pf = [r.probes for r in runs if r.probes is not None]
    if len(pf) == len(runs):
        cols = ["probe", "cell", "mode", "view", "metric", "control"]
        a.seed_probes = pf
        a.probes = pd.concat(_variant(f) for f in pf).groupby(cols, as_index=False, dropna=False)["value"].mean()
    return a


# ------------------------------------------------------------------------------------------------- the win rule

def gap_probe(exp: str) -> str | None:
    """G<k>.x -> P<k> (EXPERIMENTS.md: each gap section names its probe P<k>)."""
    m = re.match(r"G(\d+)\.", exp)
    return f"P{m.group(1)}" if m else None


def probe_mean(a: Agg, probe: str, cells=(), metric="sql", mode=None) -> float:
    if a.probes is None:
        return math.nan
    p = a.probes[(a.probes["probe"] == probe) & (a.probes["metric"] == metric) & (a.probes["view"] == "main")]
    if mode is None:
        mode = "on" if (p["mode"] == "on").any() else "main"
    p = p[p["mode"] == mode]
    if cells:
        p = p[p["cell"].astype(str).str.split("@").str[0].isin(cells)]
    else:
        p = p[~p["control"].astype(str).str.lower().isin(("true", "1"))]
    return float(p["value"].mean()) if len(p) else math.nan


@dataclass(frozen=True)
class Main:
    """target: the value is |metric - target|; cells None = the non-control cells; lower is better unless high."""
    label: str
    metric: str
    mode: str = "main"
    cells: tuple | None = None
    target: float | None = None
    high: bool = False
    variant: str = "baseline"


MAIN = {"P1": [Main("sql", "sql", "on")],
        "P2": [Main("event_sql", "event_sql", "on", ("holiday", "moving_festival"))],
        "P3": [Main("event_score", "event_score", high=True)],
        "P4": [Main("step adoption err", "adoption", cells=("step_up", "step_down"), target=1.0),
               Main("transient adoption err", "adoption", cells=("transient",), target=0.0)],
        "P5": [Main("sql", "sql")],
        "P6": [Main("gaps/clean", "sql_gaps_vs_clean"),
               Main("gaps/clean nofill_mask", "sql_gaps_vs_clean", variant="nofill_mask")],
        "P7": [Main("zero_mass_err", "zero_mass_err")],
        "P8": [Main("coherence", "coherence", "on")],
        "P9": [Main("sql", "sql")],
        "P10": [Main("pit_tv", "pit_tv"), Main("cov80 err", "cov80", target=0.8)]}


def main_value(f: pd.DataFrame, probe: str, m: Main) -> float:
    p = _variant(f, m.variant)
    p = p[(p["probe"] == probe) & (p["metric"] == m.metric) & (p["mode"] == m.mode) & (p["view"] == "main")]
    if m.cells is None:
        p = p[~p["control"].astype(str).str.lower().isin(("true", "1"))]
    else:
        p = p[p["cell"].astype(str).str.split("@").str[0].isin(m.cells)]
    v = p["value"].astype(float)
    return float((v if m.target is None else (v - m.target).abs()).mean()) if len(p) else math.nan


def main_lines(a: Agg, b: Agg, probe: str) -> list[dict]:
    out = []
    for m in MAIN.get(probe, []):
        av = float(np.mean([main_value(f, probe, m) for f in a.seed_probes])) if a.seed_probes else math.nan
        bv = [main_value(f, probe, m) for f in b.seed_probes]
        lo, hi = (min(bv), max(bv)) if bv else (math.nan, math.nan)
        if av != av or lo != lo:
            v = "n/a"
        elif lo <= av <= hi:
            v = "no effect"
        else:
            v = "improves" if (av > hi) == m.high else "worse"
        out.append(dict(probe=probe, label=m.label, value=av, lo=lo, hi=hi, verdict=v))
    return out


def _line(x: dict) -> str:
    return f"{x['label']} {_f(x['value'])} [{_f(x['lo'])}, {_f(x['hi'])}]"


P1_KIND = dict(lagged_linear="known", nonlinear_u="known", promo="known", events="known", lead_past="past",
               lag0_as_past="past", decoy="control")


def _cell_values(frames: list, probe: str, cell: str, metric: str, mode: str) -> list[float]:
    out = []
    for f in frames:
        d = _variant(f)
        v = d[(d["probe"] == probe) & (d["cell"].astype(str) == cell) & (d["metric"] == metric) & (d["mode"] == mode)
              & (d["view"] == "main")]["value"]
        if len(v):
            out.append(float(v.iloc[-1]))
    return out


def cell_table(a: Agg, b: Agg, probe: str) -> str:
    """Per cell: sql and the ON / hidden ratio, A against B0's mean and seed range."""
    rows = [f"# {probe} per cell: {a.exp} vs {b.exp} ({len(b.seeds)} seeds)", "",
            "| Cell | Kind | Metric | A | B0 mean | B0 range | A / B0 |", "|---|---|---|---|---|---|---|"]
    p = _variant(pd.concat(b.seed_probes))
    cells = p.loc[p["probe"] == probe, "cell"].astype(str).unique()
    for cell in sorted(cells, key=lambda c: (P1_KIND.get(c.split("@")[0], ""), c)):
        for metric, mode in (("sql", "on"), ("sql", "hidden"), ("sql", "main"), ("sql_on_vs_hidden", "on")):
            av, bv = _cell_values(a.seed_probes, probe, cell, metric, mode), _cell_values(b.seed_probes, probe, cell, metric, mode)
            if av and bv:
                A, B = float(np.mean(av)), float(np.mean(bv))
                rows.append(f"| {cell} | {P1_KIND.get(cell.split('@')[0], '')} | {metric}@{mode} | {A:.4f} | {B:.4f} | "
                            f"[{min(bv):.4f}, {max(bv):.4f}] | {A / B:.4f} |")
    return "\n".join(rows) + "\n"


def gain(a: float, b: float) -> float:
    return 1 - a / b if a == a and b == b else math.nan


def criterion(c: dict, a: Agg, b: Agg, noise: dict | None) -> tuple[str, str]:
    """(PASS | FAIL | UNKNOWN, text)."""
    if "score" in c:
        part = c["score"]
        av, bv = (a.dev, b.dev) if part == "dev" else (a.cov_on, b.cov_on)
        g = gain(av.get(METRIC, math.nan), bv.get(METRIC, math.nan))
        fl = NZ.floor(noise, METRIC, part)
        test = "beat" if "beat" in c else "not_worse"
        name = f"{part} {'beats B0 by more than' if test == 'beat' else 'not worse than B0 by more than'} E0.7's range"
        if g != g or fl is None:
            return "UNKNOWN", f"{name}: gain {_pct(g)}, range {'missing' if fl is None else _pct(fl)}"
        ok = g > fl if test == "beat" else g > -fl
        return ("PASS" if ok else "FAIL"), f"{name}: gain {_pct(g)} vs range {_pct(fl)}"
    if "probe" in c:
        av = probe_mean(a, c["probe"], c["cells"], c["metric"], c["mode"])
        bv = probe_mean(b, c["probe"], c["cells"], c["metric"], c["mode"])
        name = f"{c['probe']} {c['metric']}@{c['mode']} ({', '.join(c['cells']) or 'non-control cells'}) improves"
        if av != av or bv != bv:
            return "UNKNOWN", f"{name}: probe results missing"
        return ("PASS" if av < bv else "FAIL"), f"{name}: {av:.4f} vs B0 {bv:.4f}"
    if "slice" in c:
        k = c["slice"]
        av, bv = a.slices.get(k, math.nan), b.slices.get(k, math.nan)
        if av != av or bv != bv:
            return "UNKNOWN", f"own slice {k}: missing"
        return ("PASS" if av < bv else "FAIL"), f"own slice {k}: A / B0 {av / bv:.4f}"
    tol = float(c["domains_within"])
    common = sorted(set(a.domains) & set(b.domains))
    if not common:
        return "UNKNOWN", "domain slices missing"
    worse = [f"{d} {a.domains[d] / b.domains[d]:.4f}" for d in common if a.domains[d] / b.domains[d] > 1 + tol]
    return ("FAIL" if worse else "PASS"), f"no domain slice worse than B0 by more than {100 * tol:.0f}%" + (
        f": {', '.join(worse)}" if worse else "")


def verdict(results: list[str], seeds: int, comparable: bool) -> str:
    if not comparable:
        return "not comparable (other dev version / set than B0)"
    if "FAIL" in results:
        return "no win"
    if "UNKNOWN" in results:
        return "unknown"
    return "win" if seeds >= 3 else "win, 1-seed screen: confirm with 3 seeds" if seeds == 1 else f"win, {seeds} seeds: confirm with 3"


def _pct(x: float) -> str:
    return "n/a" if x != x else f"{100 * x:+.2f}%"


def _f(x: float) -> str:
    return "n/a" if x != x else f"{x:.4f}"


# ------------------------------------------------------------------------------------------------- the board

def evaluate(a: Agg, b: Agg, noise: dict | None, win) -> dict:
    res = [criterion(c, a, b, noise) for c in win]
    comparable = not (a.keys - b.keys)
    probe = gap_probe(a.exp)
    pa, pb = (probe_mean(a, probe), probe_mean(b, probe)) if probe else (math.nan, math.nan)
    lines = main_lines(a, b, probe) if probe else []
    worst = max(((d, a.domains[d] / b.domains[d]) for d in set(a.domains) & set(b.domains)), key=lambda t: t[1],
                default=("", math.nan))
    return dict(exp=a.exp, seeds=len(a.seeds), dev_MASE=a.dev.get("MASE", math.nan), dev_CRPS=a.dev.get("CRPS", math.nan),
                dev_gain=gain(a.dev.get(METRIC, math.nan), b.dev.get(METRIC, math.nan)),
                cov_MASE=a.cov_on.get("MASE", math.nan),
                cov_gain_vs_b0=gain(a.cov_on.get(METRIC, math.nan), b.cov_on.get(METRIC, math.nan)),
                on_vs_hidden=a.cov_gain, probe=probe or "", probe_ratio=pa / pb if pb == pb and pb else math.nan,
                probe_value=pa, probe_b0=pb, main=[_line(x) for x in lines],
                probe_verdict=" / ".join(x["verdict"] for x in lines), worst_domain=worst[0], worst_domain_ratio=worst[1],
                criteria=[f"{s} {t}" for s, t in res], verdict=verdict([s for s, _ in res], len(a.seeds), comparable))


def build(results: str | Path, noise_path: str | None, version: str = "dev-v1", b0: str = "E0.7") -> dict:
    from tideforge.expcfg import DEFAULT_WIN, load_exp
    finals, miles = read(results, version)
    noise = _json(Path(noise_path)) if noise_path else None
    by: dict = {}
    for r in finals:
        by.setdefault(r.exp, []).append(r)
    if b0 not in by:
        raise SystemExit(f"no final rows of the baseline {b0} in {results}")
    B = aggregate(b0, by[b0])
    rows = []
    for exp in sorted(by, key=_order):
        if exp == b0:
            continue
        try:
            win = load_exp(exp).win
        except FileNotFoundError:
            win = DEFAULT_WIN
        rows.append(evaluate(aggregate(exp, by[exp]), B, noise, win))
    runs = [dict(exp=r.exp, seed=r.seed, run_id=r.row["run_id"], code=r.row.get("code", ""), date=r.row.get("date", ""),
                 dev_MASE=r.row.get("dev_MASE", ""), dev_cov_MASE=r.row.get("dev_cov_MASE", ""),
                 dev_cov_gain=r.row.get("dev_cov_gain", ""), probes=r.row.get("probes", "")) for r in finals]
    dates = {}
    for r in finals:
        dates[r.exp] = max(dates.get(r.exp, ""), r.row.get("date", ""))
    aggs = {e: aggregate(e, v) for e, v in by.items()}
    controls = {e: [x for p in MAIN for x in main_lines(aggs[e], B, p)] for e in ("E0.8",) if e in aggs}
    return dict(rows=rows, runs=runs, milestones=miles, b0=B, b0_name=b0, controls=controls, aggs=aggs, noise=noise, dates=dates, version=version)


def _order(exp: str):
    return [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", exp)]


def board_md(b: dict) -> str:
    fl = NZ.floor(b["noise"], METRIC)
    flc = NZ.floor(b["noise"], METRIC, "dev_cov")
    lines = [f"# Results board ({b['version']}, vs {b['b0_name']}: {len(b['b0'].seeds)} seeds; E0.7 range {METRIC}: dev "
             f"{_pct(fl) if fl is not None else 'missing'}, dev_cov {_pct(flc) if flc is not None else 'missing'})", "",
             "| Exp | Seeds | Dev MASE | Gain vs B0 | Dev-cov MASE | Gain vs B0 | ON vs hidden | Probe | Main metric [B0 range] | "
             "Probe verdict | SQL A/B0 | Worst domain | Verdict |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in b["rows"]:
        lines.append(f"| {r['exp']} | {r['seeds']} | {_f(r['dev_MASE'])} | {_pct(r['dev_gain'])} | {_f(r['cov_MASE'])} | "
                     f"{_pct(r['cov_gain_vs_b0'])} | {_pct(r['on_vs_hidden'])} | "
                     f"{r['probe'] or '-'} | {'; '.join(r['main']) or '-'} | {r['probe_verdict'] or '-'} | "
                     f"{_f(r['probe_ratio']) if r['probe'] else '-'} | "
                     f"{(r['worst_domain'] + ' ' + _f(r['worst_domain_ratio'])) if r['worst_domain'] else 'n/a'} | {r['verdict']} |")
    for e, ctl in b.get("controls", {}).items():
        n = sum(x["verdict"] == "worse" for x in ctl)
        lines += ["", f"## Control {e} on every probe: {n} of {sum(x['verdict'] != 'n/a' for x in ctl)} metrics worse than B0's range",
                  "", "| Probe | Metric | Value | B0 range | Against the range |", "|---|---|---|---|---|"]
        lines += [f"| {x['probe']} | {x['label']} | {_f(x['value'])} | [{_f(x['lo'])}, {_f(x['hi'])}] | {x['verdict']} |" for x in ctl]
    return "\n".join(lines) + "\n"


def part_b(r: dict, b: dict) -> str:
    B = b["b0"]
    fl, flc = NZ.floor(b["noise"], METRIC), NZ.floor(b["noise"], METRIC, "dev_cov")
    rng = lambda x: _pct(x) if x is not None else "missing"          # noqa: E731
    lines = [BEGIN, f"Generated by tideforge.campaign.board from results.csv ({b['version']}; {r['seeds']} seed(s); B0 = "
             f"{b['b0_name']}, {len(B.seeds)} seeds). Do not edit between the markers.", "",
             "| Metric | B0 | This run | Change | Noise floor |", "|---|---|---|---|---|",
             f"| Fast dev score (MASE) | {_f(B.dev.get('MASE', math.nan))} | {_f(r['dev_MASE'])} | {_pct(r['dev_gain'])} | {rng(fl)} |",
             f"| Dev covariate (MASE, ON) | {_f(B.cov_on.get('MASE', math.nan))} | {_f(r['cov_MASE'])} | "
             f"{_pct(r['cov_gain_vs_b0'])} | {rng(flc)} |",
             f"| ON vs hidden gain | {_pct(B.cov_gain)} | {_pct(r['on_vs_hidden'])} | | |",
             f"| Probe {r['probe'] or '-'} (sql) | {_f(r['probe_b0'])} | {_f(r['probe_value'])} | "
             f"{('x ' + _f(r['probe_ratio'])) if r['probe'] else ''} | |",
             *[f"| Probe main: {m} | | | {r['probe_verdict']} | B0 range in brackets |" for m in r["main"]],
             f"| Worst domain slice (A / B0) | | {r['worst_domain'] or 'n/a'} | {_f(r['worst_domain_ratio'])} | 1% |", "",
             "Win rule (mechanical):"]
    lines += [f"- {c}" for c in r["criteria"]]
    lines += ["", f"Rule verdict: **{r['verdict']}**", END]
    return "\n".join(lines)


def write_part_b(path: Path, block: str) -> str:
    s = path.read_text(encoding="utf-8")
    if BEGIN in s and END in s:
        i, j = s.index(BEGIN), s.index(END) + len(END)
        s = s[:i] + block + s[j:]
    else:
        k = s.find("**Results:**")
        if k < 0:
            return "no **Results:** line: not written"
        head_end = s.index("\n", k) + 1
        rest = s[head_end:].lstrip("\n")
        tbl = re.match(r"(\|[^\n]*\n)+", rest)                         # the template's empty table, replaced
        rest = rest[tbl.end():] if tbl else rest
        s = s[:head_end] + "\n" + block + "\n\n" + rest.lstrip("\n")
    path.write_text(s, encoding="utf-8", newline="\n")
    return "written"


def index_row(r: dict, date: str, repo: Path = REPO) -> str:
    status = "confirmed" if r["seeds"] >= 3 else "screened"
    log = f"[{r['exp']}](experiments/{r['exp']}.md)" if (repo / "experiments" / f"{r['exp']}.md").exists() else "-"
    res = f"dev {_pct(r['dev_gain'])} vs B0, dev-cov {_pct(r['cov_gain_vs_b0'])}; {r['verdict']}"
    return f"| {r['exp']} | {date} | {status} | {res} | {log} {ROWTAG}|"


def write_index(path: Path, rows: list[tuple[str, str]]) -> list[str]:
    """rows: (exp, row text). Board rows are replaced in place; new ones go first (newest first); hand rows are left alone."""
    lines = path.read_text(encoding="utf-8").split("\n")
    h = lines.index(INDEX_HEAD)
    notes, new = [], []
    for exp, text in rows:
        pre = f"| {exp} |"
        hits = [i for i in range(h + 2, len(lines)) if lines[i].startswith(pre)]
        board = [i for i in hits if ROWTAG in lines[i]]
        if board:
            lines[board[0]] = text
        elif hits:
            notes.append(f"{exp}: a hand-written index row exists; left alone")
        else:
            new.append(text)
    lines[h + 2:h + 2] = new
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
    return notes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True)
    ap.add_argument("--noise", default=None)
    ap.add_argument("--version", default="dev-v1")
    ap.add_argument("--b0", default="E0.7")
    ap.add_argument("--out", default="runs/board")
    ap.add_argument("--write", action="store_true", help="also write the Part B tables and the index rows")
    ap.add_argument("--cells", action="append", default=[], metavar="PROBE:EXP", help="also write a per-cell table, e.g. P1:G1.13a")
    ap.add_argument("--repo", default=str(REPO), help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    b = build(a.results, a.noise, a.version, a.b0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    md = board_md(b)
    (out / "board.md").write_text(md, encoding="utf-8", newline="\n")
    pd.DataFrame([{k: (" ; ".join(v) if isinstance(v, list) else v) for k, v in r.items()} for r in b["rows"]]).to_csv(
        out / "board.csv", index=False)
    pd.DataFrame(b["runs"]).to_csv(out / "runs.csv", index=False)
    pd.DataFrame(b["milestones"], columns=["exp", "seed", "run_id", "points_B", "dev_MASE", "ckpt"]).to_csv(
        out / "milestones.csv", index=False)
    print(md)
    for spec in a.cells:
        probe, exp = spec.split(":")
        t = cell_table(b["aggs"][exp], b["b0"], probe)
        (out / f"cells_{probe}_{exp}.md").write_text(t, encoding="utf-8", newline="\n")
        print(t)
    if a.write:
        repo = Path(a.repo)
        for r in b["rows"]:
            p = repo / "experiments" / f"{r['exp']}.md"
            print(f"{r['exp']}: Part B {write_part_b(p, part_b(r, b)) if p.exists() else 'no log file: not written'}")
        for n in write_index(repo / "EXPERIMENTS.md", [(r["exp"], index_row(r, b["dates"][r["exp"]], repo)) for r in b["rows"]
                                                         if r["exp"] != a.b0]):
            print(n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
