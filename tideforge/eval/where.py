"""Where a run helps: per-task log-ratio against B0, grouped by task property (owner 2026-11-01). Analysis only.

    python -m tideforge.eval.where meta   --root $TF_ROOT/eval_t0                          # once: <root>/eval/fev_meta.csv
    python -m tideforge.eval.where report --root $TF_ROOT/eval_t0 --a <tag> --b0 <tag> <tag> ... [--out f.md]

log_ratio = log(test_error A) - mean over the B0 tags of log(test_error B0); < 0: A is better. Only tasks every tag has.
Groups: all, domain (catalog/fev_domains.csv: assigned by hand from the dataset names), frequency (task name suffix), history band
(devmetrics.HIST_BANDS on the median series length of the first window), horizon band, covariates yes / no.
CI: 95% bootstrap over DATASETS within the group, because the tasks of one dataset are not independent. b0_band = the min..max over
the B0 tags of (that tag vs the mean of all B0 tags) in the group: differences inside it are what seeds alone give.
summarise() takes any table (task, dataset, log_ratio, group columns), so the dev-v1 report only needs its own loader.
Every report on fev is a look: add a LOOKS.md row (rule 5).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.eval.devmetrics import HIST_BANDS

REPO = Path(__file__).resolve().parents[2]
HORIZON_BANDS = ((1, 12, "1-12"), (13, 48, "13-48"), (49, 168, "49-168"), (169, None, "169+"))
GROUPS = ("domain", "frequency", "history", "horizon", "covariates")
N_BOOT, MIN_DATASETS = 2000, 3


def band(x: float, bands) -> str:
    return next((n for lo, hi, n in bands if x >= lo and (hi is None or x <= hi)), "other")


def dataset_of(task: str, names: list[str]) -> str:
    return max((n for n in names if task == n or task.startswith(n + "_")), key=len, default=task)


def frequency_of(task: str) -> str:
    m = re.search(r"_(\d*(?:T|H|D|W|M|Q|Y|A))$", task)
    return m.group(1) if m else "other"


def errors(root: Path, tag: str) -> pd.Series:
    out = {}
    for p in sorted((Path(root) / "eval" / "fev" / tag).glob("*.json")):
        s = json.loads(p.read_text())
        if s.get("test_error", 0) > 0:
            out[s["task_name"]] = float(s["test_error"])
    if not out:
        raise SystemExit(f"no fev results for {tag} under {root}/eval/fev")
    return pd.Series(out)


def log_ratios(a: pd.Series, b0: list[pd.Series]) -> tuple[pd.Series, pd.DataFrame]:
    """A vs the B0 mean per common task, and each B0 tag vs the same mean (columns: tags)."""
    B = pd.concat(b0, axis=1).dropna()
    common = B.index.intersection(a.index)
    lb = np.log(B.loc[common])
    mean = lb.mean(axis=1)
    return np.log(a[common]) - mean, lb.sub(mean, axis=0)


def boot_ci(v: np.ndarray, cluster: np.ndarray, seed: int = 0) -> tuple[float, float]:
    ids = np.unique(cluster)
    by = [v[cluster == i] for i in ids]
    rng = np.random.default_rng(seed)
    stats = [np.concatenate([by[j] for j in rng.integers(0, len(ids), len(ids))]).mean() for _ in range(N_BOOT)]
    return float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975))


def summarise(t: pd.DataFrame, seed_cols: tuple = (), groups: tuple = GROUPS) -> pd.DataFrame:
    rows = []
    parts = [("all", "all", t)] + [(g, str(k), d) for g in groups for k, d in t.groupby(g)]
    for g, k, d in parts:
        lo, hi = boot_ci(d["log_ratio"].to_numpy(), d["dataset"].to_numpy())
        s = [float(d[c].mean()) for c in seed_cols]
        ds = d["dataset"].nunique()
        rows.append(dict(group=g, value=k, tasks=len(d), datasets=ds, mean=float(d["log_ratio"].mean()), lo=lo, hi=hi,
                         b0_lo=min(s) if s else np.nan, b0_hi=max(s) if s else np.nan,
                         read=("few datasets" if ds < MIN_DATASETS else "better" if hi < 0 else "worse" if lo > 0 else "unclear")))
    return pd.DataFrame(rows)


def to_md(df: pd.DataFrame, title: str) -> str:
    lines = [f"# {title}", "", "log-ratio A / B0, < 0 is better; CI over datasets", "",
             "| Group | Value | Tasks | Datasets | Mean | 95% CI | B0 band | Read |", "|---|---|---|---|---|---|---|---|"]
    for r in df.itertuples():
        lines.append(f"| {r.group} | {r.value} | {r.tasks} | {r.datasets} | {r.mean:+.4f} | [{r.lo:+.4f}, {r.hi:+.4f}] | "
                     f"[{r.b0_lo:+.4f}, {r.b0_hi:+.4f}] | {r.read} |")
    return "\n".join(lines) + "\n"


def table(a: pd.Series, b0: list[pd.Series], meta: pd.DataFrame, tags: list[str]) -> pd.DataFrame:
    lr, seeds = log_ratios(a, b0)
    seeds.columns = [f"seed_{t}" for t in tags]
    t = meta.set_index("task").join(lr.rename("log_ratio"), how="inner").join(seeds, how="inner").reset_index()
    return t


def build_meta(root: str) -> pd.DataFrame:
    from tideforge.eval import thirdparty
    S = thirdparty.score_twins()
    S.setup_env(root, offline=True)
    names = pd.read_csv(REPO / "catalog" / "fev_domains.csv")
    domain = dict(zip(names["dataset"], names["domain"]))
    rows = []
    for t in S.load_fev_tasks(S.Paths(root)):
        S.prepare(t, 0)
        past, _ = next(iter(t.iter_windows())).get_input_data()
        n = np.median([len(y) for y in past[t.target_columns[0]]])
        ds = dataset_of(t.task_name, list(domain))
        rows.append(dict(task=t.task_name, dataset=ds, domain=domain.get(ds, "other"), frequency=frequency_of(t.task_name),
                         history=band(n, HIST_BANDS), horizon=band(t.horizon, HORIZON_BANDS),
                         covariates="yes" if (t.known_dynamic_columns or t.past_dynamic_columns) else "no"))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("meta", "report"))
    ap.add_argument("--root", required=True)
    ap.add_argument("--a")
    ap.add_argument("--b0", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    meta_csv = Path(a.root) / "eval" / "fev_meta.csv"
    if a.cmd == "meta":
        m = build_meta(a.root)
        m.to_csv(meta_csv, index=False)
        print(f"{len(m)} tasks -> {meta_csv}")
        return 0
    t = table(errors(a.root, a.a), [errors(a.root, x) for x in a.b0], pd.read_csv(meta_csv), a.b0)
    md = to_md(summarise(t, tuple(f"seed_{x}" for x in a.b0)), f"{a.a} vs B0 ({len(a.b0)} seeds), fev, {len(t)} tasks")
    if a.out:
        Path(a.out).write_text(md, encoding="utf-8", newline="\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
