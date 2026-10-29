"""D2 fixable error: where a reference model does better than ours, per window (dev tuning set) or per task (fev). Never TIME
(owner 2026-10-29: designing generators around TIME's worst windows would be selecting on TIME).

    python -m tideforge.diagnose.fixable dev --ours runs/dev/dev-v1/tuning_fast/m5 --ref runs/dev/dev-v1/tuning_fast/chronos2 \
        [--ref ...] [--metric CRPS|MASE] [--out runs/diagnose/dev-v1/d2]
    python -m tideforge.diagnose.fixable fev --table docs/report/data/fev_rel_errors.csv --ours ft3_main --ref Chronos-2 [...]

Dev: the scaled error of a window is metric / metric of SeasonalNaiveTIME on the same window (devscore's windows.parquet,
the same ratio devscore aggregates); fix = ours - min over the references (who is best is kept). Only fix > 0 is fixable.
Every dataset weighs the same: a window's weight is 1 / (windows of its dataset), and a dataset's share of the total fixable
error is sum(w fix+) over its windows / sum(w fix+) over all. fev: one row per task (tasks weigh the same), the error is fev's
relative error to Seasonal Naive (D1's table, tools/fev_cov_split.py).
Outputs: fixable_windows.csv (every window, ranked), fixable_datasets.csv (share, cumulative share, CI over windows), summary.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.diagnose import _boot

KEY = ["dataset_id", "series_id", "window", "mode"]


def load_dev_run(d: str | Path) -> tuple[dict, pd.DataFrame]:
    d = Path(d)
    run = json.loads((d / "run.json").read_text(encoding="utf-8"))
    if run.get("role") != "tuning":
        raise ValueError(f"{d}: D2 runs on the dev tuning set only (role {run.get('role')!r})")
    return run, pd.read_parquet(d / "windows.parquet")


def scaled(win: pd.DataFrame, metric: str) -> pd.Series:
    return win[metric] / win[f"{metric}_naive"]


def fixable(ours: pd.DataFrame, refs: dict[str, pd.DataFrame], metric: str = "CRPS") -> pd.DataFrame:
    """One row per window: ours, each reference, best reference, fix, weight, fixable share."""
    df = ours[KEY + ["domain", "freq", "term", "hist", "cov", "ctx_len"]].copy()
    df["ours"] = scaled(ours, metric).to_numpy()
    for name, r in refs.items():
        s = r.assign(_s=scaled(r, metric))[KEY + ["_s"]].rename(columns={"_s": name})
        df = df.merge(s, on=KEY, how="left", validate="one_to_one")
    missing = df[list(refs)].isna().all(1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} windows of ours have no reference score (different sets?)")
    R = df[list(refs)].to_numpy()
    df["best_ref"] = np.array(list(refs))[np.nanargmin(R, 1)]
    df["ref_best"] = np.nanmin(R, 1)
    df["fix"] = df["ours"] - df["ref_best"]
    df["w"] = _boot.window_weights(df["dataset_id"])
    pos = df["w"] * df["fix"].clip(lower=0)
    df["share"] = pos / pos.sum() if pos.sum() > 0 else 0.0
    return df.sort_values("share", ascending=False, kind="stable").reset_index(drop=True)


def by_dataset(df: pd.DataFrame, n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """Per dataset: share of the total fixable error (CI from resampling its windows), mean fix, the reference that is most often best."""
    tot = float((df["w"] * df["fix"].clip(lower=0)).sum())
    rng = np.random.default_rng(seed)
    rows = []
    for d, g in df.groupby("dataset_id", sort=False):
        fp = g["fix"].clip(lower=0).to_numpy()
        w = g["w"].to_numpy()
        share = float((w * fp).sum() / tot) if tot > 0 else 0.0
        b = fp[rng.integers(0, len(fp), size=(n_boot, len(fp)))].mean(1) * w.sum() / tot if tot > 0 else np.zeros(n_boot)
        rows.append(dict(dataset_id=d, domain=g["domain"].iloc[0], windows=len(g), share=share,
                         share_lo=float(np.quantile(b, 0.025)), share_hi=float(np.quantile(b, 0.975)),
                         mean_fix=float(g["fix"].mean()), frac_windows_fixable=float((g["fix"] > 0).mean()),
                         best_ref=g["best_ref"].mode().iloc[0]))
    out = pd.DataFrame(rows).sort_values("share", ascending=False, kind="stable").reset_index(drop=True)
    out["cum_share"] = out["share"].cumsum()
    return out


def from_fev_table(table: str | Path, ours: str, refs: list[str]) -> pd.DataFrame:
    """fev: D1's per-task relative errors (index = task key, one column per model). One row per task, every task weighs the same."""
    t = pd.read_csv(table, index_col=0)
    for c in [ours, *refs]:
        if c not in t.columns:
            raise KeyError(f"{c!r} not in {table} (columns: {list(t.columns)[:8]} ...)")
    df = pd.DataFrame({"task": t.index, "ours": t[ours].to_numpy()})
    R = t[refs].to_numpy()
    df["best_ref"] = np.array(refs)[np.nanargmin(R, 1)]
    df["ref_best"] = np.nanmin(R, 1)
    df["fix"] = df["ours"] - df["ref_best"]
    pos = df["fix"].clip(lower=0)
    df["share"] = pos / pos.sum() if pos.sum() > 0 else 0.0
    df = df.sort_values("share", ascending=False, kind="stable").reset_index(drop=True)
    df["cum_share"] = df["share"].cumsum()
    return df


def summary(df: pd.DataFrame, ds: pd.DataFrame | None, top: int = 15) -> str:
    lines = ["# D2 fixable error", ""]
    if ds is not None:
        m, lo, hi = _boot.mean_ci(df["fix"].clip(lower=0), df["dataset_id"])
        lines += [f"Mean fixable error per window (datasets equal): {m:.4f} [{lo:.4f}, {hi:.4f}]; windows where a reference is "
                  f"better: {100 * (df['fix'] > 0).mean():.1f}%.", "",
                  f"Datasets holding half the fixable error: {int((ds['cum_share'] < 0.5).sum()) + 1} of {len(ds)}.", "",
                  "| dataset | domain | share | CI | cum. | mean fix | best ref |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r.dataset_id} | {r.domain} | {r.share:.3f} | [{r.share_lo:.3f}, {r.share_hi:.3f}] | {r.cum_share:.3f} | "
                  f"{r.mean_fix:+.3f} | {r.best_ref} |" for r in ds.head(top).itertuples()]
    else:
        lines += ["| task | ours | best ref | fix | share | cum. |", "|---|---|---|---|---|---|"]
        lines += [f"| {r.task.split('|')[1] if '|' in r.task else r.task} | {r.ours:.3f} | {r.best_ref} {r.ref_best:.3f} | "
                  f"{r.fix:+.3f} | {r.share:.3f} | {r.cum_share:.3f} |" for r in df.head(top).itertuples()]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("dev", "fev"))
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ref", action="append", required=True)
    ap.add_argument("--metric", default="CRPS", choices=("CRPS", "MASE"))
    ap.add_argument("--table", default="docs/report/data/fev_rel_errors.csv")
    ap.add_argument("--out", default="runs/diagnose/d2")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.cmd == "dev":
        run, ours = load_dev_run(a.ours)
        refs = {}
        for r in a.ref:
            rr, w = load_dev_run(r)
            if (rr["version"], rr["set"], rr["set_sha256"]) != (run["version"], run["set"], run["set_sha256"]):
                raise SystemExit(f"{r}: a different dev version / set than {a.ours}")
            refs[rr["model"]] = w
        df = fixable(ours, refs, a.metric)
        ds = by_dataset(df)
        df.to_csv(out / "fixable_windows.csv", index=False)
        ds.to_csv(out / "fixable_datasets.csv", index=False)
    else:
        df, ds = from_fev_table(a.table, a.ours, a.ref), None
        df.to_csv(out / "fixable_tasks.csv", index=False)
    text = summary(df, ds)
    (out / "summary.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
