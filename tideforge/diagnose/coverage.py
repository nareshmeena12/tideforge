"""D5 coverage map: where are real dev windows that no synthetic window resembles, and which config covers them best?

    python -m tideforge.diagnose.coverage --dev $TF_DATA/dev --version dev-v1 [--exp G4.1 --exp G5.2a ...] [--n-synth 20000]
        [--out runs/diagnose/dev-v1/d5]

Feature space: features.SYNTH_COMPARABLE (no freq, no has_cov: not defined for synthetic rows, owner 2026-10-29), with the season
detected from each series (features.detected_season) on both sides, not the real windows' frequency; each feature mapped to
its rank among the REAL windows (empirical CDF, 0..1; a missing value -> 0.5), so every feature counts on the same scale.
Per real window:
  nn_real   distance to the nearest real window of ANOTHER dataset (how close real data is to itself across datasets)
  nn_syn    distance to the nearest synthetic window of a config
  ratio     nn_syn / nn_real (> 1: synthetic data is sparser there than real data is)
A real window is uncovered by a config when nn_syn > the 95th percentile of nn_real. regions(): the uncovered windows (under B0)
grouped by domain and by the two features where the window sits furthest from its nearest synthetic neighbour, with each
config's share of them it covers. Distances: brute force in chunks (numpy), Euclidean.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.diagnose import features as FE

CHUNK = 2048


def rank_space(real: pd.DataFrame, others: dict[str, pd.DataFrame], cols=None):
    cols = list(cols or FE.SYNTH_COMPARABLE)
    srt = {c: np.sort(real[c].dropna().to_numpy(np.float64)) for c in cols}

    def tr(df):
        out = np.full((len(df), len(cols)), 0.5)
        for j, c in enumerate(cols):
            v = df[c].to_numpy(np.float64) if c in df else np.full(len(df), np.nan)
            ok = np.isfinite(v)
            if len(srt[c]):
                lo = np.searchsorted(srt[c], v[ok], "left")
                hi = np.searchsorted(srt[c], v[ok], "right")
                out[ok, j] = (lo + hi) / (2.0 * len(srt[c]))
        return out
    return tr(real), {k: tr(v) for k, v in others.items()}, cols


def nearest(A, B, exclude_a=None, exclude_b=None):
    """Per row of A: (distance, index) of the nearest row of B; rows with the same exclude label are skipped."""
    d = np.full(len(A), np.inf)
    ix = np.full(len(A), -1)
    b2 = (B * B).sum(1)
    for i in range(0, len(A), CHUNK):
        a = A[i:i + CHUNK]
        D = (a * a).sum(1)[:, None] + b2[None, :] - 2 * a @ B.T
        if exclude_a is not None:
            D = np.where(exclude_a[i:i + CHUNK, None] == exclude_b[None, :], np.inf, D)
        k = np.argmin(D, 1)
        d[i:i + CHUNK] = np.sqrt(np.maximum(D[np.arange(len(a)), k], 0))
        ix[i:i + CHUNK] = k
    return d, ix


def coverage(real: pd.DataFrame, synth: dict[str, pd.DataFrame], q: float = 0.95) -> tuple[pd.DataFrame, dict]:
    R, Ss, cols = rank_space(real, synth)
    ds = real["dataset_id"].to_numpy()
    nn_real, _ = nearest(R, R, ds, ds)
    thr = float(np.quantile(nn_real[np.isfinite(nn_real)], q))
    out = real[[c for c in ("dataset_id", "series_id", "window", "domain") if c in real]].copy()
    out["nn_real"] = nn_real
    near = {}
    for name, S in Ss.items():
        d, ix = nearest(R, S)
        out[f"nn_syn:{name}"] = d
        out[f"ratio:{name}"] = d / np.maximum(nn_real, 1e-9)
        out[f"uncovered:{name}"] = d > thr
        near[name] = (S[ix] - R)                                     # where the nearest synthetic window differs
    return out, dict(threshold=thr, cols=cols, gap=near)


def regions(cov: pd.DataFrame, info: dict, base: str = "b0", top: int = 20) -> pd.DataFrame:
    unc = cov[f"uncovered:{base}"].to_numpy()
    gap = np.abs(info["gap"][base])
    cols = np.array(info["cols"])
    two = np.sort(np.argsort(-gap, 1)[:, :2], 1)
    keys = [f"{cols[a]} + {cols[b]}" for a, b in two]
    df = cov.assign(region=keys)[unc]
    configs = [c.split(":", 1)[1] for c in cov.columns if c.startswith("uncovered:")]
    rows = []
    for (dom, reg), g in df.groupby([df.get("domain", pd.Series("all", index=df.index)), "region"]):
        r = dict(domain=dom, region=reg, windows=len(g), datasets=g["dataset_id"].nunique(),
                 share_of_uncovered=len(g) / max(1, len(df)))
        for c in configs:
            r[f"covered_by:{c}"] = float((~g[f"uncovered:{c}"]).mean())
        r["best_config"] = max(configs, key=lambda c: r[f"covered_by:{c}"])
        rows.append(r)
    return pd.DataFrame(rows).sort_values("windows", ascending=False).head(top).reset_index(drop=True)




def main(argv=None) -> int:
    import tideforge.gen as G
    from tideforge.diagnose import synth
    from tideforge.eval import devscore as DS
    from tideforge.expcfg import load_exp
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--set", default="fast")
    ap.add_argument("--exp", action="append", default=[])
    ap.add_argument("--n-synth", type=int, default=20000)
    ap.add_argument("--out", default="runs/diagnose/d5")
    a = ap.parse_args(argv)
    v = DS.Version(a.dev, a.version)
    win = pd.read_parquet(v.set_path("tuning", a.set))
    items = list(DS.main_items(v, win))
    real = FE.comparable_frame([it["ctx"] for it in items],
                               [dict(dataset_id=it["dataset_id"], series_id=it["series_id"], window=it["window"],
                                     domain=it["domain"]) for it in items])
    lengths = [len(it["ctx"]) for it in items]
    configs = {"b0": G.b0(), **{e: load_exp(e).gen for e in a.exp}}
    synth_feats = {k: FE.comparable_frame(synth.windows(gc, lengths, a.n_synth)) for k, gc in configs.items()}
    cov, info = coverage(real, synth_feats)
    reg = regions(cov, info)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cov.to_csv(out / "coverage_windows.csv", index=False)
    reg.to_csv(out / "regions.csv", index=False)
    lines = [f"threshold (95th pct of nn_real): {info['threshold']:.3f}"]
    lines += [f"{k}: uncovered {100 * cov[f'uncovered:{k}'].mean():.1f}% of real windows, median ratio "
              f"{cov[f'ratio:{k}'].median():.2f}" for k in configs]
    print("\n".join(lines))
    print(reg.to_string(index=False))
    (out / "summary.md").write_text("# D5 coverage\n\n" + "\n".join(f"- {s}" for s in lines) + "\n\n```\n" +
                                    reg.to_string(index=False) + "\n```\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
