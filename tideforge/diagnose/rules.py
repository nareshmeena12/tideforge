"""D4 feature rules: when do we lose? Fixable error (D2) against the window features (features.py), numpy only.

    python -m tideforge.diagnose.rules --fixable runs/diagnose/dev-v1/d2/fixable_windows.csv --dev $TF_DATA/dev --version dev-v1 \
        [--out runs/diagnose/dev-v1/d4]

binned(df, f)   per quantile bin of one feature (5 bins; a categorical feature by value): the mean fixable error per window, every
                dataset weighing the same (window weight 1 / windows of its dataset), its difference from the overall mean (the
                effect) and a CI from resampling datasets
tree(df, fs)    a shallow regression tree (CART on weighted squared error, depth 3, leaves of at least 5% of the weight)
report(...)     "we lose when ..." lines: the bins whose effect CI excludes 0, largest effect first
The target is fix = ours - best reference (signed: negative where we are better), so a rule says where the gap is, not only
where we are bad.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.diagnose import _boot
from tideforge.eval.devmetrics import BOOT_SEED, N_BOOT


def _dataset_bin_means(fix, ds, mask):
    """Per dataset: mean fix over its windows in the bin (NaN where the dataset has none there)."""
    ids, inv = np.unique(ds, return_inverse=True)
    s = np.bincount(inv, weights=np.where(mask, fix, 0.0), minlength=len(ids))
    c = np.bincount(inv, weights=mask.astype(float), minlength=len(ids))
    return np.where(c > 0, s / np.maximum(c, 1), np.nan), np.bincount(inv, weights=fix, minlength=len(ids)) / np.bincount(inv, minlength=len(ids))


def binned(df: pd.DataFrame, feature: str, q: int = 5, target: str = "fix", n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> pd.DataFrame:
    x = df[feature]
    fix, ds = df[target].to_numpy(np.float64), df["dataset_id"].to_numpy()
    if x.dtype.kind in "OUSb":
        bins = [(str(v), (x == v).to_numpy()) for v in sorted(x.dropna().unique(), key=str)]
    else:
        xv = x.to_numpy(np.float64)
        ok = np.isfinite(xv)
        edges = np.unique(np.quantile(xv[ok], np.linspace(0, 1, q + 1))) if ok.any() else np.array([])
        bins = []
        for i in range(len(edges) - 1):
            lo, hi = edges[i], edges[i + 1]
            m = ok & (xv >= lo) & ((xv <= hi) if i == len(edges) - 2 else (xv < hi))
            bins.append((f"[{lo:.3g}, {hi:.3g}]", m))
    rng = np.random.default_rng(seed)
    rows = []
    for label, m in bins:
        bm, allm = _dataset_bin_means(fix, ds, m)
        have = np.isfinite(bm)
        if have.sum() == 0:
            continue
        eff = bm[have] - allm[have]                                   # within-dataset contrast: bin vs the dataset's mean
        idx = rng.integers(0, have.sum(), size=(n_boot, have.sum()))
        b = eff[idx].mean(1)
        rows.append(dict(feature=feature, bin=label, windows=int(m.sum()), datasets=int(have.sum()), mean_fix=float(bm[have].mean()),
                         effect=float(eff.mean()), lo=float(np.quantile(b, 0.025)), hi=float(np.quantile(b, 0.975))))
    return pd.DataFrame(rows)


@dataclass
class Node:
    n_weight: float
    value: float
    feature: str | None = None
    thr: float | None = None
    left: "Node | None" = None
    right: "Node | None" = None
    children: list = field(default_factory=list)

    def lines(self, depth=0) -> list[str]:
        pad = "  " * depth
        if self.feature is None:
            return [f"{pad}leaf: mean fix {self.value:+.4f} (weight {self.n_weight:.3f})"]
        return ([f"{pad}{self.feature} <= {self.thr:.4g}:"] + self.left.lines(depth + 1) +
                [f"{pad}{self.feature} > {self.thr:.4g}:"] + self.right.lines(depth + 1))


def tree(df: pd.DataFrame, features: list[str], depth: int = 3, min_leaf: float = 0.05, target: str = "fix") -> Node:
    """Weighted CART (squared error); weights 1 / windows of the dataset, normalised to sum 1; NaN features go left."""
    X = df[features].to_numpy(np.float64)
    y = df[target].to_numpy(np.float64)
    w = _boot.window_weights(df["dataset_id"])
    w = w / w.sum()

    def grow(idx, d):
        ww, yy = w[idx], y[idx]
        node = Node(float(ww.sum()), float((ww * yy).sum() / ww.sum()))
        if d == depth:
            return node
        best = None
        for j, f in enumerate(features):
            xj = np.nan_to_num(X[idx, j], nan=-np.inf)
            o = np.argsort(xj, kind="stable")
            xs, ws, ys = xj[o], ww[o], yy[o]
            cw, cwy, cwy2 = np.cumsum(ws), np.cumsum(ws * ys), np.cumsum(ws * ys * ys)
            W, WY, WY2 = cw[-1], cwy[-1], cwy2[-1]
            for k in np.flatnonzero((xs[1:] != xs[:-1])):
                lw, rw = cw[k], W - cw[k]
                if lw < min_leaf or rw < min_leaf:
                    continue
                sse = (cwy2[k] - cwy[k] ** 2 / lw) + ((WY2 - cwy2[k]) - (WY - cwy[k]) ** 2 / rw)
                if best is None or sse < best[0]:
                    best = (sse, j, (xs[k] + xs[k + 1]) / 2 if np.isfinite(xs[k]) else xs[k + 1] - 1e-9)
        parent = (ww * yy * yy).sum() - (ww * yy).sum() ** 2 / ww.sum()
        if best is None or best[0] >= parent - 1e-12:
            return node
        _, j, thr = best
        xj = np.nan_to_num(X[idx, j], nan=-np.inf)
        node.feature, node.thr = features[j], float(thr)
        node.left, node.right = grow(idx[xj <= thr], d + 1), grow(idx[xj > thr], d + 1)
        return node
    return grow(np.arange(len(y)), 0)


def report(tables: list[pd.DataFrame], top: int = 12) -> list[str]:
    t = pd.concat(tables, ignore_index=True)
    sig = t[(t["lo"] > 0) | (t["hi"] < 0)].copy()
    sig = sig.reindex(sig["effect"].abs().sort_values(ascending=False).index).head(top)
    out = []
    for r in sig.itertuples():
        verb = "we lose more" if r.effect > 0 else "we lose less"
        out.append(f"{verb} when {r.feature} in {r.bin}: {r.effect:+.4f} fixable error per window vs the dataset's mean "
                   f"[{r.lo:+.4f}, {r.hi:+.4f}], {r.windows} windows in {r.datasets} datasets")
    return out


def main(argv=None) -> int:
    from tideforge.diagnose import features as FE
    from tideforge.eval import devscore as DS
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixable", required=True, help="D2's fixable_windows.csv")
    ap.add_argument("--dev", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--set", default="fast")
    ap.add_argument("--out", default="runs/diagnose/d4")
    a = ap.parse_args(argv)
    fx = pd.read_csv(a.fixable)
    v = DS.Version(a.dev, a.version)
    win = pd.read_parquet(v.set_path("tuning", a.set))
    feats = FE.frame(DS.main_items(v, win)).drop(columns=["domain", "term"], errors="ignore")
    df = fx.merge(feats, on=["dataset_id", "series_id", "window"], how="inner")
    tabs = [binned(df, f) for f in FE.FEATURES + ["freq"] if f in df.columns]
    t = tree(df, [f for f in FE.FEATURES if f in df.columns])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.concat(tabs, ignore_index=True).to_csv(out / "bins.csv", index=False)
    text = "# D4: we lose when ...\n\n" + "\n".join(f"- {s}" for s in report(tabs)) + "\n\n## Tree\n\n```\n" + \
        "\n".join(t.lines()) + "\n```\n"
    (out / "rules.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
