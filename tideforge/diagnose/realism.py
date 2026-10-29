"""D6 realism classifier: how easily are synthetic windows told from real ones, by what, and for which families?

    python -m tideforge.diagnose.realism --dev $TF_DATA/dev --version dev-v1 [--exp G5.2a] [--per-family 400] [--cnn]
        [--out runs/diagnose/dev-v1/d6]

logistic   L2-regularised logistic regression (numpy IRLS) on the standardised features.SYNTH_COMPARABLE (season detected from
           each series on both sides; no freq / has_cov). Coefficients are per sd of a feature: what it uses. AUC by 5-fold CV:
           real windows folded by dataset (never in train and test), synthetic rows at random within each family (groups_of);
           held-out scores have the training fold's prior log-odds removed before they are pooled.
cnn        three 1-D conv layers (~20k parameters) on windows z-normalised and resampled to 256 points (np.interp), same folds;
           CPU in minutes (estimate), --device cuda on Perlmutter.
by_family  AUC of real vs each family's held-out rows (the easiest to tell apart first). 0.5 = indistinguishable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.diagnose import features as FE


def auc(score, label) -> float:
    s, y = np.asarray(score, np.float64), np.asarray(label, bool)
    if y.all() or (~y).all():
        return float("nan")
    order = np.argsort(s, kind="stable")
    r = np.empty(len(s))
    r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s):                                           # average ranks of ties
        m = s == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1 = y.sum()
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(s) - n1)))


def fit_logistic(X, y, lam: float = 1e-2, iters: int = 50) -> np.ndarray:
    Xb = np.column_stack([np.ones(len(X)), X])
    w = np.zeros(Xb.shape[1])
    R = lam * np.eye(Xb.shape[1])
    R[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Xb @ w, -30, 30)))
        g = Xb.T @ (p - y) + R @ w
        Hm = (Xb * (p * (1 - p))[:, None]).T @ Xb + R
        step = np.linalg.solve(Hm, g)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return w


def predict_logistic(w, X):
    return np.column_stack([np.ones(len(X)), X]) @ w


def groups_of(real_datasets, families, k: int = 5) -> np.ndarray:
    """Real windows grouped by dataset (no dataset in train and test); synthetic rows are independent draws, so they are split
    at random within each family (every family is in every training fold, which by_family needs)."""
    fam = np.asarray(families, str)
    return np.r_["real:" + np.asarray(real_datasets, str), np.char.add(np.char.add("syn:", fam), ":" + (np.arange(len(fam)) % k).astype(str))]


def folds(groups, k: int = 5, seed: int = 0) -> np.ndarray:
    g = np.asarray(groups)
    u = np.unique(g)
    perm = np.random.default_rng(seed).permutation(len(u))
    f = {v: perm[i] % k for i, v in enumerate(u)}
    return np.array([f[v] for v in g])


def standardise(X, ref=None):
    ref = X if ref is None else ref
    mu = np.nanmean(ref, 0)
    sd = np.nanstd(ref, 0) + 1e-9
    Z = (X - mu) / sd
    return np.nan_to_num(Z, nan=0.0), mu, sd


def logistic(real: pd.DataFrame, synth: pd.DataFrame, cols=None, k: int = 5) -> dict:
    """real needs dataset_id, synth needs family. Returns CV AUC, held-out scores and the full-data coefficients."""
    cols = list(cols or FE.SYNTH_COMPARABLE)
    X = np.vstack([real[cols].to_numpy(np.float64), synth[cols].to_numpy(np.float64)])
    y = np.r_[np.zeros(len(real)), np.ones(len(synth))]
    fo = folds(groups_of(real["dataset_id"], synth["family"], k), k)
    score = np.zeros(len(y))
    for i in range(k):
        tr, te = fo != i, fo == i
        if not te.any() or len(np.unique(y[tr])) < 2:
            continue
        Z, mu, sd = standardise(X[tr])
        w = fit_logistic(Z, y[tr])
        prior = np.log(y[tr].mean() / (1 - y[tr].mean()))             # folds by group differ in class share: remove the
        score[te] = predict_logistic(w, np.nan_to_num((X[te] - mu) / sd, nan=0.0)) - prior   # training prior before pooling
    Z, _, _ = standardise(X)
    w = fit_logistic(Z, y)
    coef = pd.DataFrame(dict(feature=cols, coef=w[1:])).reindex(np.argsort(-np.abs(w[1:]))).reset_index(drop=True)
    return dict(auc=auc(score, y), score=score, label=y, coef=coef, fold=fo)


def by_family(score, label, families) -> pd.DataFrame:
    """families: the synthetic rows' labels, in the order of the synthetic rows (after the real ones)."""
    score, label = np.asarray(score), np.asarray(label, bool)
    real_s = score[~label]
    syn_s = score[label]
    fam = np.asarray(families)
    rows = [dict(family=f, n=int((fam == f).sum()), auc=auc(np.r_[real_s, syn_s[fam == f]],
                                                              np.r_[np.zeros(len(real_s)), np.ones((fam == f).sum())]))
            for f in np.unique(fam)]
    return pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)


def resample(xs, n: int = 256) -> np.ndarray:
    out = np.zeros((len(xs), n), np.float32)
    for i, x in enumerate(xs):
        x = np.asarray(x, np.float64)
        x = x[np.isfinite(x)]
        if len(x) < 2:
            continue
        x = (x - x.mean()) / (x.std() + 1e-9)
        out[i] = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x)
    return out


def cnn(real_xs, synth_xs, groups, k: int = 5, epochs: int = 8, device: str = "cpu", seed: int = 0) -> dict:
    import torch
    from torch import nn
    X = np.vstack([resample(real_xs), resample(synth_xs)])
    y = np.r_[np.zeros(len(real_xs)), np.ones(len(synth_xs))].astype(np.float32)
    fo = folds(groups, k)
    score = np.zeros(len(y))
    dev = torch.device(device)

    def net():
        return nn.Sequential(nn.Conv1d(1, 16, 7, padding=3), nn.GELU(), nn.MaxPool1d(2),
                             nn.Conv1d(16, 32, 5, padding=2), nn.GELU(), nn.MaxPool1d(2),
                             nn.Conv1d(32, 64, 5, padding=2), nn.GELU(), nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                             nn.Linear(64, 1)).to(dev)
    for i in range(k):
        tr, te = np.flatnonzero(fo != i), np.flatnonzero(fo == i)
        if not len(te) or len(np.unique(y[tr])) < 2:
            continue
        torch.manual_seed(seed + i)
        m = net()
        opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
        Xt, yt = torch.from_numpy(X[tr]).to(dev).unsqueeze(1), torch.from_numpy(y[tr]).to(dev)
        pw = torch.tensor(float((y[tr] == 0).sum() / max(1, (y[tr] == 1).sum())), device=dev)
        g = torch.Generator().manual_seed(seed + i)
        for _ in range(epochs):
            for b in torch.randperm(len(tr), generator=g).split(256):
                loss = nn.functional.binary_cross_entropy_with_logits(m(Xt[b]).squeeze(1), yt[b], pos_weight=pw)
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            st = m(Xt).squeeze(1).cpu().numpy()
            mid = 0.5 * (st[y[tr] == 0].mean() + st[y[tr] == 1].mean())    # fold offsets differ: centre on the training midpoint
            score[te] = m(torch.from_numpy(X[te]).to(dev).unsqueeze(1)).squeeze(1).cpu().numpy() - mid
    per = [auc(score[fo == i], y[fo == i]) for i in range(k)]
    return dict(auc=auc(score, y), auc_fold_mean=float(np.nanmean(per)), score=score, label=y)


def main(argv=None) -> int:
    import tideforge.gen as G
    from tideforge.diagnose import synth
    from tideforge.eval import devscore as DS
    from tideforge.expcfg import load_exp
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--set", default="fast")
    ap.add_argument("--exp", default=None, help="an experiment config instead of B0")
    ap.add_argument("--per-family", type=int, default=400)
    ap.add_argument("--cnn", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="runs/diagnose/d6")
    a = ap.parse_args(argv)
    v = DS.Version(a.dev, a.version)
    items = list(DS.main_items(v, pd.read_parquet(v.set_path("tuning", a.set))))
    real_xs = [it["ctx"] for it in items]
    real = FE.comparable_frame(real_xs, [dict(dataset_id=it["dataset_id"]) for it in items])
    gc = load_exp(a.exp).gen if a.exp else G.b0()
    syn_xs, fams = synth.by_family(gc, [len(x) for x in real_xs], a.per_family)
    syn = FE.comparable_frame(syn_xs, [dict(family=f) for f in fams])
    lg = logistic(real, syn)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fam = by_family(lg["score"], lg["label"], fams)
    lines = [f"logistic AUC (grouped 5-fold CV): {lg['auc']:.3f}", "", "what it uses (coefficient per sd):",
             lg["coef"].head(12).to_string(index=False), "", "by family (easiest to tell apart first):", fam.to_string(index=False)]
    lg["coef"].to_csv(out / "logistic_coef.csv", index=False)
    fam.to_csv(out / "logistic_by_family.csv", index=False)
    if a.cnn:
        cn = cnn(real_xs, syn_xs, groups_of(real["dataset_id"], fams), device=a.device)
        cf = by_family(cn["score"], cn["label"], fams)
        cf.to_csv(out / "cnn_by_family.csv", index=False)
        lines += ["", f"CNN AUC (same folds): pooled {cn['auc']:.3f}, mean per fold {cn['auc_fold_mean']:.3f}", cf.to_string(index=False)]
    text = "\n".join(lines)
    (out / "summary.md").write_text("# D6 realism\n\n```\n" + text + "\n```\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
