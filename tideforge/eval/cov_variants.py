"""E0.10 inference-only covariate variants on fev's covariate tasks, groups ON (swapped in like nan_variants; nothing under
third_party/ is edited). Run inside e03_score.sbatch's queue (tideforge/slurm.py), one LOOKS row per variant.

What baseline does today (third_party score_groups.encode_column, ft3_core.forecast_groups):
  * a covariate column that is not numeric (strings such as StateHoliday) becomes the index of its value in the sorted list of
    the column's distinct values in that window (0, 1, 2, ...; missing -> NaN). The lane is then normalised like any other, so
    the model sees an ordinal code that follows the alphabet, not the effect.
  * lanes: known covariates first, then past, each in the task's column order; with one target at most 7 covariates are kept
    (cap - 1), with several targets cap // 2 = 4. fev-bench (tasks.yaml, fetched 2026-10-29): 42 of 100 tasks have dynamic
    covariates, 17 of them more than the cap (rohlik x4, walmart, solar_with_weather x2, kdd_cup_2022 x3, m5 x3: one target;
    uk_covid_nation x4: three targets), so the order decides which lanes are dropped there.

Variants (every one also restricts fev to the tasks with dynamic covariates: the others take the univariate path and cannot
change):
  target_enc  a categorical covariate (non-numeric, or integer-valued with 3..MAX_CATS distinct values in the window) becomes,
              per item, the mean of the target's context over the context steps with that category; categories not seen in the
              context get the context mean. Uses the context only (no future target value is read). Two-valued integer columns
              are left alone: a two-value target encoding is an affine map of the 0 / 1 code (up to sign) and the lane
              normalisation removes it.
  order_corr  per item, the known covariates and the past covariates are each sorted by |Pearson correlation| with the first
              target over the context (highest first; constant or all-NaN lanes last, ties keep the task order). The
              known-before-past rule stays (owner 2026-10-29), so only which lanes survive the cap (and their positions) change.
  both        target_enc, then order_corr on the encoded lanes.
"""
from __future__ import annotations

from contextlib import contextmanager

import numpy as np

from tideforge.eval import thirdparty

VARIANTS = ("target_enc", "order_corr", "both")
CODE = {"target_enc": "tenc", "order_corr": "ocorr", "both": "tencocorr"}
MAX_CATS = 16


def is_categorical(raw) -> bool:
    """raw: the column's per-item arrays as fev gives them (before encode_column)."""
    arrs = [np.asarray(v) for v in raw]
    first = next((a for a in arrs if a.size), None)
    if first is None:
        return False
    if first.dtype.kind not in "fiub":
        return True
    vals = np.concatenate([a.astype(np.float64).ravel() for a in arrs])
    vals = vals[np.isfinite(vals)]
    if not vals.size or not np.all(vals == np.round(vals)):
        return False
    return 3 <= len(np.unique(vals)) <= MAX_CATS


def target_encode(codes_ctx, codes_fut, y_ctx):
    """codes (float, NaN = missing) of the context and the future (None for a past covariate) -> encoded arrays."""
    c, y = np.asarray(codes_ctx, np.float64), np.asarray(y_ctx, np.float64)
    n = min(len(c), len(y))
    cc, yy = c[len(c) - n:], y[len(y) - n:]
    ok = np.isfinite(cc) & np.isfinite(yy)
    base = float(yy[ok].mean()) if ok.any() else 0.0
    table = {k: float(yy[ok & (cc == k)].mean()) for k in np.unique(cc[ok])}

    def enc(a):
        a = np.asarray(a, np.float64)
        return np.array([table.get(v, base) if np.isfinite(v) else np.nan for v in a], np.float32)
    return enc(c), (None if codes_fut is None else enc(codes_fut))


def corr_key(lane, y):
    a, b = np.asarray(lane, np.float64), np.asarray(y, np.float64)
    n = min(len(a), len(b))
    a, b = a[len(a) - n:], b[len(b) - n:]
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or a[ok].std() == 0 or b[ok].std() == 0:
        return -1.0
    return float(abs(np.corrcoef(a[ok], b[ok])[0, 1]))


def order_lanes(group: dict) -> dict:
    y = group["targets"][0]
    kn = sorted(range(len(group["known"])), key=lambda i: (-corr_key(group["known"][i][0], y), i))
    ps = sorted(range(len(group["past"])), key=lambda i: (-corr_key(group["past"][i], y), i))
    return dict(group, known=[group["known"][i] for i in kn], past=[group["past"][i] for i in ps])


def make_window_groups(encode: bool, order: bool):
    """score_groups.window_groups (L63-L76) with the variant's changes after the baseline groups are built."""
    G = thirdparty.score_groups()
    base = G.window_groups

    def window_groups(window, max_ctx, season, fill):
        groups = base(window, max_ctx, season, fill)
        if encode:
            past, future = window.get_input_data()
            cat_p = [is_categorical(list(past[c])) for c in window.past_dynamic_columns]
            cat_k = [is_categorical(list(past[c]) + list(future[c])) for c in window.known_dynamic_columns]
            for g in groups:
                y = g["targets"][0]
                g["past"] = [target_encode(p, None, y)[0] if cat else p for p, cat in zip(g["past"], cat_p)]
                g["known"] = [target_encode(cx, fu, y) if cat else (cx, fu) for (cx, fu), cat in zip(g["known"], cat_k)]
        if order:
            groups = [order_lanes(g) for g in groups]
        return groups
    return window_groups


def make_cov_tasks_only():
    """score_groups.cmd_score imports score_twins.load_fev_tasks at call time, so the swap is on score_twins."""
    S = thirdparty.score_twins()
    base = S.load_fev_tasks

    def load_fev_tasks(paths):
        return [t for t in base(paths) if (getattr(t, "known_dynamic_columns", None) or getattr(t, "past_dynamic_columns", None))]
    return load_fev_tasks


def swaps(name: str) -> dict:
    G, S = thirdparty.score_groups(), thirdparty.score_twins()
    if name not in VARIANTS:
        raise KeyError(f"unknown covariate variant {name!r}; valid: {VARIANTS}")
    enc, order = name in ("target_enc", "both"), name in ("order_corr", "both")
    return {(G, "window_groups"): make_window_groups(enc, order), (S, "load_fev_tasks"): make_cov_tasks_only()}


@contextmanager
def use_variant(name: str):
    sw = swaps(name)
    saved = {k: getattr(*k) for k in sw}
    try:
        for (mod, attr), fn in sw.items():
            setattr(mod, attr, fn)
        yield name
    finally:
        for (mod, attr), fn in saved.items():
            setattr(mod, attr, fn)


# ------------------------------------------------------------------------------------------------ report (no model)
def report_rows(tasks) -> list[dict]:
    """Per covariate task: covariates (known / past), how many are categorical and their largest category count, how many
    lanes the cap drops (first window only: the columns are the task's)."""
    rows = []
    for t in tasks:
        kn, ps = list(t.known_dynamic_columns or []), list(t.past_dynamic_columns or [])
        if not (kn or ps):
            continue
        w = next(iter(t.iter_windows()))
        past, future = w.get_input_data()
        cats = {}
        for c in kn + ps:
            raw = list(past[c]) + (list(future[c]) if c in kn else [])
            if is_categorical(raw):
                vals = {str(x) for a in raw for x in np.asarray(a).ravel().tolist()}
                cats[c] = len(vals)
        nt = len(w.target_columns)
        cap = 7 if nt == 1 else 4
        rows.append(dict(task=t.task_name if hasattr(t, "task_name") else str(t), targets=nt, known=len(kn), past=len(ps),
                         dropped=max(0, len(kn) + len(ps) - cap), categorical=len(cats),
                         max_categories=max(cats.values()) if cats else 0, columns=";".join(f"{k}:{v}" for k, v in cats.items())))
    return rows


def main(argv=None) -> int:
    """python -m tideforge.eval.cov_variants report --root R [--out R/eval/cov_categoricals.csv]   (CPU, reads fev's data)"""
    import argparse

    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("report",))
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    S = thirdparty.score_twins()
    S.setup_env(a.root, offline=True)
    paths = S.Paths(a.root)
    df = pd.DataFrame(report_rows(S.load_fev_tasks(paths)))
    out = a.out or f"{a.root}/eval/cov_categoricals.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"{len(df)} covariate tasks; {int((df.categorical > 0).sum())} with categorical covariates; "
          f"{int((df.dropped > 0).sum())} drop lanes at the cap; written {out}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
