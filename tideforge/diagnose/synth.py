"""Synthetic windows for D5 / D6: tideforge.gen rows at the dev windows' context lengths, for B0 or any experiment config.

windows(gc, lengths, n, seed)            n rows in total, lengths drawn from the real ones (bucketed to powers of 2 for batching,
                                          then cut to the drawn length from the end), through generate_batch (all of v0's layers)
by_family(gc, lengths, per_family, seed)  the same, one batch per family (generate_batch's weights={family: 1}), so each row keeps
                                          its family label (generate_batch shuffles rows of a mixed batch). The global draws then
                                          differ from a mixed batch; D6 only needs each family's rows.
Rows are CPU numpy float64. STRICT is off (no hook test here).
"""
from __future__ import annotations

import numpy as np
import torch

import tideforge.gen as G


def _buckets(lengths, n, rng):
    L = np.asarray(lengths, np.int64)
    pick = rng.choice(L, size=n, replace=True)
    b = np.maximum(16, 2 ** np.ceil(np.log2(np.maximum(pick, 1))).astype(np.int64))
    return pick, b


def windows(gc, lengths, n: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    prof = G.configure(gc, seed=seed)
    torch.manual_seed(seed)
    pick, b = _buckets(lengths, n, rng)
    out = [None] * n
    for Lb in np.unique(b):
        idx = np.flatnonzero(b == Lb)
        x = G.generate_batch(len(idx), int(Lb), "cpu", profile=prof).double().numpy()
        for k, i in enumerate(idx):
            out[i] = x[k, Lb - pick[i]:]
    return out


def by_family(gc, lengths, per_family: int, seed: int = 0) -> tuple[list[np.ndarray], list[str]]:
    rng = np.random.default_rng(seed)
    prof = G.configure(gc, seed=seed)
    torch.manual_seed(seed)
    fams = [f for f, w in G.GEN_PROFILES[prof]["weights"].items() if w > 0]
    xs, labels = [], []
    for f in fams:
        pick, b = _buckets(lengths, per_family, rng)
        for Lb in np.unique(b):
            idx = np.flatnonzero(b == Lb)
            x = G.generate_batch(len(idx), int(Lb), "cpu", weights={f: 1.0}, profile=prof).double().numpy()
            for k, i in enumerate(idx):
                xs.append(x[k, Lb - pick[i]:])
                labels.append(f)
    return xs, labels
