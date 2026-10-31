"""Shared test helpers: build a probe at small n and score the oracle and the naive model on it."""

from tideforge import hashing as H
from tideforge.probes import common, metrics as M
from tideforge.probes.models import Naive, Oracle
from tideforge.probes.run import evaluate


def scores(mod, n=48):
    """{(model, cell, metric, mode, view): value} for the oracle and the naive model."""
    ps = mod.build(n, mod.SEED)
    out = {}
    for model in (Oracle(), Naive()):
        res, _ = evaluate(model, ps, mod)
        for cell, m in res:
            if isinstance(m, tuple):
                _, a, b = m
                out[(model.name, cell.name, a.name + "_ratio_vs_control", a.mode, a.view)] = M.unpaired(a, b)[0]
            else:
                out[(model.name, cell.name, m.name, m.mode, m.view)] = m.agg(m.comps)
    return ps, out


def same_sha_twice(mod, n=8):
    a = H.content_sha(common.flat_arrays(mod.build(n, mod.SEED)))       # what load() checks
    b = H.content_sha(common.flat_arrays(mod.build(n, mod.SEED)))
    return a == b and common.fingerprint(mod.build(n, mod.SEED)) == common.fingerprint(mod.build(n, mod.SEED))
