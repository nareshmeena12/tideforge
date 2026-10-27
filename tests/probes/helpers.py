"""Shared test helpers: build a probe at small n and score the oracle and the naive model on it."""
import hashlib

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
    a = hashlib.sha256(common.set_bytes(mod.build(n, mod.SEED))).hexdigest()
    b = hashlib.sha256(common.set_bytes(mod.build(n, mod.SEED))).hexdigest()
    return a == b and common.fingerprint(mod.build(n, mod.SEED)) == common.fingerprint(mod.build(n, mod.SEED))
