"""G3.1: a family's share of the generated series.

  family_share   set `family` (default long_cycle) to `share` of the rows: its profile weight becomes
                 w = s * (T - w0) / (1 - s), where T is the profile's weight total and w0 the family's weight, so after
                 _allocate's renormalisation the family holds s. Like a new family, this changes the rows per family and
                 every global draw after it (the whole batch changes, not only the family's rows).
"""
from __future__ import annotations

from tideforge.gen.switches import Switch, register


def target_weight(weights: dict, family: str, share: float) -> float:
    if not 0 < share < 1:
        raise ValueError(f"share must be in (0, 1), not {share}")
    w0 = float(weights.get(family, 0.0))
    rest = sum(weights.values()) - w0
    return share * rest / (1 - share)


def _install(p, inst):
    def weights(w):
        if p["family"] not in w:
            raise KeyError(f"family_share: {p['family']!r} is not in the profile's mix (is its gen_addition on?)")
        return dict(w, **{p["family"]: target_weight(w, p["family"], p["share"])})

    inst.set_profile("weights", weights)


register(Switch("family_share", "G3.1", "set one family's share of the generated series", _install,
                dict(family="long_cycle", share=0.15)))
