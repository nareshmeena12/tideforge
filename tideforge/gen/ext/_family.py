"""Install a new family by row share (G5, G7, G9, G10), and the per-family generator.

Switch params `share` (fraction of rows) and `mode`:
  "add"                      weight w = share T / (1 - share) on top of the profile's total T (the others renormalise)
  "replace:<a>[+<b>...]"     weight share T taken from <a>, then <b>, ... (EXPERIMENTS.md rule 6, read 2026-10-28 as "the
                             closest families with enough weight"); if the targets hold less than share T, the family takes
                             all of them (the replace arm is then smaller than the add arm: noted in its config)
Plan mode (config hash, no profile): the share itself is recorded as the weight.
"""
from __future__ import annotations

import torch

from tideforge.gen import profiles as _profiles


def own_generator(device) -> torch.Generator:
    """Families draw from the global stream: one seed from it, then the family's own generator (as long_cycle_cal)."""
    dev = torch.device(device)
    g = torch.Generator(device=dev)
    g.manual_seed(int(torch.randint(0, 2 ** 62, (1,))))
    return g


def weight_for(profile: str, share: float, mode: str) -> float:
    w = _profiles.GEN_PROFILES[profile]["weights"]
    tot = sum(v for v in w.values() if v > 0)
    if mode == "add":
        return share * tot / (1.0 - share)
    have = sum(w.get(o, 0.0) for o in mode.split(":", 1)[1].split("+"))
    if have <= 0:
        raise ValueError(f"{mode}: the profile has no weight in these families")
    return min(share * tot, have)


def install(inst, family: str, fn, p: dict, **flags) -> None:
    share, mode = float(p["share"]), str(p["mode"])
    if not (0.0 < share < 1.0):
        raise ValueError(f"{family}: share must be in (0, 1)")
    weight = share if inst.profile is None else weight_for(inst.profile, share, mode)
    inst.add_family(family, fn, weight=weight, mode=mode, **flags)
