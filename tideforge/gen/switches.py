"""Registry of tideforge generator switches (EXPERIMENTS.md rule 2: every change is a switch, off by default).

Generator v0 has none. A switch is added here with its off value; code behind it must draw no random numbers when it is off,
so v0 parity (tests/test_gen_parity.py) keeps holding with every switch off.
"""
from __future__ import annotations

# name -> (off value, one-line description, experiment id)
REGISTRY: dict[str, tuple[object, str, str]] = {}


def validate(switches: dict) -> None:
    unknown = sorted(set(switches) - set(REGISTRY))
    if unknown:
        raise KeyError(f"unknown generator switch(es) {unknown}; registered: {sorted(REGISTRY)}")
