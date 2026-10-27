"""Hook points of generator v0 (H1-H8) and the tables switches fill. Hand-written; the ported modules call it from lines that
tools/port_gen_v0.py inserts (marked `# [hook Hn]`). With every table empty each call returns its input unchanged and draws
no random numbers, so v0 stays bit-exact with ft3_core (tests/test_gen_parity.py runs through these calls).

    H1  layer(name, fn)       every layer call in generate_batch: an override sees the layer's inputs and the original output
    H2  "after_families"      generate_batch, after the families (and the bad-row guard), before the layers
    H3  "after_structure"     after seasonality / trend / calendar / breaks, before noise
    H4  "final"               after _finalize, before the row shuffle
    H5  group_kind(...)       _group_pass: a group kind added after the five v0 kinds
    H6  group_size(...)       _multivariatize: after the group sizes g are drawn (incl. the small-group option)
    H7  "after_groups"        end of _multivariatize (not reached when V < 2 or group_prob <= 0)
    H8  "final_blocks"        generate_blocks, the very last step: the only point where NaN may be written

Random numbers: a hook never draws from torch's global stream. ctx.rng(name) is the switch's own torch.Generator on the batch's
device, seeded from (base seed, stream, switch name, site, call counter of that site); the sites are "batch" (generate_batch:
H1-H4) and "blocks" (generate_blocks: H5-H8), and the counters advance on every call whether or not hooks are installed. So a
post-processing switch (H2-H4 and H7-H8 hooks, layer overrides) leaves every global draw of v0 unchanged; see switches.py for
the exceptions (new families, new group kinds, and value-dependent draws on CPU).

Generation is single-threaded (module state, as in ft3_core); configure() in every DataLoader worker / DDP rank.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Callable

import torch

LAYERS = ("seasonality", "trend", "calendar", "breaks", "noise", "support", "corruption", "scale", "finalize")
BATCH_POINTS = ("after_families", "after_structure", "final")          # H2 H3 H4
BLOCK_POINTS = ("after_groups", "final_blocks")                        # H7 H8
POINTS = BATCH_POINTS + BLOCK_POINTS
NAN_POINTS = ("final_blocks",)
# default priorities: hooks at the same point run by (priority, switch name)
PRIORITY = {"structure": 100, "observation": 200, "gaps": 300}

HOOKS: dict[str, list[tuple[int, str, Callable]]] = {p: [] for p in POINTS}
LAYER_OVERRIDES: dict[str, tuple[str, Callable]] = {}
GROUP_KINDS_EXT: dict[str, tuple[str, Callable]] = {}
GROUP_SIZE: list[tuple[str, Callable]] = []        # at most one
STRICT = False                                     # True: fail if a hook touches the global RNG (tests, gallery, audit)

_seed = 0
_stream = 0
_count = {"batch": 0, "blocks": 0}
_gens: dict[tuple[str, str], torch.Generator] = {}


def reset(seed: int = 0, stream: int = 0) -> None:
    """Empty every table and restart the call counters (configure() does this)."""
    global _seed, _stream
    for p in POINTS:
        HOOKS[p].clear()
    LAYER_OVERRIDES.clear()
    GROUP_KINDS_EXT.clear()
    GROUP_SIZE.clear()
    _seed, _stream = int(seed), int(stream)
    _count.update(batch=0, blocks=0)
    _gens.clear()


def installed() -> bool:
    return bool(any(HOOKS.values()) or LAYER_OVERRIDES or GROUP_KINDS_EXT or GROUP_SIZE)


def tick(site: str) -> None:
    """Start a new call of `site` (no random draw)."""
    _count[site] += 1
    for k in [k for k in _gens if k[0] == site]:
        del _gens[k]


def switch_seed(name: str, site: str, count: int, seed: int | None = None, stream: int | None = None) -> int:
    s = f"{_seed if seed is None else seed}|{_stream if stream is None else stream}|{name}|{site}|{count}"
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little") & (2 ** 63 - 1)


@dataclass
class Ctx:
    """What a hook sees. The flags (generate_batch sites) may be replaced by the hook; generate_batch carries them on."""
    site: str
    device: torch.device
    L: int
    profile: dict | None = None
    is_count: torch.Tensor | None = None
    is_nonneg: torch.Tensor | None = None
    allow_add: torch.Tensor | None = None
    passthrough: torch.Tensor | None = None
    switch: str = ""
    extra: dict = field(default_factory=dict)

    def rng(self, name: str | None = None) -> torch.Generator:
        """The switch's own generator for this call (default: the switch whose hook is running)."""
        name = name or self.switch
        key = (self.site, name)
        g = _gens.get(key)
        if g is None:
            g = torch.Generator(device=self.device)
            g.manual_seed(switch_seed(name, self.site, _count[self.site]))
            _gens[key] = g
        return g


def _rng_state(device):
    st = [torch.get_rng_state()]
    if torch.device(device).type == "cuda":
        st.append(torch.cuda.get_rng_state(device))
    return st


def _call(name: str, where: str, device, fn: Callable, *args):
    if not STRICT:
        return fn(*args)
    before = _rng_state(device)
    out = fn(*args)
    if not all(torch.equal(a, b) for a, b in zip(before, _rng_state(device))):
        raise RuntimeError(f"switch {name!r} drew from torch's global RNG at {where}: use ctx.rng()")
    return out


def _finite(x: torch.Tensor, name: str, where: str) -> None:
    if not bool(torch.isfinite(x).all()):
        raise RuntimeError(f"switch {name!r} wrote NaN / inf at {where}: missing values may only be written at final_blocks (H8)")


# ---------------------------------------------------------------------------------------------------- generate_batch (H1-H4)
def layer(name: str, fn: Callable) -> Callable:
    """H1: the layer itself, or a wrapper that runs it (so its global draws happen exactly as in v0) and then the override."""
    ov = LAYER_OVERRIDES.get(name)
    if ov is None:
        return fn
    sw, ofn = ov

    def run(*args, **kwargs):
        out = fn(*args, **kwargs)
        x = args[0]
        ctx = Ctx("batch", x.device, x.shape[-1], switch=sw)
        new = _call(sw, f"layer {name}", x.device, ofn, args, kwargs, out, ctx)
        _finite(new[0] if isinstance(new, tuple) else new, sw, f"layer {name}")
        return new
    return run


def run_batch(point, x, is_count, is_nonneg, allow_add, passthrough, device, pr, L):
    """H2-H4: returns (x, is_count, is_nonneg, allow_add), unchanged when nothing is installed at `point`."""
    hooks = HOOKS[point]
    if not hooks:
        return x, is_count, is_nonneg, allow_add
    ctx = Ctx("batch", torch.device(device), L, pr, is_count, is_nonneg, allow_add, passthrough)
    for _, sw, fn in hooks:
        ctx.switch = sw
        x = _call(sw, point, device, fn, x, ctx)
        _finite(x, sw, point)
    return x, ctx.is_count, ctx.is_nonneg, ctx.allow_add


# ---------------------------------------------------------------------------------------------------- generate_blocks (H5-H8)
def group_kind(name, z, act, mu, sd, device):
    """H5: an added group kind; fn(z, act, mu, sd, ctx) -> (y (n, V, L) in the data scale, role (n, V))."""
    sw, fn = GROUP_KINDS_EXT[name]
    ctx = Ctx("blocks", torch.device(device), z.shape[-1], switch=sw)
    y, r = _call(sw, f"group kind {name}", device, fn, z, act, mu, sd, ctx)
    _finite(y, sw, f"group kind {name}")
    return y, r


def group_size(g, kind, is_group, V, device):
    """H6: fn(g, kind, is_group, V, ctx) -> g, every size in [2, V]."""
    if not GROUP_SIZE:
        return g
    sw, fn = GROUP_SIZE[0]
    ctx = Ctx("blocks", torch.device(device), 0, switch=sw)
    g2 = _call(sw, "group_size", device, fn, g, kind, is_group, V, ctx)
    if g2.shape != g.shape or bool((g2 < 2).any() or (g2 > V).any()):
        raise RuntimeError(f"switch {sw!r}: group sizes must keep their shape and lie in [2, {V}]")
    return g2


def run_groups(point, x, role, lgid, device):
    """H7 / H8: fn(x, role, lgid, ctx) -> (x, role, lgid); unchanged when nothing is installed at `point`."""
    hooks = HOOKS[point]
    if not hooks:
        return x, role, lgid
    ctx = Ctx("blocks", torch.device(device), x.shape[-1])
    for _, sw, fn in hooks:
        ctx.switch = sw
        x, role, lgid = _call(sw, point, device, fn, x, role, lgid, ctx)
        if point not in NAN_POINTS:
            _finite(x, sw, point)
    return x, role, lgid


def final_blocks(out, device):
    """H8 around generate_blocks' return value."""
    return run_groups("final_blocks", *out, device) if HOOKS["final_blocks"] else out
