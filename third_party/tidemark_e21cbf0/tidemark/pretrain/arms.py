"""Stage A arms: ONE per-layer pattern string per arm (the existing four arms and the new five come from
the same code path, models are built layer by layer from `layer_kinds`).

A pattern is a comma-separated list with one kind per layer, input side first:
    softmax | gla | gdn | swa<window>      e.g. "gla,gla,gla,gla,gla,softmax"
A single kind ("gla") means every layer. The registry below gives each arm's pattern; a TwinConfig can also
carry its own `pattern` string (any label as arm), which wins over the registry.

Arm ids are lowercase letters and digits only (the scoring harnesses parse run tags as <arm>_s<seed>, so an
underscore inside an arm id would be read as the seed). The names used in the Stage A brief are accepted as
aliases (gla_s1 -> glas1, ...). "s1/s2/s3" = the number of full-softmax layers in the GLA stack.

Layers (1-based):   glas1: 6 = softmax          glas2: 3, 6 = softmax          glas3: 2, 4, 6 = softmax
                    swa16s1: SWA16 in 1-5, softmax in 6          gdn: Gated DeltaNet in all 6
"""
from __future__ import annotations

import math
import re

M4_ARMS = ("softmax", "gla", "swa8", "swa16")
STAGE_A_ARMS = ("gdn", "glas1", "glas2", "glas3", "swa16s1")
ALL_ARMS = M4_ARMS + STAGE_A_ARMS
ALIASES = {"gla_s1": "glas1", "gla_s2": "glas2", "gla_s3": "glas3", "swa16_s1": "swa16s1"}

ARM_PATTERNS = {
    "softmax": "softmax", "gla": "gla", "swa8": "swa8", "swa16": "swa16", "gdn": "gdn",
    "glas1": "gla,gla,gla,gla,gla,softmax",
    "glas2": "gla,gla,softmax,gla,gla,softmax",
    "glas3": "gla,softmax,gla,softmax,gla,softmax",
    "swa16s1": "swa16,swa16,swa16,swa16,swa16,softmax",
}
# Stage A arms have no pilot of their own (owner: same recipe). `--lr-from pilot` takes the rate the pilot chose
# for the arm named here; the existing four arms use their own.
# Stage B (other sizes): gdn has its own pilot, the mixes take the rate of their main ingredient. swa16s1 takes
# softmax's, because Stage B pilots only softmax, gla and gdn.
STAGE_B_DONOR = {"gdn": "gdn", "glas1": "gla", "glas2": "gla", "glas3": "gla", "swa16s1": "softmax"}
LR_DONOR = {"gdn": "gla", "glas1": "gla", "glas2": "gla", "glas3": "gla", "swa16s1": "swa16"}
_KIND = re.compile(r"^(softmax|gla|gdn|swa(\d+))$")


def canonical(arm: str) -> str:
    """The arm id (aliases of the brief resolved)."""
    return ALIASES.get(arm, arm)


def parse_kind(token: str) -> tuple[str, int | None]:
    """'swa16' -> ('swa', 16); 'gla' -> ('gla', None)."""
    m = _KIND.match(token)
    if not m:
        raise ValueError(f"unknown layer kind {token!r}; kinds: softmax, gla, gdn, swa<window>")
    return ("swa", int(m.group(2))) if m.group(2) else (token, None)


def _half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def scale_pattern(toks: tuple[str, ...], n: int) -> tuple[str, ...]:
    """Map a 6-layer hybrid pattern (one softmax kind among one other kind) to n layers, keeping the SOFTMAX
    FRACTION and the relative depths (owner, Stage B).

    Source layer i (0-based of 6) owns the depth block [i/6, (i+1)/6). With c softmax layers in the source:
      k = max(1, floor(c * n / 6 + 0.5))        softmax layers wanted (round half up; never zero)
      k >= c : source softmax layer j gets g_j layers, k // c each and the k % c extra ones to the LATER source
               layers; they are the LAST g_j layers of its depth block (the block is [round(i n/6), round((i+1) n/6)))
      k <  c : the source layers kept are the last one and evenly spaced earlier ones,
               j_t = c-1-round((k-1-t)(c-1)/(k-1)), one layer each at floor(centre * n), centre = (i+1/2)/6; a
               collision moves the earlier one down by one layer
    For any other depth (7-11, 13, ...) the blocks can be too small for k and the count is clipped to what fits.
    The other layers keep the non-softmax kind. 6 layers: unchanged. Examples (the pattern at 4 / 12 layers):
      glas1  4: g g g S           12: g x10 S S        (1 of 6 = 2 of 12; 1 of 4 = 25% instead of 17%)
      glas2  4: g g g S           12: g g g g S S g g g g S S      (2 of 6 -> 1 of 4: 1.33 rounds down, so glas2 == glas1 at 4)
      glas3  4: g S g S           12: g g S S g g S S g g S S
    """
    if n == len(toks):
        return toks
    soft = [i for i, t in enumerate(toks) if t == "softmax"]
    base = {t for t in toks if t != "softmax"}
    if not soft or len(base) != 1:
        raise ValueError(f"cannot rescale pattern {','.join(toks)}: it needs softmax layers and one other kind")
    other, c = next(iter(base)), len(soft)
    k = max(1, _half_up(c * n / len(toks)))
    pos: list[int] = []
    if k >= c:
        extra = k % c
        for j, i in enumerate(soft):
            g = k // c + (1 if j >= c - extra else 0)
            lo, hi = _half_up(i * n / len(toks)), _half_up((i + 1) * n / len(toks))
            lo = min(lo, n - 1)
            hi = max(hi, lo + 1)                            # a block is at least one layer (n < 6)
            pos += list(range(max(lo, hi - g), hi))
    else:
        keep = [soft[c - 1 - _half_up((k - 1 - t) * (c - 1) / (k - 1))] if k > 1 else soft[c - 1] for t in range(k)]
        for i in keep:
            p = min(n - 1, int(math.floor((i + 0.5) / len(toks) * n)))
            while p in pos:
                p -= 1
            pos.append(p)
    pos = sorted(set(pos))
    if len(pos) != k and (n <= len(toks) or n % len(toks) == 0):          # exact for n <= 6 and multiples of 6
        raise ValueError(f"pattern rescale to {n} layers gave {len(pos)} softmax layers, wanted {k}")
    return tuple("softmax" if i in pos else other for i in range(n))


def layer_kinds(arm: str, pattern: str = "", n_layers: int = 6) -> tuple[str, ...]:
    """The kind of every layer, e.g. ('gla', 'gla', 'softmax', 'gla', 'gla', 'softmax'). A registered hybrid is
    written for 6 layers; at another depth it is rescaled by `scale_pattern`. A pattern string given in the
    config must have exactly n_layers entries."""
    pat = pattern or ARM_PATTERNS.get(canonical(arm))
    if pat is None:
        raise ValueError(f"unknown arm {arm!r}; arms: {ALL_ARMS} (or give a pattern string)")
    toks = tuple(t.strip() for t in pat.split(",") if t.strip())
    if len(toks) == 1:
        toks = toks * n_layers
    elif not pattern and len(toks) == 6 and n_layers != 6:
        toks = scale_pattern(toks, n_layers)
    if len(toks) != n_layers:
        raise ValueError(f"pattern {pat!r} has {len(toks)} entries for {n_layers} layers")
    for t in toks:
        parse_kind(t)
    return toks


def lr_donor(arm: str) -> str:
    arm = canonical(arm)
    return LR_DONOR.get(arm, arm)


def uses(kinds: tuple[str, ...], kind: str) -> bool:
    return any(parse_kind(t)[0] == kind for t in kinds)


def describe(arm: str, pattern: str = "", n_layers: int = 6) -> str:
    k = layer_kinds(arm, pattern, n_layers)
    return ",".join(k)
