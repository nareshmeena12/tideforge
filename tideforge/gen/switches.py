"""Registry of tideforge generator switches (EXPERIMENTS.md rule 2: every change is a switch, off by default).

A switch is off unless it is listed in GenConfig.switches. When on, its `install(params, inst)` registers what it changes through
the Installer: new families, layer overrides (H1), hooks at H2-H4 / H7-H8, new group kinds (H5), group-size rules (H6, one per kind). The
generated v0 modules are never edited for a switch; the code lives in tideforge/gen/ext/. configure() first restores v0 (tables,
profiles, hooks), then installs the active switches in name order, so configure(b0()) after any switch config is v0 again.

Random numbers (see _hooks.py): hooks and layer overrides draw only from ctx.rng(), the switch's own generator, so a
post-processing switch leaves every global draw of v0 unchanged: the on output is the off output plus the change. Exceptions:
  * a new family (add_family) changes the rows per family and the global draws of every family after it: the whole batch
    changes, not only the new rows. Its function fn(n, L, device) draws from the global stream like the v0 families.
  * a new group kind (add_group_kind) changes the kind draw (one more category), so every block's kind can change.
  * value-dependent draws: on CPU torch.poisson consumes a value-dependent number of draws, so a hook or override that changes
    values BEFORE _L_noise / _L_support (H2, H3, overrides of seasonality ... breaks) can shift later global draws on CPU.
    H4, H7, H8 and overrides of noise and later layers are exact on every device.
    Measured on CPU 2026-10-27 (STRESS config + breaks, 512 x 1024, test_post p = 0.5): H3 (after_structure) left the global
    RNG state equal to v0's; H2 (after_families) did not (the noise layer's poisson draws changed).
    Measured on CUDA 2026-10-27 (Perlmutter A100, torch 2.13.0+cu130, torch's default mode):
    test_gpu_h2_h3_keep_the_other_draws passed for both H2 and H3: the CPU and CUDA RNG states after the call equal v0's and
    at least 40% of the rows are bit-identical (p = 0.5), so on CUDA the value-changing hooks before the noise layer leave
    the other global draws unchanged. (GPU tests now run in deterministic mode; the next GPU run repeats it there.)
Hooks at the same point run by (priority, switch name); the defaults are _hooks.PRIORITY: structure 100 (breaks) <
observation 200 (rounding, censoring) < gaps 300 (always last). The install plan (with priorities) goes into the config hash.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping

from tideforge.gen import _hooks
from tideforge.gen import groups as _groups
from tideforge.gen import profiles as _profiles

NAME = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Switch:
    name: str                                   # [a-z][a-z0-9_]*, used in run ids
    experiment: str                             # e.g. "G5.2"; "TEST" for the test switches
    doc: str
    install: Callable[[dict, "Installer"], None]
    params: Mapping[str, object] = field(default_factory=dict)      # defaults; a config may override any of them


REGISTRY: dict[str, Switch] = {}


def register(sw: Switch) -> Switch:
    if not NAME.match(sw.name):
        raise ValueError(f"switch name {sw.name!r} must match {NAME.pattern}")
    if sw.name in REGISTRY:
        raise KeyError(f"switch {sw.name!r} is already registered")
    REGISTRY[sw.name] = Switch(sw.name, sw.experiment, sw.doc, sw.install, MappingProxyType(dict(sw.params)))
    return REGISTRY[sw.name]


def unregister(name: str) -> None:
    REGISTRY.pop(name, None)


def _freeze(v):
    if isinstance(v, (list, tuple)):
        return tuple(_freeze(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _freeze(x)) for k, x in v.items()))
    return v


def normalize(switches) -> tuple[tuple[str, tuple], ...]:
    """{name: True | {param: value}} or ((name, params), ...) -> sorted ((name, ((param, value), ...)), ...) with every
    parameter resolved (defaults filled in). False / None means off and is dropped. Unknown switches or parameters raise."""
    items = switches.items() if isinstance(switches, Mapping) else switches
    out = {}
    for name, val in items:
        if val is False or val is None:
            continue
        if name not in REGISTRY:
            raise KeyError(f"unknown generator switch {name!r}; registered: {sorted(REGISTRY)}")
        sw = REGISTRY[name]
        given = {} if val is True else dict(val)
        bad = sorted(set(given) - set(sw.params))
        if bad:
            raise KeyError(f"switch {name!r} has no parameter(s) {bad}; parameters: {sorted(sw.params)}")
        out[name] = tuple(sorted((k, _freeze(given.get(k, d))) for k, d in sw.params.items()))
    return tuple(sorted(out.items()))


def validate(switches) -> None:
    normalize(switches)


# ---------------------------------------------------------------------------------------------------- v0 snapshot / restore
_V0_FAMILIES = dict(_profiles.FAMILIES)
_V0_PASSTHROUGH = frozenset(_profiles.PASSTHROUGH_FAMILIES)
_V0_PROFILES = dict(_profiles.GEN_PROFILES)
_V0_GROUP_KINDS = tuple(_groups.GROUP_KINDS)


def restore_v0() -> None:
    """Put back the v0 tables (in place: batch.py imported the same dict / set objects) and empty every hook table."""
    for k in [k for k in _profiles.FAMILIES if k not in _V0_FAMILIES]:
        del _profiles.FAMILIES[k]
    assert all(_profiles.FAMILIES[k] is v for k, v in _V0_FAMILIES.items()), "a v0 family entry was replaced"
    _profiles.PASSTHROUGH_FAMILIES.clear()
    _profiles.PASSTHROUGH_FAMILIES.update(_V0_PASSTHROUGH)
    for k in [k for k in _profiles.GEN_PROFILES if k not in _V0_PROFILES]:
        del _profiles.GEN_PROFILES[k]
    _profiles.GEN_PROFILES.update(_V0_PROFILES)          # clean2 is then rebuilt by resolve_profile for the config
    _groups.GROUP_KINDS = _V0_GROUP_KINDS


# ---------------------------------------------------------------------------------------------------- installer
class Installer:
    """What a switch's install() may change. apply=False only records the plan (for the config hash)."""

    def __init__(self, name: str, profile: str | None, apply: bool, plan: list):
        self.name, self.profile, self.apply, self.plan = name, profile, apply, plan

    def add_family(self, family: str, fn: Callable, *, weight: float, is_count: bool = False, is_nonneg: bool = False,
                   allow_add: bool = True, passthrough: bool = False, mode: str = "add") -> None:
        """mode "add": the family gets `weight` on top of the profile's weights (_allocate renormalises the others).
        mode "replace:<family>": the new family takes exactly `weight` from <family> (error if <family> has less); it is
        placed right after <family>, so the other families' row counts stay as they were.
        mode "replace:<a>+<b>+...": `weight` is taken from the listed families in that order (each emptied before the next is
        touched); placed right after <a>."""
        if not (weight > 0):
            raise ValueError(f"{self.name}: weight must be > 0")
        if mode != "add" and not mode.startswith("replace:"):
            raise ValueError(f"{self.name}: mode must be 'add' or 'replace:<family>', not {mode!r}")
        self.plan.append(("family", self.name, family, mode, float(weight), bool(is_count), bool(is_nonneg), bool(allow_add),
                          bool(passthrough), _qual(fn)))
        if not self.apply:
            return
        if family in _profiles.FAMILIES:
            raise KeyError(f"{self.name}: family {family!r} exists already")
        pr = _profiles.GEN_PROFILES[self.profile]
        w = dict(pr["weights"])
        if mode == "add":
            w[family] = float(weight)
        else:
            olds = mode.split(":", 1)[1].split("+")
            have = sum(w.get(o, 0.0) for o in olds)
            if have + 1e-12 < weight:
                raise ValueError(f"{self.name}: {mode} needs weight {weight}, but {'+'.join(olds)} has {have} in profile "
                                 f"{self.profile!r}")
            need = float(weight)
            for o in olds:
                take = min(need, w.get(o, 0.0))
                rest = w.get(o, 0.0) - take
                w[o] = 0.0 if abs(rest) < 1e-12 else rest
                need -= take
            w = {k2: v2 for k, v in w.items()
                 for k2, v2 in (((k, v), (family, float(weight))) if k == olds[0] else ((k, v),))}
        _profiles.FAMILIES[family] = (fn, bool(is_count), bool(is_nonneg), bool(allow_add), 0.0)
        if passthrough:
            _profiles.PASSTHROUGH_FAMILIES.add(family)
        _profiles.GEN_PROFILES[self.profile] = dict(pr, weights=w)

    def override_layer(self, layer: str, fn: Callable) -> None:
        """fn(args, kwargs, out, ctx) -> new out (same structure); runs after the v0 layer, which still makes its draws."""
        if layer not in _hooks.LAYERS:
            raise KeyError(f"{self.name}: no layer {layer!r}; layers: {_hooks.LAYERS}")
        self.plan.append(("layer", self.name, layer, _qual(fn)))
        if self.apply:
            if layer in _hooks.LAYER_OVERRIDES:
                raise ValueError(f"{self.name}: layer {layer!r} is already overridden by {_hooks.LAYER_OVERRIDES[layer][0]!r}")
            _hooks.LAYER_OVERRIDES[layer] = (self.name, fn)

    def add_hook(self, point: str, fn: Callable, priority: int | str = "observation") -> None:
        """Batch points: fn(x, ctx) -> x (ctx.is_count / is_nonneg / allow_add may be replaced). Block points:
        fn(x, role, lgid, ctx) -> (x, role, lgid). Only "final_blocks" may write NaN."""
        if point not in _hooks.POINTS:
            raise KeyError(f"{self.name}: no hook point {point!r}; points: {_hooks.POINTS}")
        prio = _hooks.PRIORITY[priority] if isinstance(priority, str) else int(priority)
        self.plan.append(("hook", point, prio, self.name, _qual(fn)))
        if self.apply:
            _hooks.HOOKS[point].append((prio, self.name, fn))
            _hooks.HOOKS[point].sort(key=lambda h: (h[0], h[1]))       # stable: one switch's hooks keep their order

    def add_group_kind(self, kind: str, fn: Callable, prob: float) -> None:
        """fn(z, act, mu, sd, ctx) -> (y, role); `prob` is appended to GROUP_KIND_PROBS (relative to the v0 kinds' sum)."""
        if not (prob > 0):
            raise ValueError(f"{self.name}: prob must be > 0")
        self.plan.append(("group_kind", self.name, kind, float(prob), _qual(fn)))
        if self.apply:
            if kind in _groups.GROUP_KINDS:
                raise KeyError(f"{self.name}: group kind {kind!r} exists already")
            _groups.GROUP_KINDS = tuple(_groups.GROUP_KINDS) + (kind,)
            _groups.GROUP_KIND_PROBS = tuple(_groups.GROUP_KIND_PROBS) + (float(prob),)
            _hooks.GROUP_KINDS_EXT[kind] = (self.name, fn)

    def set_profile(self, key: str, fn: Callable) -> None:
        """Replace one setting of the active profile: profile[key] = fn(current value). Settings the v0 layers read from the
        profile (breaks_p, miss_p, weights, ...); configure() rebuilds the profile from v0 first, so nothing carries over."""
        self.plan.append(("profile", self.name, key, _qual(fn)))
        if self.apply:
            pr = _profiles.GEN_PROFILES[self.profile]
            if key not in pr:
                raise KeyError(f"{self.name}: profile {self.profile!r} has no setting {key!r}")
            _profiles.GEN_PROFILES[self.profile] = dict(pr, **{key: fn(pr[key])})

    def set_group_size(self, fn: Callable, kinds) -> None:
        """fn(g, kind, is_group, V, ctx) -> g (first group of a block; the optional second group keeps v0's sizes). The rule
        applies only to the group kinds it claims (names); a kind claimed by two rules is an error at install time."""
        kinds = tuple(sorted(kinds))
        if not kinds:
            raise ValueError(f"{self.name}: a group-size rule must claim at least one group kind")
        self.plan.append(("group_size", self.name, kinds, _qual(fn)))
        if self.apply:
            for sw, ks, _ in _hooks.GROUP_SIZE:
                both = sorted(set(ks) & set(kinds))
                if both:
                    raise ValueError(f"{self.name}: group kind(s) {both} already have a size rule from {sw!r}")
            _hooks.GROUP_SIZE.append((self.name, kinds, fn))


def _qual(fn) -> str:
    return f"{getattr(fn, '__module__', '?')}.{getattr(fn, '__qualname__', repr(fn))}"


def plan(switches) -> list:
    """The install plan of a switch set (nothing is changed); part of GenConfig.config_hash."""
    out: list = []
    for name, params in normalize(switches):
        REGISTRY[name].install(dict(params), Installer(name, None, False, out))
    return out


def install(switches, profile: str) -> list:
    """Install the active switches for `profile` (configure() calls this after restore_v0 and resolve_profile)."""
    out: list = []
    for name, params in normalize(switches):
        REGISTRY[name].install(dict(params), Installer(name, profile, True, out))
    return out


from tideforge.gen import ext  # noqa: E402,F401  (registers the switches in tideforge/gen/ext/)
