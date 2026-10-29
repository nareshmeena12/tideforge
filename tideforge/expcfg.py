"""Experiment configs: configs/exp/<ID>.yaml, one per experiment ID. An experiment is a config, never a code change: the
generator code behind a switch lives in tideforge/gen/ext/ and is off unless a config lists it.

    id: G4.1                      # must equal the file name (configs/exp/G4.1.yaml)
    base: b0                      # b0 (ft3_main's cfg, configs/ft3_main_cfg.json) | defaults (ft3_core CONFIG; tests only)
    gen:                          # GenConfig field changes vs base; gen_additions entries are merged
      gen_additions: {breaks: true}
    switches:                     # name: true | {param: value}; parameters not given keep their defaults
      test_post: {p: 0.25}
    seeds: [0]                    # screening: 1 seed; a win is confirmed with 3 (EXPERIMENTS.md rule 4)
    notes: free text

    train:                        # batcher / training settings: stored and validated now, applied by T3 (5M recipe port)
      long_row_frac: 0.5
    model:                        # model options (tideforge.model.OPTIONS), off when absent; non-default ones join the run id
      known_attn: mask            # M1 (G1.13): off | mask
      known_lookahead: 32         # M1's band in patches (G1.13c); absent = the whole window

"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from tideforge.gen.config import GenConfig, b0

REPO = Path(__file__).resolve().parents[1]
EXP_DIR = REPO / "configs" / "exp"
KEYS = {"id", "base", "gen", "switches", "seeds", "notes", "train", "model"}
# train: keys T3 will apply (owner 2026-10-28): the batcher's long_row_frac (G3.3) and G1.2's lead noise on known lanes
TRAIN_KEYS = {"long_row_frac": float, "lead_noise": dict, "crop_breaks": dict, "max_steps": int, "pack_rows": int,
              "deterministic_gen": bool, "smoke_points": float, "target_points": float}
LEAD_NOISE_KEYS = {"p_noisy", "sigma0_lo", "sigma0_hi", "step"}
GEN_FIELDS = set(GenConfig.__dataclass_fields__) - {"version", "switches"}
# model: options (owner 2026-10-29, M1); the same names and defaults as tideforge.model.OPTIONS (a test checks it)
MODEL_KEYS = {"known_attn": ("off", "mask"), "known_lookahead": None}


@dataclass(frozen=True)
class Exp:
    id: str
    base_name: str
    base: GenConfig
    gen: GenConfig
    seeds: tuple[int, ...] = (0,)
    notes: str = ""
    train: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, compare=False)
    model: dict = field(default_factory=dict)          # non-default model options only

    def run_id(self) -> str:
        """'<ID>-<generator fragment>[-<model fragment>]', e.g. 'G4.1-v0-breaks-1a2b3c4d', 'G1.13c-v0-kamask-kl32' (the seed
        is appended by the launcher)."""
        return f"{self.id}-{self.gen.run_id_fragment(self.base)}{model_fragment(self.model)}"


def model_fragment(model: dict) -> str:
    """'-kamask-kl32' for {known_attn: mask, known_lookahead: 32}; '' when every option is off."""
    short = {"known_attn": "ka", "known_lookahead": "kl"}
    return "".join(f"-{short[k]}{model[k]}" for k in MODEL_KEYS if k in model)


def base_config(name: str) -> GenConfig:
    if name == "b0":
        return b0()
    if name == "defaults":
        return GenConfig()
    raise ValueError(f"base must be 'b0' or 'defaults', not {name!r}")


def apply_changes(base: GenConfig, gen: dict | None = None, switches: dict | None = None) -> GenConfig:
    gen = dict(gen or {})
    bad = sorted(set(gen) - GEN_FIELDS)
    if bad:
        raise KeyError(f"unknown generator setting(s) {bad}; settings: {sorted(GEN_FIELDS)}")
    for k in ("group_kind_probs",):
        if gen.get(k) is not None:
            gen[k] = tuple(float(x) for x in gen[k])
    return base.with_changes(**gen, switches=dict(switches or {}))


def load_exp(id_or_path: str | Path) -> Exp:
    p = Path(id_or_path)
    if p.suffix != ".yaml":
        p = EXP_DIR / f"{id_or_path}.yaml"
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    bad = sorted(set(d) - KEYS)
    if bad:
        raise KeyError(f"{p}: unknown key(s) {bad}; keys: {sorted(KEYS)}")
    if d.get("id") != p.stem:
        raise ValueError(f"{p}: id {d.get('id')!r} must equal the file name {p.stem!r}")
    base_name = d.get("base", "b0")
    base = base_config(base_name)
    gc = apply_changes(base, d.get("gen"), d.get("switches"))
    seeds = tuple(int(s) for s in d.get("seeds", [0]))
    return Exp(d["id"], base_name, base, gc, seeds, str(d.get("notes", "") or ""), check_train(d.get("train"), p), d,
               check_model(d.get("model"), p))


def check_model(model: dict | None, where="") -> dict:
    """The model: section with the default (off) values dropped, so a config that spells out 'known_attn: off' is B0's model
    and gets B0's run id."""
    model = dict(model or {})
    if model.get("known_attn") is False:                  # YAML 1.1 reads a bare `off` as false
        model["known_attn"] = "off"
    bad = sorted(set(model) - set(MODEL_KEYS))
    if bad:
        raise KeyError(f"{where}: unknown model option(s) {bad}; known: {sorted(MODEL_KEYS)}")
    ka = model.get("known_attn", "off")
    if ka not in MODEL_KEYS["known_attn"]:
        raise ValueError(f"{where}: model.known_attn must be one of {MODEL_KEYS['known_attn']}, not {ka!r}")
    K = model.get("known_lookahead")
    if K is not None and (isinstance(K, bool) or not isinstance(K, int) or K < 1 or ka == "off"):
        raise ValueError(f"{where}: model.known_lookahead must be an int >= 1 and needs known_attn: mask")
    return {k: v for k, v in model.items() if not (k == "known_attn" and v == "off") and v is not None}


def check_train(train: dict | None, where="") -> dict:
    train = dict(train or {})
    bad = sorted(set(train) - set(TRAIN_KEYS))
    if bad:
        raise KeyError(f"{where}: unknown train setting(s) {bad}; known: {sorted(TRAIN_KEYS)}")
    for k, typ in TRAIN_KEYS.items():
        if k in train and not isinstance(train[k], (int, float) if typ is float else typ):
            raise TypeError(f"{where}: train.{k} must be {typ.__name__}")
    if "long_row_frac" in train and not 0 <= train["long_row_frac"] <= 1:
        raise ValueError(f"{where}: train.long_row_frac must be in [0, 1]")
    bad = sorted(set(train.get("lead_noise", {})) - LEAD_NOISE_KEYS)
    if bad:
        raise KeyError(f"{where}: unknown train.lead_noise key(s) {bad}; known: {sorted(LEAD_NOISE_KEYS)}")
    return train


def parse_kv(items: list[str]) -> dict:
    """['p=0.3', 'point=final'] -> {'p': 0.3, 'point': 'final'} (values parsed as YAML scalars)."""
    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep:
            raise ValueError(f"expected key=value, got {it!r}")
        out[k.strip()] = yaml.safe_load(v)
    return out


def gc_from_cli(base: str = "b0", exp: str | None = None, switch: list[str] | None = None, param: list[str] | None = None,
                set_: list[str] | None = None) -> tuple[GenConfig, GenConfig, str]:
    """(base GenConfig, GenConfig, label) for the tools: --exp ID, or --base + --switch NAME + --param [NAME.]k=v +
    --set field=value (gen_additions.x=true for one addition)."""
    if exp:
        e = load_exp(exp)
        return e.base, e.gen, e.id
    b = base_config(base)
    sw = {n: {} for n in (switch or [])}
    for k, v in parse_kv(param).items():
        name, dot, key = k.rpartition(".")
        if not dot:
            if len(sw) != 1:
                raise ValueError(f"--param {k}: say which switch (NAME.{k}=...)")
            name = next(iter(sw))
        sw.setdefault(name, {})[key] = v
    gen: dict = {}
    for k, v in parse_kv(set_).items():
        if k.startswith("gen_additions."):
            gen.setdefault("gen_additions", {})[k.split(".", 1)[1]] = v
        else:
            gen[k] = v
    gc = apply_changes(b, gen, {n: (p or True) for n, p in sw.items()})
    return b, gc, gc.run_id_fragment(b)
