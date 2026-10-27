"""GenConfig: the generator version and every generation setting, plus a stable run-id fragment.

The fields are the cfg keys that ft3_core's generation reads (resolve_profile, set_group_options, PackedBatcher's group mode and
the holdout settings), with ft3_core CONFIG's values as defaults. The defaults are NOT B0: B0 is ft3_main's real cfg, loaded with
b0() from configs/ft3_main_cfg.json (no silent fallback when it is missing).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from tideforge.gen import switches as _switches
from tideforge.gen.profiles import GEN_ADDITIONS_ALL

REPO = Path(__file__).resolve().parents[2]
FT3_MAIN_CFG = REPO / "configs" / "ft3_main_cfg.json"
VERSION = "v0"

# ft3_core CONFIG["gen_additions"]
_CONFIG_ADDITIONS = dict(agg=True, periods=True, epidemic=True, repeat=True, daytype=True, weekday=True,
                         yearly=False, oneoff=False, bursty=False, breaks=False, tv_season=False, sarima=False)


def _additions(d: dict | None) -> tuple[tuple[str, bool], ...]:
    """Effective additions: d merged onto GEN_ADDITIONS_ALL, exactly as make_clean2_profile does."""
    a = dict(GEN_ADDITIONS_ALL)
    for k, v in (d or {}).items():
        if k not in a:
            raise KeyError(f"unknown gen_addition {k!r}; valid: {sorted(a)}")
        a[k] = bool(v)
    return tuple(sorted(a.items()))


@dataclass(frozen=True)
class GenConfig:
    version: str = VERSION
    corpus: str = "clean2"
    gen_additions: tuple[tuple[str, bool], ...] = field(default_factory=lambda: _additions(_CONFIG_ADDITIONS))
    period_ext_share: float | None = 0.15
    # group mode (PackedBatcher: only with decoder="causal" and group_attn; else group_lanes 1 and group_prob 0)
    group_lanes: int = 8
    group_prob: float = 0.7
    group_kind_probs: tuple[float, ...] | None = None
    cov_small_group_p: float = 0.0
    cov_dominant_p: float = 0.0
    second_group_p: float = 0.0
    cov_r3: bool = False
    # holdout (validation) stream
    val_corpus: str | None = "clean"
    val_group_prob: float = 0.7
    # tideforge switches (tideforge/gen/switches.py); v0 has none
    switches: tuple[tuple[str, object], ...] = ()

    def __post_init__(self):
        _switches.validate(dict(self.switches))

    @classmethod
    def from_ft3_cfg(cls, cfg: dict) -> "GenConfig":
        """The generation settings of an ft3 training cfg (e.g. ft3_main.pt["cfg"])."""
        group = cfg.get("decoder", "pool") == "causal" and bool(cfg.get("group_attn", False))
        gkp = cfg.get("group_kind_probs")
        return cls(corpus=cfg.get("corpus", "v1"),
                   gen_additions=_additions(cfg.get("gen_additions")),
                   period_ext_share=cfg.get("period_ext_share", None),
                   group_lanes=int(cfg.get("group_lanes", 8)) if group else 1,
                   group_prob=float(cfg.get("group_prob", 0.5)) if group else 0.0,
                   group_kind_probs=tuple(float(x) for x in gkp) if gkp else None,
                   cov_small_group_p=float(cfg.get("cov_small_group_p") or 0.0),
                   cov_dominant_p=float(cfg.get("cov_dominant_p") or 0.0),
                   second_group_p=float(cfg.get("second_group_p") or 0.0),
                   cov_r3=bool(cfg.get("cov_r3") or False),
                   val_corpus=cfg.get("val_corpus"),
                   val_group_prob=float(cfg.get("val_group_prob", cfg.get("group_prob", 0.5))) if group else 0.0)

    def to_ft3_cfg(self) -> dict:
        """The cfg keys ft3_core's resolve_profile / set_group_options read."""
        return dict(corpus=self.corpus, gen_additions=dict(self.gen_additions), period_ext_share=self.period_ext_share,
                    group_kind_probs=self.group_kind_probs, cov_small_group_p=self.cov_small_group_p,
                    cov_dominant_p=self.cov_dominant_p, second_group_p=self.second_group_p, cov_r3=self.cov_r3)

    def with_changes(self, **kw) -> "GenConfig":
        if "gen_additions" in kw and isinstance(kw["gen_additions"], dict):
            kw["gen_additions"] = _additions(dict(dict(self.gen_additions), **kw["gen_additions"]))
        if "switches" in kw and isinstance(kw["switches"], dict):
            kw["switches"] = tuple(sorted(kw["switches"].items()))
        return replace(self, **kw)

    def diff(self, base: "GenConfig") -> dict:
        """{key: value} where self differs from base; gen_additions and switches are compared entry by entry."""
        out = {}
        a, b = asdict(self), asdict(base)
        for k in a:
            if k in ("gen_additions", "switches"):
                da, db = dict(a[k]), dict(b[k])
                out.update({f"{k}.{n}": da.get(n) for n in sorted(set(da) | set(db)) if da.get(n) != db.get(n)})
            elif a[k] != b[k]:
                out[k] = a[k]
        return out

    def config_hash(self) -> str:
        s = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]

    def run_id_fragment(self, base: "GenConfig") -> str:
        """'v0-b0-1a2b3c4d' for B0 itself, else 'v0-' + the changed settings + '-' + hash, e.g. 'v0-breaks-cfa09e12'."""
        d = self.diff(base)
        parts = []
        for k, v in d.items():
            name = k.split(".", 1)[1] if "." in k else k
            parts.append(name if v is True else f"no{name}" if v is False else f"{name}{v}")
        tag = "+".join(parts) if parts else "b0"
        return f"{self.version}-{tag}-{self.config_hash()}"


def load_ft3_cfg(path: Path | str = FT3_MAIN_CFG) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{p} is missing: B0 is ft3_main's real cfg (configs/README.md says how to dump it)")
    return json.loads(p.read_text(encoding="utf-8"))


def b0(path: Path | str = FT3_MAIN_CFG) -> GenConfig:
    """Baseline B0 = ft3_main's generation settings. Raises if configs/ft3_main_cfg.json is missing."""
    return GenConfig.from_ft3_cfg(load_ft3_cfg(path))
