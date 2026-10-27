"""Run configs. A run is fully described by its RunConfig; the run id is a hash
of it. Bump PROTOCOL whenever a code change affects results, so rows written
by older code are never silently reused."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

PROTOCOL = 1
LITE = "lite"                 # protocol of every run in LEAN_PLAN.md (2026-10-11)

ARMS = ("softmax", "gla")                 # the main pair (P0 grids, diagnostics)
LAYER_KINDS = ("softmax", "gla", "swa", "gdn")   # every attention module behind the interface
ATTENTIONS = LAYER_KINDS + ("hybrid",)    # "hybrid": one kind per layer (layer_pattern), F1
TASKS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "C1a", "C1", "C1b", "C3", "C2", "C4f", "C4p", "C1L", "C1P")
HEADS = ("last", "flatten")
SELECTS = ("pooled", "per_series")

# Fields added after the first P0 runs, with the value every earlier run implicitly had.
# At that value they are left out of the run-id hash, so rows whose results do not change
# keep their ids; a run that sets a new value gets a new id.
LEGACY_DEFAULTS = {("model", "head"): "last", ("model", "head_tokens"): None,
                   ("train", "loss_scale"): "original", ("data_version",): 1,
                   ("model", "swa_window"): None, ("model", "layer_pattern"): None,
                   ("train", "select"): "pooled", ("distances",): (), ("contexts",): (),
                   ("prefix_max",): 0, ("train", "stop_min_steps"): None,
                   ("train_distances",): (), ("train", "plateau_stop"): True}


@dataclass(frozen=True)
class ModelConfig:
    attention: str                    # "softmax" | "gla" | "swa" | "hybrid": the arm (with
                                      # swa_window and layer_pattern, the only fields that differ)
    d_model: int
    n_layers: int
    n_heads: int
    ffn_ratio: int = 2                # d_ff = ffn_ratio * d_model
    dropout: float = 0.0              # FFN and group attention (never the time-attention weights)
    patch_len: int = 8
    group_attn: bool = False
    horizon: int = 16                 # steps emitted by the direct head
    eps: float = 1e-6
    # fla GLA published defaults (plan section 3); present in every arm's config
    gla_expand_k: float = 0.5
    gla_expand_v: float = 1.0
    gla_gate_rank: int = 16
    # forecast head: "last" = linear on the last token; "flatten" = one linear layer over
    # all token states (PatchTST style; its size grows with the number of tokens)
    head: str = "last"
    head_tokens: int | None = None    # number of context tokens, required by "flatten"
    swa_window: int | None = None     # tokens each query sees (itself included); SWA layers only
    # "hybrid" only: the attention kind of each layer, input side first (F1),
    # e.g. ("swa", "gla") = an SWA layer, then a GLA layer
    layer_pattern: tuple[str, ...] | None = None

    def __post_init__(self):
        if self.attention not in ATTENTIONS:
            raise ValueError(f"unknown attention {self.attention!r}")
        if (self.attention == "hybrid") != (self.layer_pattern is not None):
            raise ValueError("layer_pattern is required by, and only by, attention 'hybrid'")
        if self.layer_pattern is not None:
            if len(self.layer_pattern) != self.n_layers:
                raise ValueError("layer_pattern needs one entry per layer")
            if any(k not in LAYER_KINDS for k in self.layer_pattern):
                raise ValueError(f"layer_pattern entries must be in {LAYER_KINDS}")
        if self.uses("swa") != (self.swa_window is not None):
            raise ValueError("swa_window is required by, and only by, SWA layers")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.head not in HEADS:
            raise ValueError(f"unknown head {self.head!r}")
        if self.head == "flatten" and not self.head_tokens:
            raise ValueError("head 'flatten' needs head_tokens")

    def layer_attention(self, layer: int) -> str:
        return self.layer_pattern[layer] if self.layer_pattern is not None else self.attention

    def uses(self, kind: str) -> bool:
        return kind in (self.layer_pattern or (self.attention,))


@dataclass(frozen=True)
class TrainConfig:
    lr: float
    batch_size: int
    max_steps: int | None = None      # diagnostics
    max_epochs: int | None = None     # supervised
    patience: int | None = None       # supervised early stopping (epochs)
    val_every: int = 1000             # diagnostics: steps between validations
    warmup_frac: float = 0.05
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    bf16: bool = False                # autocast bf16 for training; evaluation is always fp32
    # "original": MSE in the data scale; "window_std": prediction and target divided by
    # the window's instance std (C3, whose series scales differ by orders of magnitude)
    loss_scale: str = "original"
    # control tasks (C3, C4): checkpoint selection and the reported error. "pooled": ratio
    # of sums over all series; "per_series": each series' squared errors divided by its
    # window variance first (Milestone 3: pooled C3 validation is dominated by a few
    # high-variance AR series)
    select: str = "pooled"
    # automatic stops (lite protocol, LEAN_PLAN rule 6): None = off. From this step on,
    # "solved" stops a run; "plateau" applies after half the schedule
    # (diagnostics/train.py: StopRule)
    stop_min_steps: int | None = None
    # False: the plateau stop is off (C2H); "solved" still applies
    plateau_stop: bool = True


@dataclass(frozen=True)
class RunConfig:
    exp: str
    task: str
    model: ModelConfig
    train: TrainConfig
    seed: int = 0
    lookback: int | None = None       # supervised
    n_values: tuple[int, ...] = ()    # recall tasks: N list (one N per training batch)
    distances: tuple[int, ...] = ()   # C2: gap lengths D in tokens (one D per training batch)
    contexts: tuple[int, ...] = ()    # C4: context lengths in timesteps (one per batch);
                                      # C1L: its one context length (timesteps)
    prefix_max: int = 0               # C2, C1P: Gaussian noise prefix of 0..prefix_max tokens
    n_val: int = 2000                 # diagnostics: fixed validation series per setting
    n_eval: int = 2000                # diagnostics: fixed test series per setting
    protocol: int | str = PROTOCOL   # 1 = P0 / Milestone 2 / full Milestone 3; "lite"
    # data generator version of the task; 1 = original. C3 = 2 since 2026-10-08 (GPU,
    # frequency-domain AR): same task definition, different random stream.
    data_version: int = 1
    # C2H: the gaps D that training batches are drawn from; () = the evaluation gaps in `distances` (C2)
    train_distances: tuple[int, ...] = ()

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f"unknown task {self.task!r}")
        if self.train.select not in SELECTS:
            raise ValueError(f"unknown select {self.train.select!r}")

    @property
    def arm(self) -> str:
        return arm_label(self.model)


def arm_label(m: ModelConfig) -> str:
    """softmax | gla | swa<w> | a hybrid: its layer kinds joined by '>' (input side first),
    e.g. 'swa32>gla'."""
    name = lambda k: f"swa{m.swa_window}" if k == "swa" else k  # noqa: E731
    if m.layer_pattern is not None:
        return ">".join(name(k) for k in m.layer_pattern)
    return name(m.attention)


def model_for_arm(m: ModelConfig, label: str) -> ModelConfig:
    """The same model with arm `label` (inverse of arm_label): only attention, swa_window
    and layer_pattern change."""
    kinds, window = [], None
    for part in label.split(">"):
        if part.startswith("swa") and part[3:].isdigit():
            w = int(part[3:])
            if window is not None and w != window:
                raise ValueError(f"one SWA window per model: {label!r}")
            kinds.append("swa")
            window = w
        elif part in ("softmax", "gla"):
            kinds.append(part)
        else:
            raise ValueError(f"unknown arm {label!r}")
    if len(kinds) == 1:
        return replace(m, attention=kinds[0], swa_window=window, layer_pattern=None)
    return replace(m, attention="hybrid", swa_window=window, layer_pattern=tuple(kinds))


def config_dict(cfg: RunConfig) -> dict:
    d = asdict(cfg)
    for path, legacy in LEGACY_DEFAULTS.items():
        parent = d
        for key in path[:-1]:
            parent = parent[key]
        if parent[path[-1]] == legacy:
            del parent[path[-1]]
    return d


def config_json(cfg: RunConfig) -> str:
    return json.dumps(config_dict(cfg), sort_keys=True, separators=(",", ":"))


def run_id(cfg: RunConfig) -> str:
    return hashlib.sha1(config_json(cfg).encode()).hexdigest()[:12]


def with_arm(cfg: RunConfig, arm: str) -> RunConfig:
    """Same run, other arm: only the attention fields change."""
    return replace(cfg, model=model_for_arm(cfg.model, arm))


_TUPLES = ("n_values", "distances", "contexts", "train_distances")


def config_from_json(text: str) -> RunConfig:
    """Inverse of config_json (fields left out of the hash take their defaults, which are
    their legacy values). Rebuilds a run's model for the saved weights (D1, D2)."""
    d = json.loads(text)
    model = d.pop("model")
    if model.get("layer_pattern") is not None:
        model["layer_pattern"] = tuple(model["layer_pattern"])
    train = d.pop("train")
    train["betas"] = tuple(train["betas"])
    return RunConfig(model=ModelConfig(**model), train=TrainConfig(**train),
                     **{k: tuple(v) if k in _TUPLES else v for k, v in d.items()})
