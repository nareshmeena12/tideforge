"""Twin configs (Milestone 4, LEAN_PLAN.md section 4).

A twin is fully described by its TwinConfig; the run id is a hash of it. Fields that do
not change results (rows per GPU, compile, paths, the fla kernel for packed rows) live in
the training options, not here. Every value except `arm` is the same for all arms.

Model: the 30M Tide decoder (4.1). Data: the 100M script's mix (reference/
tide_v5_100m_300b.py, RUN section): clean2 + repeat + daytype + weekday + long hidden spans
(cpm_long_p 0.3) + yearly; no one-off anomalies, no round-3 covariates; holdout corpus
"clean", fixed. Batch: a global 512 rows x 8,192 points as in the 100M script (owner,
2026-10-15; this replaces the 1,024 in LEAN_PLAN 4.1's first draft).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace

from tidemark.pretrain.arms import ALL_ARMS, M4_ARMS, canonical, layer_kinds

M4_PROTOCOL = "m4"
M4_SEEDS = (0, 1)
PILOT_LRS = (1.5e-4, 3e-4)
PILOT_POINTS = 5e9
QUANTILE_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
GEN_ADDITIONS = dict(agg=True, periods=True, epidemic=True, repeat=True, daytype=True, weekday=True,
                     yearly=True, oneoff=False, bursty=False, breaks=False, tv_season=False, sarima=False)


def arm_window(arm: str) -> int | None:
    """SWA window in patches (swa8 -> 8), None for softmax / GLA and for every mixed arm (their windows are
    in the layer pattern, see pretrain.arms)."""
    if arm not in ALL_ARMS:
        raise ValueError(f"unknown twin arm {arm!r}; arms: {ALL_ARMS}")
    return int(arm[3:]) if arm in ("swa8", "swa16") else None


@dataclass(frozen=True)
class TwinModel:
    patch_len: int = 32
    d_model: int = 512
    d_kv: int = 64
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 6
    dropout: float = 0.1          # FFN and group attention; never the time-attention weights
    eps: float = 1e-6
    group_lanes: int = 8
    quantile_levels: tuple[float, ...] = QUANTILE_LEVELS
    arcsinh: bool = True
    horizon_loss_share: float = 0.5
    max_pass_patches: int = 32
    # fla GLA published defaults (plan section 3)
    gla_expand_k: float = 0.5
    gla_expand_v: float = 1.0
    gla_gate_rank: int = 16


@dataclass(frozen=True)
class TwinData:
    max_ctx: int = 8192
    max_segs_per_row: int = 32
    min_ctx: int = 8
    long_row_frac: float = 0.25
    long_min_ctx: int = 2048
    len_bucket_edges: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
    gen_pool_points: float = 16e6
    gen_pool_max_rows: int = 32768
    corpus: str = "clean2"
    gen_additions: tuple[tuple[str, bool], ...] = tuple(sorted(GEN_ADDITIONS.items()))
    period_ext_share: float = 0.15
    val_corpus: str = "clean"
    group_prob: float = 0.7
    val_group_prob: float = 0.7
    cov_r3: bool = False
    cpm_prob: float = 0.8
    cpm_min_patches: int = 2
    cpm_max_patches: int = 16
    cpm_suffix_frac: float = 0.75
    cpm_long_p: float = 0.3
    cpm_long_max: int = 32
    block_steps: int = 200        # data stream reseeded every 200 steps: (seed, rank, step)
    holdout_rows: int = 1024      # the 100M script's 2 batches x 512 rows
    holdout_batch_rows: int = 64
    holdout_seed: int = 1234


@dataclass(frozen=True)
class TwinTrain:
    target_points: float = 100e9
    global_rows: int = 512
    warmup_pct: float = 0.02
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    bf16: bool = True
    val_every: int = 1000
    print_every: int = 50
    milestones: tuple[float, ...] = (25e9, 50e9)
    spike_warm: int = 1000
    spike_n: int = 256
    spike_loss_x: float = 2.5
    spike_gn_x: float = 8.0
    spike_rollback_after: int = 20
    val_rollback_x: float = 1.10
    win_rollback_x: float = 1.30
    win_rollback_count: int = 3
    max_rollbacks: int = 3


@dataclass(frozen=True)
class TwinConfig:
    arm: str
    seed: int
    lr: float
    model: TwinModel = field(default_factory=TwinModel)
    data: TwinData = field(default_factory=TwinData)
    train: TwinTrain = field(default_factory=TwinTrain)
    protocol: str = M4_PROTOCOL
    pattern: str = ""             # per-layer kinds; "" = the arm's registered pattern (left out of the run id)

    def __post_init__(self):
        if not self.pattern:
            arm_window(self.arm)
        layer_kinds(self.arm, self.pattern, self.model.n_layers)
        if self.train.global_rows % self.model.group_lanes:
            raise ValueError("global_rows must be a multiple of group_lanes")
        if self.data.max_ctx % self.model.patch_len:
            raise ValueError("max_ctx must be a multiple of patch_len")

    @property
    def window(self) -> int | None:
        return arm_window(self.arm)

    @property
    def kinds(self) -> tuple[str, ...]:
        """Kind of every layer, input side first, e.g. ('gla', 'gla', 'softmax')."""
        return layer_kinds(self.arm, self.pattern, self.model.n_layers)

    @property
    def name(self) -> str:
        return f"{self.arm}_s{self.seed}_lr{self.lr:g}"

    def gen_cfg(self, rows: int) -> dict:
        """The reference's CONFIG keys that PackedBatcher and the generators read."""
        m, d = self.model, self.data
        return dict(patch_len=m.patch_len, max_ctx=d.max_ctx, pack_rows=rows, max_segs_per_row=d.max_segs_per_row,
                    decoder="causal", pred_len=128, group_attn=True, group_lanes=m.group_lanes,
                    group_prob=d.group_prob, val_group_prob=d.val_group_prob, corpus=d.corpus,
                    val_corpus=d.val_corpus, gen_additions=dict(d.gen_additions),
                    period_ext_share=d.period_ext_share, cov_r3=d.cov_r3, min_ctx=d.min_ctx,
                    long_row_frac=d.long_row_frac, long_min_ctx=d.long_min_ctx,
                    len_bucket_edges=list(d.len_bucket_edges), gen_pool_points=d.gen_pool_points,
                    gen_pool_max_rows=d.gen_pool_max_rows, cpm_prob=d.cpm_prob,
                    cpm_min_patches=d.cpm_min_patches, cpm_max_patches=d.cpm_max_patches,
                    cpm_suffix_frac=d.cpm_suffix_frac, cpm_long_p=d.cpm_long_p, cpm_long_max=d.cpm_long_max)


def twin_config(arm: str, seed: int, lr: float, **over) -> TwinConfig:
    """The twin of one arm; `over` replaces sub-configs (tests use tiny ones)."""
    pattern = over.pop("pattern", "")
    return replace(TwinConfig(arm=canonical(arm), seed=seed, lr=float(lr), pattern=pattern), **over)


def config_json(cfg: TwinConfig) -> str:
    d = asdict(cfg)
    if not d["pattern"]:                  # new field left out at its default: old run ids and configs unchanged
        del d["pattern"]
    return json.dumps(d, sort_keys=True)


def run_id(cfg: TwinConfig) -> str:
    return hashlib.sha1(config_json(cfg).encode()).hexdigest()[:12]


def config_from_json(text: str) -> TwinConfig:
    d = json.loads(text)
    m = TwinModel(**{**d["model"], "quantile_levels": tuple(d["model"]["quantile_levels"])})
    dd = dict(d["data"])
    dd["len_bucket_edges"] = tuple(dd["len_bucket_edges"])
    dd["gen_additions"] = tuple(tuple(x) for x in dd["gen_additions"])
    t = dict(d["train"])
    t["betas"], t["milestones"] = tuple(t["betas"]), tuple(t["milestones"])
    return TwinConfig(arm=d["arm"], seed=d["seed"], lr=d["lr"], model=m, data=TwinData(**dd),
                      train=TwinTrain(**t), protocol=d["protocol"], pattern=d.get("pattern", ""))
