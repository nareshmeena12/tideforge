"""Verbatim copy of reference/tide_v5_ft3.py lines 69-4086 (Fluxion-Tide v5, round 3): model, synthetic generators, group
builder, PackedBatcher, inference helpers, probes, holdout and the reference single-GPU training loop. Produced by
scripts/real/extract_ft3.py; DO NOT EDIT BY HAND (a test compares it with the reference). Nothing here runs on import except the
definitions and the constants of the reference, and `_T0 = time.time()`. The RUN section of the reference (it trains) is not copied.
"""
import math, os, time, json, glob, gc, shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_T0 = time.time()     # session clock: Kaggle stops a session at a hard limit (see max_hours)

# ============================== 1. CONFIG ==================================
CONFIG = dict(
    # ---- where checkpoints go. NEW dir: packed checkpoints are not drop-in compatible
    # with the old ones (mask channel changes the patch embedding shape).
    out_dir        = "/kaggle/working/ft/_unset",   # set per run in the RUN section
    max_hours      = 11.0,     # stop cleanly (last.pt saved) this many hours after the cell started;
                               # keep it below Kaggle's session limit. None = no time limit.
    resume_from    = None,     # "auto": newest last.pt found under /kaggle/input (the Output of an
                               # earlier session added as Input) | a path to a .pt file | None

    # ---- point budget. Counts REAL points only: observed context points + target
    # points. Left-padding is never counted.
    target_points  = 100e9,    # 1 TRILLION points. The LR schedule is stretched over the whole budget.
    milestone_points = None,   # also save ckpt_100B.pt, ckpt_200B.pt, ... (never overwritten): evaluate
                               # them WHILE the run continues, and they give the scaling curve for free
    ema_decay      = None,     # running average of the weights, saved next to the raw weights in every
                               # checkpoint (eval can score either). None = off.

    # ---- row geometry. Every row is `max_ctx` points = max_ctx/patch_len patches.
    max_ctx        = 8192,
    pred_len       = 128,
    patch_len      = 32,
    pack_rows      = 1024,      # rows per step (the old batch_size). Drop to 384/256 on OOM.
    max_segs_per_row = 32,     # cap on series per row (bounds the head / loss slots)

    # ---- context-length distribution
    min_ctx        = 8,        # shortest context (points). Left-padded to a patch boundary.
    long_row_frac  = 0.25,     # fraction of rows whose FIRST series is long, so long-context
    long_min_ctx   = 2048,     # skill is not starved (log-uniform alone makes 8k contexts rare)

    # ---- generation. Series are generated per length bucket (a series with context c is
    # generated at the smallest bucket >= c, then randomly cropped to c + pred_len).
    # Generating per bucket keeps period filtering etc. appropriate to the actual length.
    len_bucket_edges = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192],
    gen_pool_points  = 16e6,   # points generated per bucket refill (amortises kernel launches)
    gen_pool_max_rows = 32768, # cap rows per refill (tiny buckets would otherwise be huge)

    # ---- v2 switches (run them as separate ablation steps if you want attribution)
    decoder        = "causal", # "causal" = dense next-patch decoder | "pool" = v1 mean-pool head
    corpus         = "clean2", # "clean2" = clean + gen_additions | "clean" = v2 corpus | "v1"
    # clean2 switches. The fev-bench result of the all-on run (37.72 vs 37.29 for v2, a tie)
    # decided the defaults: weekly/monthly tasks improved 4.4% (agg + periods) and ecdc_ili 10%
    # (epidemic), while hourly electricity load at 168 steps got 10-33% worse and redset 9%
    # worse. So only the three with evidence stay ON; the other five remain available as
    # switches (off) for single-addition ablations.
    gen_additions  = dict(agg=True, periods=True, epidemic=True, repeat=True, daytype=True, weekday=True,
                          yearly=False, oneoff=False,
                          bursty=False, breaks=False, tv_season=False, sarima=False),
    # ROUND 3: repeat, weekday (daily_weekly) and daytype (repeat_week) are ON in every run: they are
    # what ft2_main was trained with. "yearly" turns on the new long_cycle family (a long period seen
    # only 1-4 times) and "oneoff" puts one-off anomalies on 30% of the repeating series.
    # share of seasonal periods drawn from the NEW periods (13/26/52/672/1008/2016). The all-on
    # run picked uniformly from all 18 periods, which cut the share of 24- and 168-step cycles
    # by a third across the WHOLE corpus -- a likely cause of the hourly-load regression. Now
    # the original 12 periods keep 85% of the mass. None = uniform over all 18 (the v3 run).
    period_ext_share = 0.15,
    # holdout corpus for val loss. Keep it FIXED ("clean") across ablation runs so val losses
    # are comparable between runs; None = same as `corpus`. fev-bench is still the real metric.
    val_corpus     = "clean",

    # ---- GROUP ATTENTION (multivariate targets + covariates), causal decoder only.
    # Rows are organised in blocks of `group_lanes` rows that share one layout, so the variates
    # of a group sit in different rows at the SAME patch columns. Each block cell holds either
    # a related group (2..group_lanes variates: mixed factors, lead-lag, covariate-driven
    # targets, decoys) or unrelated single series. A group-attention layer after every time-
    # attention layer lets a token see the other variates of its group at the same patch.
    group_attn     = True,
    group_lanes    = 8,        # rows per block = max variates per training group; pack_rows % this == 0
    group_prob     = 0.7,      # fraction of block cells that hold a related group (v4: 0.5). With the
                               # new kind mix, covariate-driven targets go from ~2.3% to ~5.3% of series
    init_from      = None,     # optional: path to a univariate checkpoint (e.g. the v2 final.pt) to
                               # warm-start from; the group output is zeroed, so step 0 == that model.
                               # None = train from scratch (group layers initialised normally)
    # ---- FINE-TUNE switches (set per run in the RUN section)
    finetune_from  = None,     # checkpoint whose weights every run starts from (fresh optimizer + schedule)
    run_name       = "run",
    group_kind_probs = None,   # (mix, leadlag, known_cov, past_cov, unrelated); None = v5 (0.15, 0.10, 0.50, 0.10, 0.15)
    cov_small_group_p = 0.0,   # share of covariate groups limited to 2-4 variates (v5: sizes 2-8)
    cov_dominant_p = 0.0,      # share of covariate groups whose target is almost entirely its covariates
    second_group_p = 0.0,      # chance of a second related group in a block's upper lanes (when they are free)
    cov_r3         = False,    # ROUND 3 covariate data: rare shutdown flags, weak drivers, look-alike covariates that
                               # share the target's cycle, drivers that act through their sum. The holdout never
                               # uses it, so holdout losses stay comparable.
    val_group_prob = 0.7,      # the holdout always uses the base group share, so runs stay comparable
    probe_at_val   = True,     # print the repeat, week, year and flag probes at every validation
    cpm_prob       = 0.8,      # fraction of series with one contiguous span of patches hidden (v4: 0.5)
    cpm_min_patches = 2,       # shortest hidden span. A span of 1 creates NO multi-patch target at the
                               # end of a series (its only patch is forecast from real history)
    cpm_max_patches = 16,      # longest hidden span (16 x 32 = 512 steps; fev-bench max h=288)
    cpm_suffix_frac = 0.75,    # of those spans, fraction placed at the END (= a forecast) (v4: 0.5)
    cpm_long_p     = 0.3,      # share of LONG series (>= 35 patches) whose hidden span is drawn from (as in ft2_main)
    cpm_long_max   = 32,       # cpm_max_patches+1 .. cpm_long_max instead (never more than half the series).
                               # 0 = off. The holdout never uses it, so holdout losses stay comparable.
    # MULTI-PATCH LOSS SHARE. A forecast for patch p+1 is made either from real history (the
    # input patch p is observed: "one-patch") or from a hidden placeholder patch ("multi-patch":
    # the 2nd, 3rd, ... patch of a horizon). In v4 a series' loss was the plain mean over all
    # its forecasts, so multi-patch forecasts were ~2% of the signal -- and on fev-bench v4 was
    # 5% behind Chronos-2 (univariate) on one-patch horizons but 17% behind on longer ones.
    # Now each series' loss = (1 - share) * mean(one-patch) + share * mean(multi-patch).
    # None = the v4 plain mean.
    horizon_loss_share = 0.5,
    max_pass_patches = 32,     # inference: horizon patches read in one pass before rollout

    # ---- model input (pool decoder only; the causal decoder always uses both)
    mask_channel   = True,     # feed the validity mask as a 2nd patch channel. Needed so the
                               # model can tell left-padding from genuine zeros.
    arcsinh        = True,     # asinh() after standardisation (as in Chronos-2 / t0). REQUIRED
                               # with short contexts: a flat 12-point context has std ~0, so a
                               # normal target lands thousands of std away. Measured without it:
                               # 1% of series carried 94% of the loss. asinh is linear near 0 and
                               # log-like in the tails; being monotone, quantiles map exactly.

    # ---- architecture (matches run1 -> ~21M params)
    d_model = 512, d_kv = 64, n_heads = 8, d_ff = 2048, n_layers = 6, dropout = 0.1,
    skip_group_attn = True,

    # ---- optimisation
    lr = 1e-4, warmup_pct = 0.03, grad_clip = 1.0,     # fine-tune: 4e-6 -> 1e-4 (3% of steps) -> 0
    compile_model = True,      # torch.compile(mode="default"); all packed shapes are static
    amp = "auto",              # auto -> bf16 on A100
    seed = 0,

    # ---- logging / checkpointing / resume
    print_every = 50, ckpt_every = 2000, val_every = 2000, val_batches = 2,
    resume = True,
    max_steps = 10_000_000,
)

RUN_INSPECTION = False    # corpus audit + packing layout audit before training
RUN_VERIFY     = True     # packed-vs-alone equivalence check before training (seconds)
RUN_GROUP_PROBE = False   # short throwaway run: does the model learn to USE covariates / sibling
                          # series? (minutes; prints the group gain; saves nothing)
RUN_TRAINING   = True     # set False to only inspect / verify

QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


# ======================= 2. MODEL ===========================================
class SharedRoPE(nn.Module):
    """Rotary tables computed fresh from the (fixed) inv_freq buffer on every call.

    Deliberately does NOT cache cos/sin in a Python dict: that cache is fatal under
    torch.compile(mode="reduce-overhead") (CUDA-graph pool overwrites). Recomputing is
    mathematically identical and negligible in cost.
    """

    def __init__(self, dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    # q,k: (B, H, T, D) ; cos,sin: (T, D)
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class FusedQKV(nn.Module):
    def __init__(self, d_model, d_kv, n_heads, n_kv_heads=None):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.d_kv = d_kv
        self.q_dim = n_heads * d_kv
        self.kv_dim = self.n_kv_heads * d_kv
        self.proj = nn.Linear(d_model, self.q_dim + 2 * self.kv_dim, bias=False)

    def forward(self, x):
        qkv = self.proj(x)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        B, T, _ = x.shape
        q = q.view(B, T, self.n_heads, self.d_kv).transpose(1, 2)
        k = k.view(B, T, self.n_kv_heads, self.d_kv).transpose(1, 2)
        v = v.view(B, T, self.n_kv_heads, self.d_kv).transpose(1, 2)
        return q, k, v


def rms_norm(x, weight, eps):
    return F.rms_norm(x, (x.shape[-1],), weight.to(x.dtype), eps)


class TimeSelfAttention(nn.Module):
    """Full (non-causal) self-attention over the patch sequence, with RoPE.
    `attn_mask` (B, 1, T, T) bool, True = may attend. None = attend everywhere."""

    def __init__(self, d_model, d_kv, n_heads, dropout, eps, rope: SharedRoPE):
        super().__init__()
        self.qkv = FusedQKV(d_model, d_kv, n_heads)
        self.out_proj = nn.Linear(n_heads * d_kv, d_model, bias=False)
        self.norm_w = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.dropout_p = dropout
        self.rope = rope
        self.d_kv = d_kv

    def forward(self, x, attn_mask=None, is_causal=False):
        B, T, D = x.shape
        h = rms_norm(x, self.norm_w, self.eps)
        q, k, v = self.qkv(h)
        cos, sin = self.rope(T, x.device, q.dtype)
        q, k = apply_rope(q, k, cos, sin)
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal and attn_mask is None,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        o = o.transpose(1, 2).reshape(B, T, -1)
        return x + self.out_proj(o)


class GroupSelfAttention(nn.Module):
    """Cross-series attention across the batch axis (Chronos-2 'group' mechanism).
    NOT packing-aware -- keep skip_group_attn=True with packed training."""

    def __init__(self, d_model, d_kv, n_heads, dropout, eps):
        super().__init__()
        self.qkv = FusedQKV(d_model, d_kv, n_heads)
        self.out_proj = nn.Linear(n_heads * d_kv, d_model, bias=False)
        self.norm_w = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.dropout_p = dropout

    def forward(self, x, group_mask=None):
        B, T, D = x.shape
        xt = x.transpose(0, 1)
        h = rms_norm(xt, self.norm_w, self.eps)
        q, k, v = self.qkv(h)
        o = F.scaled_dot_product_attention(
            q, k, v, attn_mask=group_mask, is_causal=False,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        o = o.transpose(1, 2).reshape(T, B, -1)
        out = xt + self.out_proj(o)
        return out.transpose(0, 1)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout, eps):
        super().__init__()
        self.norm_w = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.wi = nn.Linear(d_model, d_ff, bias=False)
        self.wo = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        h = rms_norm(x, self.norm_w, self.eps)
        return x + self.drop(self.wo(self.act(self.wi(h))))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, d_kv, n_heads, d_ff, dropout, eps, rope, skip_group_attn):
        super().__init__()
        self.time = TimeSelfAttention(d_model, d_kv, n_heads, dropout, eps, rope)
        self.skip_group_attn = skip_group_attn
        self.group = None if skip_group_attn else GroupSelfAttention(d_model, d_kv, n_heads, dropout, eps)
        self.ffn = FeedForward(d_model, d_ff, dropout, eps)

    def forward(self, x, attn_mask=None, group_mask=None, is_causal=False):
        x = self.time(x, attn_mask, is_causal)
        if not self.skip_group_attn:
            x = self.group(x, group_mask)
        return self.ffn(x)


class Chronos2Style(nn.Module):
    def __init__(self, patch_len=32, pred_len=256, d_model=512, d_kv=64, n_heads=8,
                 d_ff=2048, n_layers=6, dropout=0.1, eps=1e-6, skip_group_attn=True,
                 quantile_levels=QUANTILE_LEVELS, mask_channel=False, arcsinh=False):
        super().__init__()
        self.patch_len = patch_len
        self.pred_len = pred_len
        self.quantile_levels = quantile_levels
        self.n_q = len(quantile_levels)
        self.mask_channel = mask_channel
        self.arcsinh = arcsinh

        self.rope = SharedRoPE(d_kv)
        self.patch_embed = nn.Linear(patch_len * (2 if mask_channel else 1), d_model)
        self.blocks = nn.ModuleList([
            EncoderBlock(d_model, d_kv, n_heads, d_ff, dropout, eps, self.rope, skip_group_attn)
            for _ in range(n_layers)
        ])
        self.final_norm_w = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.head = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Linear(d_ff, pred_len * self.n_q),
        )

    # ---- shared pieces ----------------------------------------------------
    def _embed(self, vals_norm, mask):
        """vals_norm, mask: (B, L), L a multiple of patch_len. Pads must already be 0."""
        B, L = vals_norm.shape
        pl = self.patch_len
        P = L // pl
        v = vals_norm[:, :P * pl].reshape(B, P, pl)
        if self.mask_channel:
            m = mask[:, :P * pl].reshape(B, P, pl).to(v.dtype)
            v = torch.cat([v, m], dim=-1)
        return self.patch_embed(v)

    def _fwd_tf(self, z):
        """standardised value -> model space"""
        return torch.asinh(z) if self.arcsinh else z

    def _inv_tf(self, o):
        """model space -> standardised value (clamped so sinh can never overflow)"""
        return torch.sinh(o.clamp(-20.0, 20.0)) if self.arcsinh else o

    def _trunk(self, x, attn_mask=None):
        for blk in self.blocks:
            x = blk(x, attn_mask=attn_mask)
        return rms_norm(x, self.final_norm_w, self.eps)

    def _pinball(self, out, target_norm):
        """Per-series pinball loss, averaged over horizon and quantiles -> (N,)."""
        q = torch.tensor(self.quantile_levels, device=out.device, dtype=out.dtype)
        e = target_norm.unsqueeze(-1) - out
        return torch.maximum(q * e, (q - 1) * e).mean(dim=(1, 2))

    # ---- forward ----------------------------------------------------------
    def forward(self, ctx, target=None, mask=None, seg_patch=None, slot_valid=None):
        """
        UNPACKED (one series per row, used at inference):
          ctx    (B, L) raw values, L a multiple of patch_len. Left-pad short series with 0
                 and pass mask=False on the padding.
          target (B, pred_len) optional.   mask (B, L) bool, True = observed.
        PACKED (training): pass seg_patch (R, P) long = series slot of every patch, and
          slot_valid (S,) bool. ctx/mask are (R, W) packed rows; target is (S, pred_len).
        Returns dict(quantile_preds (N, pred_len, n_q) in data scale, [loss, per_series_loss]).
        """
        if seg_patch is not None:
            return self._forward_packed(ctx, mask, seg_patch, target, slot_valid)

        B, L = ctx.shape
        if mask is None:
            mask = torch.ones_like(ctx, dtype=torch.bool)
        ctx = torch.where(mask, ctx, torch.zeros_like(ctx))
        maskf = mask.to(ctx.dtype)
        cnt = maskf.sum(dim=1, keepdim=True).clamp(min=1)
        mean = (ctx * maskf).sum(dim=1, keepdim=True) / cnt
        var = ((ctx - mean) ** 2 * maskf).sum(dim=1, keepdim=True) / cnt
        std = var.clamp(min=1e-6).sqrt()
        ctx_norm = torch.where(mask, self._fwd_tf((ctx - mean) / std), torch.zeros_like(ctx))

        x = self._trunk(self._embed(ctx_norm, mask))
        pl = self.patch_len
        P = x.shape[1]
        pv = mask[:, :P * pl].reshape(B, P, pl).any(-1).to(x.dtype)       # patch has data
        pooled = (x * pv.unsqueeze(-1)).sum(1) / pv.sum(1, keepdim=True).clamp(min=1)
        out = self.head(pooled).view(B, self.pred_len, self.n_q)
        result = {"quantile_preds": self._inv_tf(out.float()) * std.unsqueeze(-1) + mean.unsqueeze(-1)}
        if target is not None:
            per = self._pinball(out.float(), self._fwd_tf(((target - mean) / std).float()))
            result["loss"] = per.mean()
            result["per_series_loss"] = per
        return result

    def _forward_packed(self, x_pts, mask, seg_patch, target, slot_valid):
        R, W = x_pts.shape
        pl = self.patch_len
        P = W // pl
        S = slot_valid.shape[0]
        dev = x_pts.device

        # ---- per-series instance norm (statistics never cross a series boundary)
        seg_pt = seg_patch.repeat_interleave(pl, dim=1)                   # (R, W)
        flat_seg = seg_pt.reshape(-1)
        mf = mask.reshape(-1).float()
        xv = torch.where(mask, x_pts, torch.zeros_like(x_pts)).reshape(-1).float()
        cnt = torch.zeros(S, device=dev).index_add_(0, flat_seg, mf).clamp(min=1)
        mean = torch.zeros(S, device=dev).index_add_(0, flat_seg, xv) / cnt
        dv = (xv - mean[flat_seg]) * mf
        var = torch.zeros(S, device=dev).index_add_(0, flat_seg, dv * dv) / cnt
        std = var.clamp(min=1e-6).sqrt()
        xn = torch.where(mask, self._fwd_tf((x_pts.float() - mean[seg_pt]) / std[seg_pt]),
                         torch.zeros((), device=dev)).to(x_pts.dtype)

        # ---- block-diagonal attention: a patch only attends within its own series
        attn_mask = (seg_patch.unsqueeze(2) == seg_patch.unsqueeze(1)).unsqueeze(1)
        h = self._trunk(self._embed(xn, mask), attn_mask)                 # (R, P, D)

        # ---- per-series mean pool over patches that contain observed points
        pv = mask.reshape(R, P, pl).any(-1).reshape(-1).float()
        flat_sp = seg_patch.reshape(-1)
        hs = h.reshape(R * P, -1).float() * pv.unsqueeze(-1)
        pooled = torch.zeros(S, h.shape[-1], device=dev).index_add_(0, flat_sp, hs)
        pcnt = torch.zeros(S, device=dev).index_add_(0, flat_sp, pv).clamp(min=1)
        pooled = (pooled / pcnt.unsqueeze(-1)).to(h.dtype)

        out = self.head(pooled).view(S, self.pred_len, self.n_q)
        result = {"quantile_preds": self._inv_tf(out.float()) * std[:, None, None] + mean[:, None, None]}
        if target is not None:
            per = self._pinball(out.float(), self._fwd_tf((target.float() - mean[:, None]) / std[:, None]))
            v = slot_valid.float()
            result["loss"] = (per * v).sum() / v.sum().clamp(min=1)
            result["per_series_loss"] = per
        return result




# ======================= 2b. CAUSAL PATCH DECODER (decoder="causal") =======================
#
# Why this replaces the mean-pool head (Chronos2Style above):
#  1. RECENCY. Mean-pooling every patch into one vector gives the last patch -- the most
#     informative one for a forecast -- the same weight as history 8000 steps back.
#     Chronos-2, t0, TiRex, Toto and TimesFM all decode from positions, never a pooled mean.
#  2. DENSE SUPERVISION. The pooled head yields ONE forecast target per series. Here every
#     patch predicts the next patch, so an 8192-point series yields up to 255 targets --
#     far more learning signal per generated point. And because patch p only sees patches
#     <= p, every series also trains every context length from 1 point upward.
#  3. HORIZON WITHOUT A GROWING HEAD. The head always emits one patch. Longer horizons are
#     read from BLANKED future patches appended to the context (contiguous patch masking,
#     TiRex / t0 / Toto-2): blanked patches are shown as mask=0 during training, so at
#     inference the model forecasts several patches ahead in ONE forward pass.
#  4. CAUSAL NORMALISATION (t0): the input at step t is standardised with the running mean /
#     std of observed points up to t; the target of the forecast issued at the end of patch p
#     is standardised with the statistics FROZEN at that point. No future information can
#     reach the model through scaling -- the classic silent leak of instance normalisation.
#  5. MONOTONE QUANTILES by construction (lowest level + cumulative softplus increments).
#
# Causal attention also makes this the natural baseline for a GLA comparison (GLA is causal).

def _monotone_quantiles(o):
    """(..., n_q) raw -> increasing quantiles: q0 = o0, q_k = q_{k-1} + softplus(o_k)."""
    return torch.cat([o[..., :1], o[..., :1] + torch.cumsum(F.softplus(o[..., 1:]), dim=-1)], dim=-1)


def _causal_stats(x, obs, seg_pt, seg_start_pt, n_slots, std_floor=1e-3, patch_len=32):
    """Running mean / std of OBSERVED points of the same series, from the series' first point
    up to and including t. x, obs: (R, W); seg_pt: (R, W) global series id; seg_start_pt:
    (R, W) column where that series starts in its row (kept for API compatibility).
    Returns mu, sd, count (R, W).

    Computed in float64 after subtracting each series' first observed value, and with a
    SEGMENTED scan that never sums across series: within-patch cumsums (a patch never holds
    two series) + the totals of the series' OWN earlier patches via a (P x P) same-series,
    strictly-earlier mask. The previous version took one cumsum over the whole ROW and
    subtracted the value at the series start; a single huge series (values ~1e9, squares
    ~1e18) earlier in the row left float64 rounding error ~1e6 in the running sum of squares
    of every later series -- wrong normalisation for them in training, and the packed-vs-alone
    check failed (rel diff 7.5 on a clean2 GPU batch). Now each series' statistics depend only
    on its own values, packed or alone."""
    R, W = x.shape
    dev = x.device
    pl = patch_len
    P = W // pl
    col = torch.arange(W, device=dev).expand(R, W)
    first = torch.full((n_slots,), W - 1, device=dev, dtype=torch.long).scatter_reduce(
        0, seg_pt.reshape(-1), torch.where(obs, col, torch.full_like(col, W - 1)).reshape(-1),
        reduce="amin", include_self=True)
    shift = torch.gather(x, 1, first[seg_pt]).double()
    o = obs.double()
    xs = torch.where(obs, x.double() - shift, torch.zeros((), device=dev, dtype=torch.float64))

    v = torch.stack([o, xs, xs * xs], dim=-1).view(R, P, pl, 3)
    local = v.cumsum(2)                                              # own patch only
    segp = seg_pt[:, ::pl]                                           # (R, P) series of each patch
    pidx = torch.arange(P, device=dev)
    earlier = ((segp.unsqueeze(2) == segp.unsqueeze(1))
               & (pidx.view(1, P, 1) > pidx.view(1, 1, P))).double()  # [r, p, q]: q<p, same series
    before = torch.bmm(earlier, local[:, :, -1, :])                  # (R, P, 3)
    cs = (local + before.unsqueeze(2)).view(R, W, 3)
    n, s1, s2 = cs[..., 0], cs[..., 1], cs[..., 2]

    nc = n.clamp(min=1.0)
    m = s1 / nc
    var = (s2 / nc - m * m).clamp(min=0.0)
    has = n > 0
    mu = torch.where(has, m + shift, torch.zeros_like(m)).float()
    sd = var.sqrt().float().clamp(min=std_floor)
    return mu, sd, n.float()


ROLE_TARGET, ROLE_PAST, ROLE_KNOWN = 0, 1, 2     # variate roles inside a group


class LaneGroupAttention(nn.Module):
    """Attention ACROSS the variates of a group at the SAME patch position.

    Rows come in blocks of `lanes` consecutive rows that share one layout, so the variates of
    a group occupy the same patch columns in different rows. x (R, P, D) is viewed as
    (R/lanes blocks, P, lanes, D) and attention runs over the `lanes` axis only; `gmask`
    (blocks*P, 1, lanes, lanes) lets a token attend exactly to the tokens with its group id
    (itself included, so a single series is a group of one). No positional encoding: a group
    is a set. Cost is O(R * P * lanes) -- negligible next to time attention.

    Causality is preserved: a token at patch p only receives tokens at patch p, each of which
    has only seen its own series up to p. Initialised like any other layer: in a controlled
    test a zero-initialised output projection delayed the first use of covariates from ~800 to
    ~2300 steps. zero_group_layers() zeroes it for a warm start from a univariate checkpoint."""

    def __init__(self, d_model, d_kv, n_heads, dropout, eps):
        super().__init__()
        self.qkv = FusedQKV(d_model, d_kv, n_heads)
        self.out_proj = nn.Linear(n_heads * d_kv, d_model, bias=False)
        self.norm_w = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.dropout_p = dropout

    def forward(self, x, lanes, gmask):
        R, P, D = x.shape
        Bk = R // lanes
        h = rms_norm(x, self.norm_w, self.eps)
        h = h.view(Bk, lanes, P, D).transpose(1, 2).reshape(Bk * P, lanes, D)
        q, k, v = self.qkv(h)
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=gmask,
                                           dropout_p=self.dropout_p if self.training else 0.0)
        o = o.transpose(1, 2).reshape(Bk, P, lanes, -1).transpose(1, 2).reshape(R, P, -1)
        return x + self.out_proj(o)


class DecoderBlock(nn.Module):
    """time attention -> (group attention) -> feed-forward. Parameter names match EncoderBlock,
    so univariate checkpoints load into group_attn=False models unchanged, and into
    group_attn=True models with strict=False (CONFIG['init_from'])."""

    def __init__(self, d_model, d_kv, n_heads, d_ff, dropout, eps, rope, group_attn):
        super().__init__()
        self.time = TimeSelfAttention(d_model, d_kv, n_heads, dropout, eps, rope)
        self.group = LaneGroupAttention(d_model, d_kv, n_heads, dropout, eps) if group_attn else None
        self.ffn = FeedForward(d_model, d_ff, dropout, eps)

    def forward(self, x, attn_mask=None, is_causal=False, lanes=1, gmask=None):
        x = self.time(x, attn_mask, is_causal)
        if self.group is not None:
            x = self.group(x, lanes, gmask)
        return self.ffn(x)


class CausalPatchDecoder(nn.Module):
    is_decoder = True

    def __init__(self, patch_len=32, d_model=512, d_kv=64, n_heads=8, d_ff=2048, n_layers=6,
                 dropout=0.1, eps=1e-6, quantile_levels=QUANTILE_LEVELS, arcsinh=True,
                 max_pass_patches=32, group_attn=False, group_lanes=8, horizon_loss_share=None):
        super().__init__()
        self.patch_len = patch_len
        self.horizon_loss_share = horizon_loss_share
        self.group_attn = group_attn
        self.group_lanes = group_lanes       # largest group seen in training (eval caps at this)
        self.pred_len = patch_len            # native output: one patch per position
        self.quantile_levels = quantile_levels
        self.n_q = len(quantile_levels)
        self.mask_channel = True
        self.arcsinh = arcsinh
        self.eps = eps
        self.max_pass_patches = max_pass_patches
        self.rope = SharedRoPE(d_kv)
        # residual patch embedding (Chronos-2 / t0 style): values + validity mask channels
        self.emb_w1 = nn.Linear(2 * patch_len, d_ff)
        self.emb_w2 = nn.Linear(d_ff, d_model)
        self.emb_skip = nn.Linear(2 * patch_len, d_model)
        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, d_kv, n_heads, d_ff, dropout, eps, self.rope, group_attn)
            for _ in range(n_layers)])
        if group_attn:      # what a variate IS in its group: target / past covariate / known covariate
            self.role_emb = nn.Embedding(3, d_model)
            nn.init.normal_(self.role_emb.weight, std=0.02)
            # LOOKAHEAD for known covariates: their token at patch p also embeds patch p+1. The
            # token at p is the one that forecasts the target's patch p+1, so through group
            # attention the target sees the covariate for the patch it is forecasting, AND the
            # covariate for its own current patch (which is what lets it learn, in context,
            # how the target responds to the covariate).
            self.look_w1 = nn.Linear(2 * patch_len, d_ff)
            self.look_w2 = nn.Linear(d_ff, d_model)
            self.look_skip = nn.Linear(2 * patch_len, d_model)
        self.final_norm_w = nn.Parameter(torch.ones(d_model))
        self.head = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(),
                                  nn.Linear(d_ff, patch_len * self.n_q))

    def _fwd_tf(self, z):
        return torch.asinh(z) if self.arcsinh else z

    def _inv_tf(self, o):
        return torch.sinh(o.clamp(-20.0, 20.0)) if self.arcsinh else o

    def _core(self, x, obs, seg_pt, seg_start_pt, n_slots, attn_mask=None, is_causal=False,
              lanes=1, gid=None, role=None):
        """Returns quantiles for the NEXT patch at every position, in the normalised space of
        the stats frozen at that patch's end: q (R, P, pl, nq); and mu_e, sd_e, cnt_e (R, P).
        Group models: rows are blocks of `lanes` rows sharing a layout; gid (R, P) = group id
        of each token (tokens attend across lanes to equal ids at the same patch), role (R, P)
        = ROLE_* of each token. gid=None -> every row is its own group (univariate)."""
        R, W = x.shape
        pl = self.patch_len
        P = W // pl
        mu, sd, cnt = _causal_stats(x, obs, seg_pt, seg_start_pt, n_slots, patch_len=pl)
        z = torch.where(obs, self._fwd_tf((x.float() - mu) / sd), torch.zeros((), device=x.device))
        u = torch.cat([z.view(R, P, pl), obs.view(R, P, pl).to(z.dtype)], dim=-1).to(x.dtype)
        h = self.emb_w2(F.relu(self.emb_w1(u))) + self.emb_skip(u)
        gmask = None
        if self.group_attn:
            if role is None:
                h = h + self.role_emb.weight[ROLE_TARGET].to(h.dtype)
            else:
                segp = seg_pt[:, ::pl]
                same = F.pad(segp[:, 1:] == segp[:, :-1], (0, 1))            # next patch = same series
                un = torch.cat([u[:, 1:], torch.zeros_like(u[:, :1])], 1) * same.unsqueeze(-1).to(u.dtype)
                look = self.look_w2(F.relu(self.look_w1(un))) + self.look_skip(un)
                known = (role == ROLE_KNOWN).unsqueeze(-1).to(h.dtype)
                h = h + self.role_emb(role).to(h.dtype) + look * known
            if gid is None:
                lanes = 1
                gmask = torch.ones(R * P, 1, 1, 1, dtype=torch.bool, device=x.device)
            else:
                g = gid.view(R // lanes, lanes, P).transpose(1, 2).reshape((R // lanes) * P, lanes)
                gmask = (g.unsqueeze(2) == g.unsqueeze(1)).unsqueeze(1)
        for blk in self.blocks:
            h = blk(h, attn_mask=attn_mask, is_causal=is_causal, lanes=lanes, gmask=gmask)
        h = rms_norm(h, self.final_norm_w, self.eps)
        q = _monotone_quantiles(self.head(h).float().view(R, P, pl, self.n_q))
        e = pl - 1
        return q, mu[:, e::pl], sd[:, e::pl], cnt[:, e::pl]

    def _packed_geometry(self, seg_patch, n_slots):
        R, P = seg_patch.shape
        pl = self.patch_len
        pidx = torch.arange(P, device=seg_patch.device).expand(R, P)
        start_patch = torch.full((n_slots,), P, device=seg_patch.device, dtype=torch.long).scatter_reduce(
            0, seg_patch.reshape(-1), pidx.reshape(-1), reduce="amin", include_self=True)
        seg_pt = seg_patch.repeat_interleave(pl, dim=1)
        seg_start_pt = start_patch[seg_pt].clamp(max=P - 1) * pl
        causal = torch.ones(P, P, dtype=torch.bool, device=seg_patch.device).tril()
        attn = ((seg_patch.unsqueeze(2) == seg_patch.unsqueeze(1)) & causal).unsqueeze(1)
        return seg_pt, seg_start_pt, attn

    def forward(self, x, target=None, mask=None, seg_patch=None, slot_valid=None, blank=None,
                lanes=1, gid=None, role=None, slot_loss=None):
        """PACKED (training): x, mask (R, W); seg_patch (R, P); slot_valid (S,); blank (R, W)
        = contiguous-patch-masked input points (hidden from the model, still used as targets);
        lanes / gid (R, P) / role (R, P) = group structure; slot_loss (S,) = series that carry
        loss (known covariates do not: their future is given, not forecast).
        UNPACKED: x, mask (B, L) one left-padded series per row -> next-patch quantiles in
        data scale at every position: quantile_preds_all (B, P, pl, nq). With lanes/gid/role the
        B rows are blocks of `lanes` aligned variates (a group run on its own)."""
        if mask is None:
            mask = torch.ones_like(x, dtype=torch.bool)
        if seg_patch is None:
            B, L = x.shape
            seg_pt = torch.arange(B, device=x.device).unsqueeze(1).expand(B, L)
            q, mu_e, sd_e, _ = self._core(x, mask, seg_pt, torch.zeros_like(seg_pt), B, is_causal=True,
                                          lanes=lanes, gid=gid, role=role)
            return {"quantile_preds_all": self._inv_tf(q) * sd_e[..., None, None] + mu_e[..., None, None]}

        R, W = x.shape
        pl = self.patch_len
        P = W // pl
        S = slot_valid.shape[0]
        obs_in = mask & ~blank if blank is not None else mask
        seg_pt, seg_start_pt, attn = self._packed_geometry(seg_patch, S)
        q, mu_e, sd_e, cnt_e = self._core(x, obs_in, seg_pt, seg_start_pt, S, attn_mask=attn,
                                          lanes=lanes, gid=gid, role=role)

        # dense next-patch loss: patch p predicts patch p+1 of the SAME series, whenever
        # patch p's input had >= 1 observed point. Targets use the TRUE mask (blanked
        # points are hidden from the input but are still valid targets).
        y = x.float().view(R, P, pl)[:, 1:]
        ym = mask.view(R, P, pl)[:, 1:]
        ok_b = (seg_patch[:, 1:] == seg_patch[:, :-1]) & (cnt_e[:, :-1] > 0)
        tn = self._fwd_tf((y - mu_e[:, :-1, None]) / sd_e[:, :-1, None])
        qq = q[:, :-1]
        lv = torch.tensor(self.quantile_levels, device=x.device, dtype=qq.dtype)
        err = tn.unsqueeze(-1) - qq
        pin = torch.maximum(lv * err, (lv - 1) * err).mean(-1)             # (R, P-1, pl)
        w = (ym & ok_b.unsqueeze(-1)).float()
        seg_b = seg_patch[:, :-1].reshape(-1)
        pw, ww = (pin * w).sum(-1), w.sum(-1)                              # (R, P-1)
        zS = lambda: torch.zeros(S, device=x.device)
        if blank is not None:       # split by what the forecast was made FROM
            bl = blank.view(R, P, pl).all(-1)[:, :-1].float()              # input patch hidden
        else:
            bl = torch.zeros_like(ww)
        num_m = zS().index_add_(0, seg_b, (pw * bl).reshape(-1))           # multi-patch forecasts
        den_m = zS().index_add_(0, seg_b, (ww * bl).reshape(-1))
        num_o = zS().index_add_(0, seg_b, (pw * (1 - bl)).reshape(-1))     # one-patch forecasts
        den_o = zS().index_add_(0, seg_b, (ww * (1 - bl)).reshape(-1))
        per_o, per_m = num_o / den_o.clamp(min=1.0), num_m / den_m.clamp(min=1.0)
        has_o, has_m = den_o > 0, den_m > 0
        den = den_o + den_m
        if self.horizon_loss_share is None:
            per = (num_o + num_m) / den.clamp(min=1.0)                     # v4: plain mean
        else:
            a = float(self.horizon_loss_share)
            per = torch.where(has_o & has_m, (1 - a) * per_o + a * per_m, torch.where(has_m, per_m, per_o))
        has = (den > 0) & slot_valid
        if slot_loss is not None:
            has = has & slot_loss
        hf = has.float()
        nan = torch.full_like(per, float("nan"))
        return {"loss": (per * hf).sum() / hf.sum().clamp(min=1.0),
                "per_series_loss": torch.where(has, per, nan),
                "_per_one": torch.where(has & has_o, per_o, nan),
                "_per_multi": torch.where(has & has_m, per_m, nan),
                "_q": q, "_mu_e": mu_e, "_sd_e": sd_e}

    @torch.no_grad()
    def predict_horizon(self, ctx, mask, horizon):
        """ctx, mask (B, L): left-padded, L a multiple of patch_len. Appends ceil(horizon/pl)
        blank patches and reads every horizon patch from ONE forward pass. (B, horizon, nq)."""
        B, L = ctx.shape
        pl = self.patch_len
        hp = -(-horizon // pl)
        xx = torch.cat([ctx, torch.zeros(B, hp * pl, device=ctx.device, dtype=ctx.dtype)], 1)
        mm = torch.cat([mask, torch.zeros(B, hp * pl, device=ctx.device, dtype=torch.bool)], 1)
        allq = self.forward(xx, mask=mm)["quantile_preds_all"]            # (B, P, pl, nq)
        pc = L // pl
        return allq[:, pc - 1:pc - 1 + hp].reshape(B, hp * pl, self.n_q)[:, :horizon]


def zero_group_layers(model):
    """Make a group model behave EXACTLY like the univariate decoder it shares weights with
    (used for a warm start: the group pathway then grows from zero)."""
    for blk in model.blocks:
        if blk.group is not None:
            nn.init.zeros_(blk.group.out_proj.weight)
    if getattr(model, "group_attn", False):
        nn.init.zeros_(model.role_emb.weight)


def build_model(cfg):
    if cfg.get("decoder", "pool") == "causal":
        return CausalPatchDecoder(
            patch_len=cfg["patch_len"], d_model=cfg["d_model"], d_kv=cfg["d_kv"],
            n_heads=cfg["n_heads"], d_ff=cfg["d_ff"], n_layers=cfg["n_layers"],
            dropout=cfg["dropout"], arcsinh=cfg.get("arcsinh", True),
            max_pass_patches=cfg.get("max_pass_patches", 32),
            group_attn=bool(cfg.get("group_attn", False)), group_lanes=int(cfg.get("group_lanes", 8)),
            horizon_loss_share=cfg.get("horizon_loss_share", None))
    return Chronos2Style(
        patch_len=cfg["patch_len"], pred_len=cfg["pred_len"], d_model=cfg["d_model"],
        d_kv=cfg["d_kv"], n_heads=cfg["n_heads"], d_ff=cfg["d_ff"],
        n_layers=cfg["n_layers"], dropout=cfg["dropout"],
        skip_group_attn=cfg["skip_group_attn"],
        mask_channel=cfg.get("mask_channel", False), arcsinh=cfg.get("arcsinh", False))


# ============ 3. SYNTHETIC GENERATOR ============

_MAX_SEG = 48          # max change-point segments per series (bounds the gather table)
_CLAMP = 1e9           # final magnitude clamp, keeps everything comfortably finite in fp32
_EXP_CLAMP = 8.0       # clamp exponent args before exp() so nothing overflows to inf

# Periods chosen to match real sampling frequencies actually present in TS corpora:
# 4=quarterly-in-year, 7=day-of-week, 12=month-of-year, 24=hour-of-day,
# 48=half-hourly-day, 96=15min-day, 144=10min-day, 168=hour-of-week,
# 288=5min-day, 336=half-hour-week, 365=day-of-year, 720=hour-of-month
PERIOD_BANK = [4, 7, 12, 24, 48, 96, 144, 168, 288, 336, 365, 720]
# clean2 "periods": + 13 / 26 / 52 (quarter / half-year / year in WEEKLY data -- absent before,
# and weekly tasks were among the worst) and weekly cycles at 15 / 10 / 5-minute resolution
# (672 / 1008 / 2016 -- redset, LOOP_SEATTLE, bizitobs)
PERIOD_BANK_EXT = [4, 7, 12, 13, 24, 26, 48, 52, 96, 144, 168, 288, 336, 365, 672, 720,
                   1008, 2016]

PERIOD_BANK_NEW = [p for p in PERIOD_BANK_EXT if p not in PERIOD_BANK]

# set by generate_batch() from the active profile (single-threaded generation)
_PERIOD_BANK_ACTIVE = PERIOD_BANK
_PERIOD_EXT_SHARE = None      # None: uniform over the active bank
_TV_SEASON = False
_ONE_OFF_P = 0.0              # share of repeating series that get a one-off anomaly (round 3)


# ============================ low-level primitives ============================

def _u(B, n, device, lo, hi):
    """Uniform (B, n)."""
    return torch.empty(B, n, device=device).uniform_(lo, hi)


def _bern(B, device, p):
    """Per-row Bernoulli gate -> (B, 1) bool."""
    return torch.rand(B, 1, device=device) < p


def _pick(B, values, device):
    """Sample one scalar per row from `values` -> (B, 1) float."""
    v = torch.tensor(values, device=device, dtype=torch.float32)
    idx = torch.randint(0, len(values), (B, 1), device=device)
    return v[idx.squeeze(-1)].unsqueeze(-1)


def _pick_period(B, L, device, min_cycles=3.0):
    """Sample a seasonal period that can actually MANIFEST in a window of length L.

    A period-720 component inside a 1024-point window completes 1.4 cycles -- that reads
    as a trend, not as seasonality, and no forecaster can learn a cycle it never sees
    repeat. Restricting the bank to periods fitting at least `min_cycles` times is what
    makes the seasonal structure learnable at the window length actually being trained on.
    (Measured effect: seasonal-strength coverage went from 18% to the 40-60% range that
    matches real corpora, purely from this filter.)
    """
    usable = [p for p in _PERIOD_BANK_ACTIVE if p * min_cycles <= L]
    if not usable:
        usable = [max(2, int(L // max(min_cycles, 1)))]
    if _PERIOD_EXT_SHARE is None:
        return _pick(B, usable, device)
    # protected share: the original periods keep (1 - share) of the mass, the new ones get share
    base = [p for p in usable if p in PERIOD_BANK] or usable
    new = [p for p in usable if p in PERIOD_BANK_NEW]
    if not new:
        return _pick(B, base, device)
    return torch.where(_bern(B, device, _PERIOD_EXT_SHARE), _pick(B, new, device), _pick(B, base, device))


def _fft_filter(eps, h):
    """Circular convolution of eps (B, N) with impulse response h (B, N), via FFT.

    This is the workhorse: every linear-Gaussian process in this file (AR, ARMA, OU,
    exponentially-smoothed noise) is generated by choosing an impulse response and
    calling this, instead of running a sequential recursion.
    """
    n = eps.shape[1]
    return torch.fft.irfft(torch.fft.rfft(eps, dim=1) * torch.fft.rfft(h, dim=1), n=n, dim=1)


def _ar1_filter(B, L, device, phi, noise=None):
    """Stationary AR(1) with per-row coefficient phi (B, 1). Returns (B, L).

    Generated at double length and the SECOND half returned, so the burn-in transient
    from the zero initial condition is discarded and the output is properly stationary.
    """
    N = 2 * L
    k = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0)
    h = phi ** k                      # impulse response of 1/(1 - phi B)
    eps = torch.randn(B, N, device=device) if noise is None else noise
    return _fft_filter(eps, h)[:, L:]


def _power_law_noise(B, L, device, beta):
    """Spectral synthesis of noise with PSD ~ |f|^-beta, per-row beta (B, 1) -> (B, L).

    One parameter spans a whole spectrum of real-world behaviours:
      beta=0   white noise
      beta~1   pink / 1-f noise (network traffic, many natural processes)
      beta~2   brown noise == random walk == the unit-root/I(1) case (asset prices)
      0<beta<2 fractional Gaussian noise / long memory, Hurst=(beta+1)/2
               (river flow -- saugeen -- and self-similar network traffic)
    """
    N = 2 * L
    f = torch.fft.rfftfreq(N, device=device).unsqueeze(0)        # (1, N//2+1)
    amp = torch.where(f > 0, f.clamp(min=1e-8) ** (-beta / 2.0), torch.zeros_like(f))
    nf = f.shape[1]
    z = torch.complex(torch.randn(B, nf, device=device), torch.randn(B, nf, device=device))
    x = torch.fft.irfft(amp.to(torch.float32) * z, n=N, dim=1)
    return x[:, L:]


def _segment_ids(B, L, device, lo_breaks=1.0, hi_breaks=8.0):
    """Per-row change-point segment index (B, L) long, values in [0, _MAX_SEG).

    Bernoulli jump indicators -> cumsum. This is how every piecewise / regime-switching
    structure in this file is built without a sequential Markov chain.
    """
    rate = _u(B, 1, device, lo_breaks, hi_breaks) / max(L, 1)
    jumps = (torch.rand(B, L, device=device) < rate).float()
    jumps[:, 0] = 0.0
    return jumps.cumsum(1).long().clamp_(max=_MAX_SEG - 1)


def _piecewise_constant(B, L, device, lo_breaks=1.0, hi_breaks=8.0, heavy=False):
    """Piecewise-constant process (B, L): random level per change-point segment."""
    seg = _segment_ids(B, L, device, lo_breaks, hi_breaks)
    vals = torch.randn(B, _MAX_SEG, device=device)
    if heavy:   # occasional large regime jumps rather than all-similar levels
        vals = vals * torch.exp(_u(B, _MAX_SEG, device, -0.5, 1.5))
    return torch.gather(vals, 1, seg)


def _forward_fill(x, observed):
    """Vectorized forward-fill: carry the last observed value across gaps.

    Implemented with cummax over the observed-index, which is the standard trick for
    doing this without a sequential scan.
    """
    B, L = x.shape
    idx = torch.arange(L, device=x.device).expand(B, L)
    vi = torch.where(observed, idx, torch.zeros_like(idx))
    last = torch.cummax(vi, dim=1).values
    return torch.gather(x, 1, last)


def _smooth_noise(B, L, device, lo_phi=0.90, hi_phi=0.999):
    """Slowly-varying latent driver (heavily smoothed AR(1)), unit-ish scale.
    Used for weather-style multiplicative modulation and stochastic volatility."""
    phi = _u(B, 1, device, lo_phi, hi_phi)
    x = _ar1_filter(B, L, device, phi)
    return x / (x.std(dim=1, keepdim=True) + 1e-6)


def _standardize(x):
    return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-6)


# ============================ base dynamical families ============================
# Each: fn(B, L, device) -> (B, L) float32 latent process.

def f_gp_kernel(B, L, device):
    """Composed-kernel GP via circulant embedding -- the KernelSynth family, kept because
    it genuinely covers smooth stationary structure well, with an extended kernel bank
    (adds Matern-1/2 = OU/exponential, and a locally-linear kernel) and up to 3 kernels
    combined by random addition or multiplication."""
    d = torch.linspace(0, 1, L, device=device).unsqueeze(0)      # lag grid (1, L)

    def layer():
        kind = torch.randint(0, 4, (B, 1), device=device)
        ls = _u(B, 1, device, 0.02, 0.5)
        a = _u(B, 1, device, 0.1, 2.0)
        per = (_pick_period(B, L, device) / L).clamp(min=1e-4)
        rbf = torch.exp(-0.5 * d ** 2 / ls ** 2)                              # smooth
        es = torch.exp(-2.0 * torch.sin(math.pi * d / per) ** 2)              # periodic
        rq = (1 + d ** 2 / (2 * a * ls ** 2)) ** (-a)                         # rational quad
        matern12 = torch.exp(-d / ls)                                          # rough / OU
        return torch.where(kind == 0, rbf,
               torch.where(kind == 1, es,
               torch.where(kind == 2, rq, matern12)))

    nk = torch.randint(1, 4, (B, 1), device=device)
    row = layer()
    for dep in range(1, 3):
        nxt = layer()
        add = torch.rand(B, 1, device=device) < 0.5
        row = torch.where(nk > dep, torch.where(add, row + nxt, row * nxt), row)

    c = torch.cat([row, torch.flip(row[:, 1:-1], dims=[1])], dim=1)
    m = c.shape[1]
    lam = torch.clamp(torch.fft.fft(c, dim=1).real, min=0.0)
    z = torch.complex(torch.randn(B, m, device=device), torch.randn(B, m, device=device))
    y = torch.fft.fft(torch.sqrt(lam / m).to(z.dtype) * z, dim=1)[:, :L].real
    return torch.nan_to_num(_standardize(y))


def f_power_law(B, L, device):
    """Long-memory / fractional / pink / brown noise. Covers unit-root behaviour."""
    return _power_law_noise(B, L, device, _u(B, 1, device, 0.2, 2.4))


def f_random_walk(B, L, device):
    """Explicit random walk with optional drift -- the canonical financial price process."""
    drift = _u(B, 1, device, -0.02, 0.02)
    steps = torch.randn(B, L, device=device) + drift
    return steps.cumsum(dim=1)


def f_ar_process(B, L, device):
    """AR(1) across the full stability range, INCLUDING negative phi (oscillatory /
    mean-crossing behaviour, which positive-only AR never produces)."""
    phi = torch.where(torch.rand(B, 1, device=device) < 0.25,
                      _u(B, 1, device, -0.95, -0.2),
                      _u(B, 1, device, 0.2, 0.995))
    return _ar1_filter(B, L, device, phi)


def f_mean_revert(B, L, device):
    """Ornstein-Uhlenbeck around a non-zero level: slow mean reversion, the classic
    spread / inventory / temperature-anomaly process."""
    phi = _u(B, 1, device, 0.95, 0.9995)
    level = _u(B, 1, device, -3.0, 3.0)
    return _ar1_filter(B, L, device, phi) + level


def f_stochastic_vol(B, L, device):
    """Volatility clustering + fat tails: returns whose log-variance follows a slow AR(1),
    then cumulated to a price-like path. This is the GARCH stand-in described in the module
    docstring, and it is the single most important family for your Nifty end-goal -- calm
    periods punctuated by turbulent ones, with return kurtosis well above Gaussian."""
    logvol = _smooth_noise(B, L, device, 0.97, 0.9995) * _u(B, 1, device, 0.3, 1.2)
    vol = torch.exp(logvol.clamp(-_EXP_CLAMP, _EXP_CLAMP))
    rets = torch.randn(B, L, device=device) * vol
    # ~half stay as returns (stationary, fat-tailed), ~half cumulate into a price path
    as_price = _bern(B, device, 0.5)
    return torch.where(as_price, rets.cumsum(dim=1), rets)


def f_regime_switch(B, L, device):
    """Markov-switching between two AR regimes with different persistence and scale --
    e.g. a machine in normal vs degraded mode, a market in calm vs stressed mode."""
    a = _ar1_filter(B, L, device, _u(B, 1, device, 0.3, 0.9))
    b = _ar1_filter(B, L, device, _u(B, 1, device, 0.9, 0.999)) * _u(B, 1, device, 1.5, 4.0)
    seg = _segment_ids(B, L, device, 1.0, 6.0)
    in_b = (seg % 2 == 1).float()
    return a * (1 - in_b) + b * in_b


def f_level_shift(B, L, device):
    """Piecewise-constant levels with mild within-segment noise: sensor recalibration,
    contract changes, capacity steps, deployment-driven jumps in cloud metrics."""
    base = _piecewise_constant(B, L, device, 1.0, 6.0, heavy=True)
    return base + torch.randn(B, L, device=device) * _u(B, 1, device, 0.02, 0.3)


def f_piecewise_trend(B, L, device):
    """Trend whose SLOPE changes at breakpoints (cumsum of piecewise-constant slopes).
    Growth that accelerates, stalls, then reverses -- extremely common in economic series
    and the thing pure-linear-trend models get badly wrong."""
    slopes = _piecewise_constant(B, L, device, 1.0, 5.0) * _u(B, 1, device, 0.01, 0.2)
    return slopes.cumsum(dim=1) + torch.randn(B, L, device=device) * 0.1


def f_logistic_growth(B, L, device):
    """S-curve saturation: epidemic waves, technology adoption, market penetration.
    Includes multi-wave variants (a second, later S-curve superimposed)."""
    t = torch.linspace(0, 1, L, device=device).unsqueeze(0)
    mid = _u(B, 1, device, 0.15, 0.85)
    steep = _u(B, 1, device, 4.0, 30.0)
    s = torch.sigmoid(steep * (t - mid))
    second = _bern(B, device, 0.35)
    mid2 = _u(B, 1, device, 0.5, 0.95)
    s2 = torch.sigmoid(_u(B, 1, device, 6.0, 30.0) * (t - mid2)) * _u(B, 1, device, 0.3, 1.2)
    return torch.where(second, s + s2, s) + torch.randn(B, L, device=device) * 0.02


def f_damped_osc(B, L, device):
    """Decaying oscillation / impulse response of a second-order physical system --
    step responses in control systems, settling behaviour after a shock."""
    t = torch.linspace(0, 1, L, device=device).unsqueeze(0)
    decay = _u(B, 1, device, 0.5, 8.0)
    freq = _u(B, 1, device, 2.0, 40.0)
    phase = _u(B, 1, device, 0, 2 * math.pi)
    return torch.exp(-decay * t) * torch.cos(2 * math.pi * freq * t + phase)


def f_diurnal_envelope(B, L, device):
    """Solar-style generation: a hard-zero night mask (relu of a sinusoid raised to a
    random power) modulated by smooth multiplicative 'cloud cover' noise.

    The structural zeros are the point. No amount of GP sampling produces a series that
    is EXACTLY zero for a contiguous, periodic, deterministic fraction of every cycle,
    yet that is precisely what solar irradiance does -- and GIFT-Eval scores you on four
    solar configs.
    """
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    P = _pick_period(B, L, device, min_cycles=3.0)
    phase = _u(B, 1, device, 0, 2 * math.pi)
    env = torch.relu(torch.sin(2 * math.pi * t / P + phase)) ** _u(B, 1, device, 0.7, 2.5)
    cloud = torch.exp((_smooth_noise(B, L, device, 0.95, 0.999)
                       * _u(B, 1, device, 0.1, 0.8)).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    return env * cloud


def f_spike_train(B, L, device):
    """Sparse events convolved with an exponential decay kernel: incident-and-recovery,
    promotion-and-decay, viral spikes with an aftermath tail."""
    rate = _u(B, 1, device, 2.0, 40.0) / max(L, 1)
    events = (torch.rand(B, L, device=device) < rate).float()
    mags = torch.exp(_u(B, L, device, -0.5, 2.0))
    impulses = events * mags
    N = 2 * L
    k = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0)
    decay = torch.exp(-k / _u(B, 1, device, 2.0, 60.0))
    padded = torch.cat([impulses, torch.zeros(B, L, device=device)], dim=1)
    return _fft_filter(padded, decay)[:, :L]


def f_burst_onoff(B, L, device):
    """On/off bursty traffic with heavy-tailed level contrast -- the self-similar
    behaviour of web/cloud workloads (bitbrains, bizitobs in GIFT-Eval)."""
    seg = _segment_ids(B, L, device, 3.0, 20.0)
    on = (torch.rand(B, _MAX_SEG, device=device) < 0.45).float()
    lvl = torch.exp(_u(B, _MAX_SEG, device, -1.0, 2.5)) * on
    base = torch.gather(lvl, 1, seg)
    return base * torch.exp((torch.randn(B, L, device=device) * 0.3).clamp(-_EXP_CLAMP, _EXP_CLAMP))


def f_nonlinear_ar(B, L, device):
    """Threshold/saturating nonlinear dynamics: an AR process squashed through tanh, so
    large excursions compress. Real bounded physical processes behave this way and a
    linear-Gaussian model never generates it."""
    x = _ar1_filter(B, L, device, _u(B, 1, device, 0.7, 0.99))
    gain = _u(B, 1, device, 0.5, 3.0)
    return torch.tanh(x * gain / (x.std(dim=1, keepdim=True) + 1e-6))


def f_trend_season_classic(B, L, device):
    """The textbook additive trend + seasonal + noise decomposition. Deliberately
    included as its own family because a large fraction of Monash/M4 really does look
    like this, and the model should see plenty of the easy canonical case, not only
    exotic ones."""
    t = torch.linspace(0, 1, L, device=device).unsqueeze(0)
    trend = t * _u(B, 1, device, -4.0, 4.0)
    P = _pick_period(B, L, device)
    tt = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    seas = torch.sin(2 * math.pi * tt / P + _u(B, 1, device, 0, 2 * math.pi))
    seas = seas * _u(B, 1, device, 0.3, 3.0)
    return trend + seas + torch.randn(B, L, device=device) * _u(B, 1, device, 0.05, 0.5)


def f_flat_sparse(B, L, device):
    """Near-constant series that change rarely. Real corpora are full of these (low-volume
    hospital departments, slow-moving parts, idle sensors) and a model trained only on
    'interesting' series over-forecasts variation on them. Including them is what stops
    the model inventing structure that isn't there."""
    seg = _segment_ids(B, L, device, 0.5, 3.0)
    vals = torch.randn(B, _MAX_SEG, device=device) * 0.3
    return torch.gather(vals, 1, seg) + torch.randn(B, L, device=device) * 0.01


def f_intermittent_latent(B, L, device):
    """Latent intensity for intermittent demand: low baseline with occasional elevated
    demand periods. The actual zero-inflation and integerization happen in the support
    layer, so seasonality/trend can modulate the INTENSITY first (seasonal demand)."""
    base = _u(B, 1, device, -1.5, 0.3)
    bumps = _piecewise_constant(B, L, device, 2.0, 12.0) * _u(B, 1, device, 0.2, 1.0)
    return base + bumps


def f_count_latent(B, L, device):
    """Latent intensity for count data (births, admissions, deaths): smooth, positive-ish
    drive that the support layer will turn into Poisson/NegBin counts."""
    smooth = _smooth_noise(B, L, device, 0.97, 0.9995)
    return smooth * _u(B, 1, device, 0.3, 1.2) + _u(B, 1, device, 0.5, 3.0)


def f_multiscale_mix(B, L, device):
    """Superposition of a slow component and a fast component with independent dynamics --
    e.g. a slowly drifting baseline with fast measurement dynamics riding on top. Produces
    spectra with two distinct regimes, which single-kernel GPs cannot."""
    slow = _power_law_noise(B, L, device, _u(B, 1, device, 1.5, 2.4))
    fast = _ar1_filter(B, L, device, _u(B, 1, device, 0.1, 0.7))
    w = _u(B, 1, device, 0.2, 0.8)
    return w * _standardize(slow) + (1 - w) * fast


def f_short_macro_trend(B, L, device):
    """Near-monotonic, low-noise trend: annual/quarterly macro and demographic series (life
    expectancy, GDP, population, CO2, tourism arrivals). Targets fev-bench's worst cluster
    (world_life_expectancy 5.1x, us_consumption_1Y 2.9x, world_co2_emissions 2.6x): short,
    trend-dominated, no seasonality, and very rarely reversing more than once or twice.
    trend_season always adds a sinusoid and piecewise_trend's noise/slope ranges are tuned
    for long windows, so neither produces this shape. allow_add=False keeps the generic
    seasonality/trend/calendar layers off it. Noise is scaled to the trend's OWN range, so
    the series stays near-monotonic at any L -- which matters now that short L is common."""
    t = torch.linspace(0, 1, L, device=device).unsqueeze(0)
    seg = _segment_ids(B, L, device, lo_breaks=0.3, hi_breaks=2.5)
    dom_sign = torch.where(torch.rand(B, 1, device=device) < 0.5, 1.0, -1.0)
    raw_slopes = _u(B, _MAX_SEG, device, 0.4, 1.0) * dom_sign
    flip = torch.rand(B, _MAX_SEG, device=device) < 0.18
    raw_slopes = torch.where(flip, -raw_slopes * _u(B, _MAX_SEG, device, 0.15, 0.55), raw_slopes)
    trend = torch.gather(raw_slopes, 1, seg).cumsum(dim=1) / max(L, 1)
    curve = _u(B, 1, device, -0.5, 0.5) * (t - 0.5) ** 2 * dom_sign
    base = _u(B, 1, device, -2.0, 2.0) + trend + curve
    rng = (base.amax(1, keepdim=True) - base.amin(1, keepdim=True)).clamp(min=1e-3)
    return base + torch.randn(B, L, device=device) * rng * _u(B, 1, device, 0.01, 0.06)


# ---------------- Chronos-2-style "clean" base generators (used by corpus="clean") ----------------
# Chronos-2-Synth reached 45.9 fev-bench skill trained on synthetic data only, from textbook
# components: AR, ETS, trend+seasonality+irregular (TSI) and KernelSynth. These are those
# components, written in the same fully-batched O(n log n) style as everything above.

def _causal_conv(e, h):
    """y[t] = sum_{k<=t} h[t-k] e[k] with zero initial state (linear, not circular)."""
    B, L = e.shape
    n = 2 * L
    return torch.fft.irfft(torch.fft.rfft(e, n=n, dim=1) * torch.fft.rfft(h, n=n, dim=1), n=n, dim=1)[:, :L]


def _seasonal_cumsum(e, m_vals):
    """s[t] = sum of e[k] over k <= t with k = t (mod m), per-row period m -- the seasonal
    random walk of ETS (s_t = s_{t-m} + gamma*e_t). Loops only over DISTINCT periods (<= 12)."""
    out = torch.zeros_like(e)
    B, L = e.shape
    for m in m_vals.unique().tolist():
        rows = (m_vals == m).nonzero().squeeze(1)
        m = max(1, int(m))
        C = -(-L // m)
        v = F.pad(e[rows], (0, C * m - L)).view(len(rows), C, m).cumsum(1).view(len(rows), C * m)[:, :L]
        out[rows] = v
    return out


def _seasonal_drift(B, L, device, p=0.4):
    """clean2 tv_season: slowly drifting seasonal amplitude (multiplier) and phase (radians
    of the fundamental) for a fraction p of rows; identity for the rest."""
    on = _bern(B, device, p)
    ph = _smooth_noise(B, L, device, 0.99, 0.9995) * _u(B, 1, device, 0.05, 0.5)
    am = torch.exp((_smooth_noise(B, L, device, 0.98, 0.9995)
                    * _u(B, 1, device, 0.1, 0.5)).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    return (torch.where(on, am, torch.ones_like(am)), torch.where(on, ph, torch.zeros_like(ph)))


def _harmonic_profile(B, L, device, P, K=4):
    """Periodic shape with K harmonics and power-law amplitude decay (smooth humps to spikes).
    With tv_season on (clean2), 40% of rows get amplitude and phase drift."""
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    decay = _u(B, 1, device, 0.3, 1.6)
    tv = _TV_SEASON
    if tv:
        am, ph = _seasonal_drift(B, L, device)
    s = torch.zeros(B, L, device=device)
    for k in range(1, K + 1):
        amp = torch.full((1, 1), float(k), device=device) ** (-decay)
        ang = 2 * math.pi * k * t / P
        if tv:
            ang = ang + k * ph
        s = s + amp * (torch.randn(B, 1, device=device) * torch.cos(ang)
                       + torch.randn(B, 1, device=device) * torch.sin(ang))
    if tv:
        s = s * am
    return s / (s.std(1, keepdim=True) + 1e-6)


def f_ets(B, L, device):
    """Exponential smoothing state-space models (the ETS family): single-source-of-error
    level + {none, additive, damped} trend + optional seasonal random walk, additive or
    multiplicative. All components are linear filters of ONE innovation sequence e, so the
    whole thing is cumsums / an FFT convolution -- no sequential recursion."""
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    e = torch.randn(B, L, device=device) * _u(B, 1, device, 0.05, 0.6)
    ce = e.cumsum(1)
    y = _u(B, 1, device, 0.05, 0.8) * ce                                  # level: alpha * sum e
    kind = torch.randint(0, 3, (B, 1), device=device)                      # 0 none 1 additive 2 damped
    beta = _u(B, 1, device, 0.001, 0.1)
    b0 = torch.randn(B, 1, device=device) * _u(B, 1, device, 0.0, 0.05)
    phi = _u(B, 1, device, 0.8, 0.98)
    add_tr = b0 * t + beta * ce.cumsum(1)
    damp_b = _causal_conv(beta * e, phi ** t) + b0 * phi ** t              # b_t = phi b_{t-1} + beta e_t
    damp_tr = damp_b.cumsum(1)
    y = y + torch.where(kind == 1, add_tr, torch.where(kind == 2, damp_tr, torch.zeros_like(y)))
    m = _pick_period(B, L, device, min_cycles=2.0)                         # (B,1) float
    seas = (_harmonic_profile(B, L, device, m) * _u(B, 1, device, 0.5, 3.0) * e.std(1, keepdim=True)
            + _u(B, 1, device, 0.0, 0.3) * _seasonal_cumsum(e, m.squeeze(1).long()))
    y = y + torch.where(_bern(B, device, 0.65), seas, torch.zeros_like(seas)) + e
    mult = torch.exp((_standardize(y) * _u(B, 1, device, 0.1, 0.6)).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    return torch.where(_bern(B, device, 0.3), mult, y)


def f_ar_p(B, L, device):
    """Stationary AR(p), p <= 4, built from random roots: up to two complex-conjugate pairs
    (pseudo-cycles of random period and persistence -- the oscillatory behaviour AR(1) can
    never produce) and up to two real roots. Filtered in the frequency domain: X = E / A."""
    N = 2 * L
    a = torch.zeros(B, 5, device=device)
    a[:, 0] = 1.0

    def mul(a, c1, c2):                     # a(z) * (1 + c1 z^-1 + c2 z^-2)
        out = a.clone()
        out[:, 1:] = out[:, 1:] + c1 * a[:, :-1]
        out[:, 2:] = out[:, 2:] + c2 * a[:, :-2]
        return out

    n_pairs = torch.randint(0, 3, (B, 1), device=device)
    n_real = torch.where(n_pairs == 2, torch.zeros_like(n_pairs), torch.randint(0, 3, (B, 1), device=device))
    n_real = torch.where((n_pairs + n_real) == 0, torch.ones_like(n_real), n_real)
    for i in range(2):
        rho = _u(B, 1, device, 0.5, 0.995)
        th = math.pi * _u(B, 1, device, 0.0, 1.0) ** 2 + 1e-3          # bias toward slow cycles
        on = n_pairs > i
        a = mul(a, torch.where(on, -2 * rho * torch.cos(th), torch.zeros_like(rho)),
                torch.where(on, rho ** 2, torch.zeros_like(rho)))
    for i in range(2):
        r = torch.where(torch.rand(B, 1, device=device) < 0.2, _u(B, 1, device, -0.9, -0.2),
                        _u(B, 1, device, 0.1, 0.99))
        on = n_real > i
        a = mul(a, torch.where(on, -r, torch.zeros_like(r)), torch.zeros_like(r))
    # FLOAT64 IS REQUIRED HERE. With roots near 1 (a slow pair + two real roots ~0.99),
    # A(e^{iw}) near w=0 is ~3e-9 while float32 rounding on the coefficients is ~4e-7: the
    # computed A can be exactly 0 -> inf -> an all-NaN row -> torch.poisson device-side
    # assert on GPU (seen after ~50 steps of the first v2 run). In float64 the rounding is
    # ~1e-15, far below the smallest |A| these root ranges can produce (~1e-10).
    eps = torch.randn(B, N, device=device, dtype=torch.float64)
    x = torch.fft.irfft(torch.fft.rfft(eps, dim=1) / torch.fft.rfft(a.double(), n=N, dim=1), n=N, dim=1)[:, L:]
    return _standardize(x).float()


def f_tsi(B, L, device):
    """Trend + Seasonality + Irregularity (Chronos-2's TSI generator): one trend shape,
    1-2 seasonal components (the second nested, e.g. weekly-on-daily), AR(1) irregular."""
    t = torch.linspace(0, 1, L, device=device).unsqueeze(0)
    kind = torch.randint(0, 5, (B, 1), device=device)
    lin = t * _u(B, 1, device, -3.0, 3.0)
    expo = (torch.exp((t * _u(B, 1, device, 0.5, 3.0)).clamp(max=_EXP_CLAMP)) - 1.0) * _u(B, 1, device, -1.5, 1.5)
    logi = torch.sigmoid(_u(B, 1, device, 4.0, 25.0) * (t - _u(B, 1, device, 0.1, 0.9))) * _u(B, 1, device, -3.0, 3.0)
    piece = (_piecewise_constant(B, L, device, 1.0, 4.0) * _u(B, 1, device, 0.5, 3.0) / max(L, 1)).cumsum(1)
    trend = torch.where(kind == 1, lin, torch.where(kind == 2, expo, torch.where(kind == 3, logi,
            torch.where(kind == 4, piece, torch.zeros_like(lin)))))
    P1 = _pick_period(B, L, device, min_cycles=2.0)
    s = _harmonic_profile(B, L, device, P1) * _u(B, 1, device, 0.3, 2.0)
    P2 = P1 * _pick(B, [7.0, 12.0, 4.0], device)
    s2 = _harmonic_profile(B, L, device, P2) * _u(B, 1, device, 0.2, 1.0)
    s = s + torch.where(_bern(B, device, 0.4) & (P2 * 2.0 <= L), s2, torch.zeros_like(s2))
    s = torch.where(_bern(B, device, 0.8), s, torch.zeros_like(s))
    irr = _ar1_filter(B, L, device, _u(B, 1, device, 0.0, 0.8))
    irr = irr / (irr.std(1, keepdim=True) + 1e-6) * _u(B, 1, device, 0.05, 0.6)
    return trend + s + irr


def f_flat_hold(B, L, device):
    """Near-constant series that STAY near-constant: exact or almost-exact flat levels with at
    most one or two small steps. fev-bench's boomlet_285 scored 49.6x worse than naive
    because the model had only ever seen flat stretches followed by jumps (intermittent
    demand, level shifts) -- it had never learned that flat context -> flat future. Bypasses
    the noise/corruption layers (see PASSTHROUGH_FAMILIES) so the flatness survives."""
    seg = _segment_ids(B, L, device, 0.0, 1.5)
    steps = torch.randn(B, _MAX_SEG, device=device) * _u(B, 1, device, 0.02, 0.5)
    steps[:, 0] = 0.0
    lvl = torch.gather(steps.cumsum(1), 1, seg) + torch.randn(B, 1, device=device) * 2.0
    noise = torch.randn(B, L, device=device) * _u(B, 1, device, 0.0005, 0.01)
    return lvl + torch.where(_bern(B, device, 0.5), torch.zeros_like(noise), noise)


def f_cumulative(B, L, device):
    """Running totals (cumulative cases, odometers, year-to-date counters): monotone
    non-decreasing, integer or continuous increments with a slowly varying rate. Target of
    uk_covid_*/cumulative. Bypasses noise/corruption so monotonicity survives."""
    rate = torch.exp((_smooth_noise(B, L, device, 0.95, 0.999) * _u(B, 1, device, 0.2, 1.0)
                      + _u(B, 1, device, -2.0, 2.0)).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    rate = torch.nan_to_num(rate, nan=1.0, posinf=math.exp(_EXP_CLAMP), neginf=0.0)
    inc = torch.where(_bern(B, device, 0.5), torch.poisson(rate), rate * _u(B, 1, device, 0.5, 1.5))
    return inc.cumsum(1)


# ================= 3c. clean2 additions: generators for MEASURED fev-bench gaps =================
# Every family below exists because a specific group of fev-bench tasks was worse than the
# rest in the v2 eval (0.373). Each is switchable in CONFIG["gen_additions"].

_AGG_FINE_CAP = 1 << 17      # max fine-series length generated per row (memory bound)

# (aggregation factor k, short cycle of the FINE series, annual cycle of the FINE series, weight)
# k may be fractional: 30.436875 = mean days per month, so months come out 30 or 31 days
# long, exactly the irregular-block (trading-day) structure of real monthly totals.
_AGG_SCENARIOS = [
    (7.0,       7.0,  365.25, 0.38),   # daily   -> weekly    (annual cycle becomes 52.18)
    (30.436875, 7.0,  365.25, 0.30),   # daily   -> monthly   (annual cycle becomes 12)
    (3.0,       None, 12.0,   0.08),   # monthly -> quarterly (annual cycle becomes 4)
    (12.0,      None, 12.0,   0.07),   # monthly -> yearly    (annual cycle cancels: macro)
    (24.0,      24.0, 168.0,  0.12),   # hourly  -> daily     (weekly cycle becomes 7)
    (4.0,       96.0, 672.0,  0.05),   # 15-min  -> hourly
]


def _fine_calendar(n, N, device, p_short, p_annual):
    """A FINE-resolution positive calendar process (e.g. daily sales): log-level random walk
    + trend + short cycle (day-of-week) + annual cycle + sparse holiday / promo shocks +
    multiplicative noise; 35% of rows become fine-level counts (slow movers), which aggregate
    into small integers."""
    t = torch.arange(N, device=device, dtype=torch.float32).unsqueeze(0) / N
    logy = _standardize(torch.randn(n, N, device=device).cumsum(1)) * _u(n, 1, device, 0.05, 0.7)
    logy = logy + t * _u(n, 1, device, -1.5, 1.5)
    if p_short is not None:
        logy = logy + torch.where(_bern(n, device, 0.8),
                                  _harmonic_profile(n, N, device, p_short) * _u(n, 1, device, 0.0, 0.5),
                                  torch.zeros(n, N, device=device))
    logy = logy + torch.where(_bern(n, device, 0.75),
                              _harmonic_profile(n, N, device, p_annual) * _u(n, 1, device, 0.0, 0.6),
                              torch.zeros(n, N, device=device))
    hol = (torch.rand(n, N, device=device) < _u(n, 1, device, 0.0, 0.005)).float()
    logy = logy + hol * torch.randn(n, N, device=device) * _u(n, 1, device, 0.2, 1.0)
    logy = logy + torch.randn(n, N, device=device) * _u(n, 1, device, 0.02, 0.5)
    y = torch.exp(logy.clamp(-_EXP_CLAMP, _EXP_CLAMP))
    rate = torch.nan_to_num(y * torch.exp(_u(n, 1, device, -2.5, 3.0)), nan=1.0,
                            posinf=math.exp(12.0), neginf=0.0).clamp(0.0, math.exp(12.0))
    return torch.where(_bern(n, device, 0.35), torch.poisson(rate), y)


def _block_aggregate(y, k, L):
    """(n, N) fine -> (n, L) coarse: SUM over consecutive blocks of (possibly fractional)
    length k with a random phase (flows: sales, counts, energy), or, for 20% of rows, the
    LAST value of each block (stocks: prices, levels, end-of-period readings). Block 0 can
    be partial and is dropped."""
    n, N = y.shape
    dev = y.device
    phase = torch.rand(n, 1, device=dev, dtype=torch.float64) * k
    t = torch.arange(N, device=dev, dtype=torch.float64).unsqueeze(0)
    bid = torch.floor((t + phase) / k).long().clamp_(max=L + 2)
    s = torch.zeros(n, L + 3, device=dev, dtype=y.dtype).scatter_add_(1, bid, y)[:, 1:L + 1]
    j = torch.arange(1, L + 1, device=dev, dtype=torch.float64).unsqueeze(0)
    idx = (torch.ceil((j + 1) * k - phase) - 1).long().clamp_(0, N - 1)
    last = torch.gather(y, 1, idx)
    return torch.where(_bern(n, dev, 0.2), last, s)


def f_aggregated(B, L, device):
    """Low-frequency series made the way real ones are: by aggregating a finer process.
    Weekly / monthly data is smoother than daily, carries non-integer annual periods
    (52.18 weeks), irregular month lengths, and no day-of-week cycle (it sums out). Long
    windows only use small k (a 8192-point monthly series would need 250k fine points and
    does not exist in practice)."""
    ok = [s for s in _AGG_SCENARIOS if (L + 3) * s[0] <= _AGG_FINE_CAP]
    if not ok:
        ok = [min(_AGG_SCENARIOS, key=lambda s: s[0])]
    which = torch.multinomial(torch.tensor([s[3] for s in ok]), B, replacement=True)
    out = torch.zeros(B, L, device=device)
    for i, (k, p_short, p_annual, _) in enumerate(ok):
        rows = (which == i).nonzero().squeeze(1)
        if rows.numel() == 0:
            continue
        N = int(math.ceil((L + 3) * k))
        y = _fine_calendar(rows.numel(), N, device, p_short, p_annual)
        out[rows.to(device)] = _block_aggregate(y, k, L).float()
    return out


def f_bursty_load(B, L, device):
    """Query / job load (redset): a daily + weekly baseline intensity, bursts from a Hawkes
    CLUSTER process (burst-starting events ~ Poisson(baseline), each with a heavy-tailed
    lognormal mass spread by an exponential kernel -- the Poisson-cluster representation of
    self-excitation, so no sequential recursion), plus scheduled jobs firing at a fixed
    offset every Pj steps (ETL / cron) for 40% of rows. Half counts, half continuous load."""
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    ppd = _pick(B, [24.0, 96.0, 288.0], device)                       # 1H / 15T / 5T
    log0 = (_u(B, 1, device, -1.0, 3.0)
            + _harmonic_profile(B, L, device, ppd, K=6) * _u(B, 1, device, 0.3, 1.5)
            + _harmonic_profile(B, L, device, 7.0 * ppd) * _u(B, 1, device, 0.0, 0.5)
            + _smooth_noise(B, L, device, 0.99, 0.9995) * _u(B, 1, device, 0.0, 0.4))
    lam0 = torch.exp(log0.clamp(-_EXP_CLAMP, _EXP_CLAMP))
    starts = torch.poisson((lam0 * _u(B, 1, device, 0.01, 0.2)).clamp(0.0, 1e5))
    mass = starts * torch.exp((torch.randn(B, L, device=device) * _u(B, 1, device, 0.5, 1.5))
                              .clamp(-_EXP_CLAMP, _EXP_CLAMP)) * _u(B, 1, device, 1.0, 20.0)
    h = torch.exp(-t / _u(B, 1, device, 0.5, 12.0))
    burst = _causal_conv(mass, h / h.sum(1, keepdim=True)).clamp(min=0.0)
    Pj = _pick(B, [12.0, 24.0, 48.0, 96.0, 288.0], device)
    off = torch.floor(_u(B, 1, device, 0.0, 1.0) * Pj)
    fire = ((torch.remainder(t - off, Pj) == 0) & (torch.rand(B, L, device=device) < 0.9)).float()
    job = fire * _u(B, 1, device, 1.0, 10.0) * lam0.mean(1, keepdim=True)
    lam = lam0 + burst + torch.where(_bern(B, device, 0.4), job, torch.zeros_like(job))
    lam = torch.nan_to_num(lam, nan=1.0, posinf=1e6, neginf=0.0).clamp(0.0, 1e6)
    cont = lam * torch.exp(torch.randn(B, L, device=device) * _u(B, 1, device, 0.0, 0.2))
    return torch.where(_bern(B, device, 0.5), torch.poisson(lam), cont)


def f_daily_weekly(B, L, device):
    """Sub-daily series whose intraday SHAPE depends on the day type: weekday profile vs a
    different weekend / holiday profile (later, flatter peak), a small per-day-of-week level,
    an AR(1) day-to-day level, slow drift and an AR(1) irregular. A multiplicative weekend
    factor on one fixed profile -- what the seasonality and calendar layers produce -- can't
    represent this; real load, traffic and query data look like this."""
    ppd = _pick(B, [24.0, 48.0, 96.0, 144.0, 288.0], device)
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0) \
        + torch.floor(_u(B, 1, device, 0.0, 1.0) * 7.0 * ppd)
    day = torch.floor(t / ppd)
    tod = t - day * ppd
    dayl = day.long()
    dow = dayl % 7
    nd = int((L + 7 * 288) // 24) + 2                                # static bound on day index
    hol = torch.rand(B, nd, device=device) < _u(B, 1, device, 0.0, 0.03)
    wkend = torch.where(_bern(B, device, 0.2), dow == 6, dow >= 5) | torch.gather(hol, 1, dayl)
    decay = _u(B, 1, device, 0.3, 1.5)
    rho = _u(B, 1, device, 0.0, 0.85)
    orth = torch.sqrt(1.0 - rho * rho)
    Wp = torch.zeros(B, L, device=device)
    Ep = torch.zeros(B, L, device=device)
    for k in range(1, 7):
        a = torch.full((1, 1), float(k), device=device) ** (-decay)
        ang = 2 * math.pi * k * tod / ppd
        c1, c2 = torch.randn(B, 1, device=device) * a, torch.randn(B, 1, device=device) * a
        e1 = rho * c1 + orth * torch.randn(B, 1, device=device) * a
        e2 = rho * c2 + orth * torch.randn(B, 1, device=device) * a
        Wp = Wp + c1 * torch.cos(ang) + c2 * torch.sin(ang)
        Ep = Ep + e1 * torch.cos(ang) + e2 * torch.sin(ang)
    Wp = Wp / (Wp.std(1, keepdim=True) + 1e-6)
    Ep = Ep / (Ep.std(1, keepdim=True) + 1e-6) * _u(B, 1, device, 0.3, 1.2) + _u(B, 1, device, -1.5, 0.5)
    prof = torch.where(wkend, Ep, Wp)
    dow_lvl = torch.gather(torch.randn(B, 7, device=device) * _u(B, 1, device, 0.0, 0.3), 1, dow)
    dl = _ar1_filter(B, nd, device, _u(B, 1, device, 0.3, 0.97))
    dl = torch.gather(dl / (dl.std(1, keepdim=True) + 1e-6), 1, dayl) * _u(B, 1, device, 0.0, 0.8)
    slow = _smooth_noise(B, L, device, 0.995, 0.9999) * _u(B, 1, device, 0.0, 1.0)
    irr = _ar1_filter(B, L, device, _u(B, 1, device, 0.0, 0.9))
    irr = irr / (irr.std(1, keepdim=True) + 1e-6) * _u(B, 1, device, 0.05, 0.4)
    y = prof * _u(B, 1, device, 1.0, 3.0) + dow_lvl + dl + slow + irr
    mult = torch.exp((_standardize(y) * _u(B, 1, device, 0.2, 0.6)).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    return torch.where(_bern(B, device, 0.4), mult, y)


def f_epidemic(B, L, device):
    """Epidemic incidence: 1-4 asymmetric waves (sech^2 bumps, slower decay than rise, centres
    possibly outside the window so a series can start mid-wave), for 30% of rows a recurring
    seasonal wave train (flu: period 52 weekly / 365 daily), a small endemic baseline, and for
    half of rows a weekday reporting pattern (weekend dip). Counts or continuous rates."""
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    inc = torch.zeros(B, L, device=device)
    n_w = torch.randint(1, 5, (B, 1), device=device)
    for w in range(4):
        c = _u(B, 1, device, -0.1, 1.1) * L
        wl = _u(B, 1, device, 0.02, 0.2) * L + 1.0
        wr = wl * _u(B, 1, device, 0.8, 3.0)
        z = (t - c) / torch.where(t < c, wl, wr)
        bump = torch.exp(_u(B, 1, device, -1.5, 1.5)) * (1.0 - torch.tanh(z) ** 2)
        inc = inc + torch.where(n_w > w, bump, torch.zeros_like(bump))
    P = _pick(B, [52.0, 365.0], device)
    seas = torch.relu(torch.cos(2 * math.pi * t / P + _u(B, 1, device, 0.0, 2 * math.pi))) \
        ** _u(B, 1, device, 2.0, 8.0) * torch.exp(_u(B, 1, device, -1.0, 1.0))
    inc = inc + torch.where(_bern(B, device, 0.3), seas, torch.zeros_like(seas))
    peak = inc.amax(1, keepdim=True) + 1e-6
    inc = inc + _u(B, 1, device, 0.0, 0.05) * peak + 1e-3 * peak
    rate = inc / (inc.amax(1, keepdim=True) + 1e-12) * torch.exp(_u(B, 1, device, 0.5, 8.0))
    dow = torch.remainder(t + torch.randint(0, 7, (B, 1), device=device).float(), 7.0)
    rep = 1.0 - _u(B, 1, device, 0.2, 0.8) * (dow >= 5).float()
    rate = torch.where(_bern(B, device, 0.5), rate * rep, rate)
    rate = torch.nan_to_num(rate, nan=1.0, posinf=1e6, neginf=0.0).clamp(0.0, 1e6)
    cont = rate * torch.exp(torch.randn(B, L, device=device) * _u(B, 1, device, 0.0, 0.15))
    return torch.where(_bern(B, device, 0.6), torch.poisson(rate), cont)


def f_sarima(B, L, device):
    """Seasonal ARIMA (p,d,q)(P,D,Q)_m with p,q,P,Q <= 1 at a CALENDAR lag m (not random
    sparse lags): (1 - phi z)(1 - Phi z^m) x = (1 + theta z)(1 + Theta z^m) e, filtered in the
    frequency domain in float64 (|A| >= (1-|phi|)(1-|Phi|) > 0), then optional regular and
    seasonal integration (d, D in {0, 1}) -- the airline-model class of weekly/monthly data."""
    N = 2 * L
    m = _pick_period(B, L, device, min_cycles=3.0).long()
    f64 = torch.float64
    z = torch.zeros(B, 1, device=device)
    # mostly persistent, positive seasonal AR (visible seasonality); MA terms mostly negative
    # (the airline model (0,1,1)(0,1,1)_m has theta, Theta < 0)
    phi = torch.where(_bern(B, device, 0.75), _u(B, 1, device, -0.3, 0.95), z).double()
    th = torch.where(_bern(B, device, 0.5), _u(B, 1, device, -0.8, 0.4), z).double()
    Phi = torch.where(_bern(B, device, 0.8), _u(B, 1, device, 0.1, 0.95), _u(B, 1, device, -0.4, 0.1)).double()
    Th = torch.where(_bern(B, device, 0.5), _u(B, 1, device, -0.8, 0.3), z).double()
    a = torch.zeros(B, N, device=device, dtype=f64)
    a[:, 0] = 1.0
    a[:, 1:2] -= phi
    a.scatter_add_(1, m, -Phi)
    a.scatter_add_(1, m + 1, phi * Phi)
    b = torch.zeros(B, N, device=device, dtype=f64)
    b[:, 0] = 1.0
    b[:, 1:2] += th
    b.scatter_add_(1, m, Th)
    b.scatter_add_(1, m + 1, th * Th)
    e = torch.randn(B, N, device=device, dtype=f64)
    x = torch.fft.irfft(torch.fft.rfft(e, dim=1) * torch.fft.rfft(b, dim=1)
                        / torch.fft.rfft(a, dim=1), n=N, dim=1)[:, L:]
    x = _standardize(x).float()
    combo = torch.multinomial(torch.tensor([0.30, 0.20, 0.30, 0.20]), B, replacement=True).to(device)
    d = ((combo == 1) | (combo == 3)).unsqueeze(1)
    D = ((combo == 2) | (combo == 3)).unsqueeze(1)
    x = torch.where(D, _seasonal_cumsum(x, m.squeeze(1)), x)
    x = torch.where(d, x.cumsum(1), x)
    return _standardize(x)


def _L_breaks(x, device, p=0.2):
    """clean2 breaks: persistent structural breaks at a LOW rate (p of rows, 1-2 breaks).
    Every break is also an unforecastable target for the patches right after it under the
    dense next-patch loss, so a high rate would teach the model to widen every interval.
    What we want it to learn is 'a break happened in the context -> trust the recent part'.
    Kinds: 0 level shift, 1 slope change, 2 variance / amplitude change, 3 shock with V / U
    (full recovery, fast / slow) or L (partial, persistent) recovery, 4 level + amplitude."""
    B, L = x.shape
    if L < 8:
        return x
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    gate = _bern(B, device, p)
    sd = x.std(1, keepdim=True) + 1e-6
    m = x.mean(1, keepdim=True)
    one = torch.ones(B, 1, device=device)
    n_b = torch.where(_bern(B, device, 0.3), 2 * one, one)
    for j in range(2):
        active = gate & (n_b > j)
        tb = torch.floor(_u(B, 1, device, 0.05, 0.95) * L)
        after = (t >= tb).float()
        dt = (t - tb).clamp(min=0.0)
        kind = torch.randint(0, 5, (B, 1), device=device)
        sgn = torch.where(_bern(B, device, 0.5), one, -one)
        level = sgn * _u(B, 1, device, 0.5, 3.0) * sd * after
        slope = torch.randn(B, 1, device=device) * _u(B, 1, device, 0.5, 4.0) * sd * dt / L
        amp = (x - m) * (torch.exp(_u(B, 1, device, -1.2, 1.2)) - 1.0) * after
        tau = _u(B, 1, device, 0.005, 0.3) * L + 1.0
        keep = torch.where(_bern(B, device, 0.5), torch.zeros_like(one), _u(B, 1, device, 0.2, 1.0))
        ssgn = torch.where(_bern(B, device, 0.8), -one, one)
        shock = ssgn * _u(B, 1, device, 1.0, 5.0) * sd * after * ((1 - keep) * torch.exp(-dt / tau) + keep)
        delta = torch.where(kind == 0, level, torch.where(kind == 1, slope, torch.where(
            kind == 2, amp, torch.where(kind == 3, shock, level + amp))))
        x = x + torch.where(active, delta, torch.zeros_like(delta))
    return x


def f_repeat(B, L, device):
    """EXACTLY REPEATING patterns with ANY period and full-detail shapes.

    Why: a probe of the 700B model (noise-free sawtooth / square / spike / random-template series
    at 16 periods) showed it continues a pattern only when the period is one this generator
    uses -- the period bank 4/7/12/24/48/96/144/168/... -- and only when the shape is smooth
    (the seasonal layer builds shapes from ~4 harmonics). At period 10 it failed even on a sine;
    spikes and random templates scored ~0 at every period above 24. Chronos-2 scored 1.00 on
    all of it. Real machine metrics (fev-bench boomlet, redset) are exactly this kind of series.

    Here: period log-uniform in [2, min(L/4, 1024)] (integer for 85% of rows, so the repetition
    is exact), one of seven shapes, a quarter of rows are the sum of two patterns with different
    periods, optional slow amplitude drift / level trend, and half of the rows carry NO noise.
    The family is in PASSTHROUGH_FAMILIES: no seasonality / noise / corruption layer touches it."""
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    p_hi = float(max(2.0, min(L / 4.0, 1024.0)))
    M = int(p_hi) + 2                                   # width of the shape tables

    def one(n):
        P = torch.exp(_u(n, 1, device, math.log(2.0), math.log(p_hi)))
        kind = torch.randint(0, 7, (n, 1), device=device)
        integer = (kind >= 2) | _bern(n, device, 0.7)
        P = torch.where(integer, P.round().clamp(min=2.0), P)
        ph = _u(n, 1, device, 0.0, 1.0) * P
        ph = torch.where(integer, ph.floor(), ph)
        pos = torch.remainder(t + ph, P)                # position inside the period, in steps
        u = (pos / P).clamp(0.0, 1.0)                   # phase in [0, 1)
        # 0 sawtooth: (curved) ramp and instant reset, up or down
        saw = 2.0 * u.pow(torch.exp(_u(n, 1, device, -1.0, 1.0))) - 1.0
        saw = torch.where(_bern(n, device, 0.5), saw, -saw)
        # 1 square wave with a random duty cycle
        sq = torch.where(u < torch.minimum(_u(n, 1, device, 0.05, 0.6), 1.0 - 1.0 / P - 1e-4), 1.0, -1.0)
        # 2 spike train: 1-3 steps high once per period, flat otherwise
        width = torch.minimum(torch.randint(1, 4, (n, 1), device=device).float(), (P - 1.0).clamp(min=1.0))
        spike = (pos < width).float()
        # 3-6 table shapes: one value per step of the period, looked up exactly
        k = torch.arange(M, device=device, dtype=torch.float32).unsqueeze(0)
        rough = torch.randn(n, M, device=device)                                   # 3 any shape at all
        walk = torch.randn(n, M, device=device).cumsum(1)                          # 4 smooth, jump at the wrap
        end = walk.gather(1, (P - 1.0).long().clamp(0, M - 1))
        bridge = walk - k / (P - 1.0).clamp(min=1.0) * end                         #   ... or continuous at the wrap
        smooth = torch.where(_bern(n, device, 0.5), walk, bridge)
        jump = (torch.rand(n, M, device=device) < _u(n, 1, device, 1.0, 6.0) / P).long()
        jump[:, 1] = 1                                                             #   (always at least two levels)
        seg = jump.cumsum(1).clamp(max=63)
        stair = torch.randn(n, 64, device=device).gather(1, seg)                   # 5 staircase: a few held levels
        sparse = torch.randn(n, M, device=device).abs() * (torch.rand(n, M, device=device)
                                                           < _u(n, 1, device, 1.0, 4.0) / P).float()
        sparse[:, 0] = 1.0                                                         # 6 a few spikes of fixed heights
        tab = torch.where(kind == 3, rough, torch.where(kind == 4, smooth, torch.where(kind == 5, stair, sparse)))
        tmpl = tab.gather(1, pos.long().clamp(0, M - 1))
        y = torch.where(kind == 0, saw, torch.where(kind == 1, sq, torch.where(kind == 2, spike, tmpl)))
        return (y - y.mean(1, keepdim=True)) / (y.std(1, keepdim=True) + 1e-6)

    y = one(B)
    y = y + torch.where(_bern(B, device, 0.25), _u(B, 1, device, 0.3, 1.0), torch.zeros(B, 1, device=device)) * one(B)
    tt = t / max(L - 1, 1) - 0.5
    zero = torch.zeros(B, 1, device=device)
    y = y * (1.0 + torch.where(_bern(B, device, 0.25), _u(B, 1, device, -0.8, 0.8), zero) * tt)       # amplitude drift
    y = y + torch.where(_bern(B, device, 0.25), torch.randn(B, 1, device=device) * _u(B, 1, device, 0.5, 3.0), zero) * tt
    noise = torch.where(_bern(B, device, 0.5), zero, torch.exp(_u(B, 1, device, math.log(0.003), math.log(0.3))))
    return _one_off(y + noise * torch.randn(B, L, device=device), device)


def f_repeat_week(B, L, device):
    """DAY-TYPE REPEAT: a daily shape that repeats, with a DIFFERENT shape on some days of the week.

    Why: after the repeat fine-tune, the tasks furthest behind Chronos-2 on fev-bench (M_DENSE,
    entsoe, ercot) and on GIFT-Eval (m_dense, electricity, us_births, m4_hourly) are load, traffic
    and births series: a sharp daily shape, another shape at the weekend, a level that wanders
    from day to day. f_repeat has one shape per period; daily_weekly has day types but only
    smooth 6-harmonic profiles on five fixed sampling rates, and was never switched on.

    Here: D days per week (7 for 75% of rows, else 2-14), any number of steps per day (24 / 48 /
    96 / 144 / 288 when at least two weeks fit, else anything from 1 up; 1 = daily data with a
    day-of-week pattern), the last 1-2 days of the week use their own shape (related to the
    ordinary day's or not), six shapes (peaks, plateau, bell on a flat floor, random curve, rough
    table, staircase). 40% of rows are EXACT: the week repeats with nothing random. The rest get
    a day-to-day amplitude and level (AR(1) over days), rare holidays, and noise. "Solar" rows are
    a bell with an exactly flat night and a different height every day.
    The family is in PASSTHROUGH_FAMILIES: no seasonality / noise / corruption layer touches it."""
    Mx = int(max(1, min(512, L // 4)))
    M = Mx + 2                                                       # width of the shape tables
    tl = torch.arange(L, device=device).unsqueeze(0)
    zero = torch.zeros(B, 1, device=device)
    D = torch.where(_bern(B, device, 0.75), torch.full((B, 1), 7, device=device),
                    torch.randint(2, 15, (B, 1), device=device))     # days per week
    cap = torch.div(torch.full_like(D, L), 2 * D, rounding_mode="floor").clamp(min=1, max=Mx)   # two weeks must fit
    real = _pick(B, [24.0, 48.0, 96.0, 144.0, 288.0], device).long()
    free = torch.exp(_u(B, 1, device, 0.0, 1.0) * torch.log(cap.float())).round().long().clamp(min=1)
    ppd = torch.where((real <= cap) & _bern(B, device, 0.65), real, torch.minimum(free, cap))    # steps per day
    pf = ppd.float()
    tt = tl + torch.floor(_u(B, 1, device, 0.0, 1.0) * (D * ppd).float()).long()                  # random phase in the week
    dayl = torch.div(tt, ppd, rounding_mode="floor")                 # (B, L) index of the day
    tod = tt - dayl * ppd                                            # step inside the day
    dow = torch.remainder(dayl, D)                                   # day of the week
    nd = L + 16                                                      # static bound on the day index
    exact = _bern(B, device, 0.4)                                    # nothing random from day to day
    k = torch.arange(M, device=device, dtype=torch.float32).unsqueeze(0)
    u = (k / pf).clamp(max=1.0)                                      # time of day in [0, 1)
    inday = (k < pf).float()

    def shape():
        kind = torch.randint(0, 6, (B, 1), device=device)
        bumps = torch.zeros(B, M, device=device)                     # 0: one to three peaks (rush hours)
        nb = torch.randint(1, 4, (B, 1), device=device)
        for j in range(3):
            c = _u(B, 1, device, 0.1, 0.9)
            w = torch.exp(_u(B, 1, device, math.log(0.02), math.log(0.2)))
            dist = torch.remainder(u - c + 0.5, 1.0) - 0.5
            bumps = bumps + (nb > j).float() * _u(B, 1, device, 0.3, 1.0) * torch.exp(-0.5 * (dist / w) ** 2)
        lo = _u(B, 1, device, 0.1, 0.5)                              # 1: plateau (working hours), one or two levels
        hi = lo + _u(B, 1, device, 0.15, 0.45)
        mid = lo + (hi - lo) * _u(B, 1, device, 0.2, 0.8)
        plat = ((u >= lo) & (u < hi)).float() \
            + _u(B, 1, device, -0.6, 0.6) * (_bern(B, device, 0.5) & (u >= mid) & (u < hi)).float()
        s0 = _u(B, 1, device, 0.15, 0.35)                            # 2: bell between sunrise and sunset, flat outside
        s1 = _u(B, 1, device, 0.6, 0.85)
        bell = torch.sin(math.pi * ((u - s0) / (s1 - s0)).clamp(0.0, 1.0)).clamp(min=0.0).pow(torch.exp(_u(B, 1, device, -0.5, 1.0)))
        walk = torch.randn(B, M, device=device).cumsum(1)            # 3: random curve, continuous at midnight
        bridge = walk - k / (pf - 1.0).clamp(min=1.0) * walk.gather(1, (ppd - 1).clamp(0, M - 1))
        rough = torch.randn(B, M, device=device)                     # 4: any value at any time of day
        jump = (torch.rand(B, M, device=device) < _u(B, 1, device, 1.0, 6.0) / pf).long()
        jump[:, 1] = 1
        stair = torch.randn(B, 64, device=device).gather(1, jump.cumsum(1).clamp(max=63))    # 5: a few held levels
        tab = torch.where(kind == 0, bumps, torch.where(kind == 1, plat, torch.where(kind == 2, bell,
              torch.where(kind == 3, bridge, torch.where(kind == 4, rough, stair)))))
        mu = (tab * inday).sum(1, keepdim=True) / pf
        sd = (((tab - mu) ** 2 * inday).sum(1, keepdim=True) / pf).sqrt()
        return torch.where(sd > 1e-4, (tab - mu) / (sd + 1e-6), torch.zeros_like(tab)), kind

    T0, k0 = shape()
    floored = k0 <= 2                                                # shapes that sit on a floor
    solar = (k0 == 2) & (ppd >= 8) & _bern(B, device, 0.6)           # exact flat night, a different peak every day
    floor_ = torch.where(floored, (T0 + (1.0 - inday) * 1e9).min(1, keepdim=True).values, zero)

    def other():                                                     # the shape of a special day
        N, _ = shape()
        mix = torch.where(_bern(B, device, 0.5), _u(B, 1, device, 0.5, 0.95), zero)       # related to T0, or not
        return (mix * T0 + (1.0 - mix * mix).sqrt() * N) * _u(B, 1, device, 0.3, 1.2) + _u(B, 1, device, -1.5, 0.5)

    tab = torch.stack([T0, other(), other()], 1).reshape(B, 3 * M)
    n_off = torch.minimum(torch.randint(1, 3, (B, 1), device=device), D - 1)              # 1-2 special days end the week
    rel = dow - (D - n_off)
    typ = (rel >= 0).long() + ((rel == 1) & _bern(B, device, 0.4)).long()                 # 0 ordinary, 1 / 2 special
    hol = torch.rand(B, nd, device=device) < torch.where(exact, zero, _u(B, 1, device, 0.0, 0.03))
    typ = torch.where(hol.gather(1, dayl) & (typ == 0), torch.ones_like(typ), typ)        # a holiday looks like a weekend day
    typ = torch.where(solar, torch.zeros_like(typ), typ)
    base = tab.gather(1, typ * M + tod)

    on = lambda p: _bern(B, device, p) & ~exact
    amp_sd = torch.where(on(0.6), torch.exp(_u(B, 1, device, math.log(0.02), math.log(0.4))), zero)
    amp_sd = torch.where(solar & ~exact, _u(B, 1, device, 0.2, 0.9), amp_sd)
    da = _ar1_filter(B, nd, device, _u(B, 1, device, 0.0, 0.9))
    amp = torch.exp((amp_sd * da / (da.std(1, keepdim=True) + 1e-6)).clamp(-3.0, 3.0)).gather(1, dayl)
    lvl_sd = torch.where(on(0.5) & ~solar, torch.exp(_u(B, 1, device, math.log(0.02), math.log(0.5))), zero)
    dl = _ar1_filter(B, nd, device, _u(B, 1, device, 0.3, 0.97))
    lvl = (lvl_sd * dl / (dl.std(1, keepdim=True) + 1e-6)).gather(1, dayl)
    dow_sd = torch.where(ppd <= 3, torch.ones_like(zero),
                         torch.where(_bern(B, device, 0.4), _u(B, 1, device, 0.05, 0.5), zero))
    dowl = (torch.randn(B, 14, device=device) * torch.where(solar, zero, dow_sd)).gather(1, dow)   # a level per day of the week
    tn = tl.float() / max(L - 1, 1) - 0.5
    amp = amp * (1.0 + torch.where(_bern(B, device, 0.25), _u(B, 1, device, -0.8, 0.8), zero) * tn)     # slow amplitude drift
    y = floor_ + amp * (base - floor_) + lvl + dowl
    y = y + torch.where(_bern(B, device, 0.25) & ~solar,
                        torch.randn(B, 1, device=device) * _u(B, 1, device, 0.5, 3.0), zero) * tn       # level trend
    nz = torch.where(on(0.7), torch.exp(_u(B, 1, device, math.log(0.003), math.log(0.6))), zero)
    e = torch.randn(B, L, device=device)
    mult = (solar | _bern(B, device, 0.3)) & floored                 # noise that grows with the height above the floor
    return _one_off(torch.where(mult, floor_ + (y - floor_) * torch.exp((nz * e).clamp(-3.0, 3.0)), y + nz * e), device)


def _one_off(y, device):
    """ONE-OFF ANOMALIES for the repeating families: something that happens ONCE and does not come back.

    Why: every series of the repeat families was a pure repetition, so the model learned that
    whatever it saw once at some place in a cycle returns at that place in every cycle. On
    LOOP_SEATTLE_5T ft2_main replays one dip of the history every day (+11% error against ft_repeat).

    For a share _ONE_OFF_P of the rows, 1-2 anomalies at random places, each one of:
      outage  the value falls to the series' floor for a while
      spike   1-3 steps far above or below
      stuck   the value is held (a frozen sensor)
      quiet   a stretch scaled towards the floor (a quiet day) or away from it (a busy one)
    Widths are log-uniform from 1 step to L/16 (at most 256). With _ONE_OFF_P = 0 the series is
    returned untouched and no random number is drawn."""
    B, L = y.shape
    if _ONE_OFF_P <= 0 or L < 64:
        return y
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    lo = y.min(1, keepdim=True).values
    sd = y.std(1, keepdim=True) + 1e-6
    gate = _bern(B, device, _ONE_OFF_P)
    n_a = torch.where(_bern(B, device, 0.35), 2, 1)
    w_hi = math.log(max(2.0, min(L / 16.0, 256.0)))
    for j in range(2):
        act = gate & (n_a > j)
        w = torch.exp(_u(B, 1, device, 0.0, w_hi)).floor().clamp(min=1.0)
        t0 = torch.floor(_u(B, 1, device, 0.03, 0.97) * L)
        kind = torch.randint(0, 4, (B, 1), device=device)
        w = torch.where(kind == 1, w.clamp(max=3.0), w)
        inside = (t >= t0) & (t < t0 + w)
        outage = (lo - _u(B, 1, device, 0.0, 0.5) * sd * _bern(B, device, 0.3).float()).expand(B, L)
        spike = y + torch.where(_bern(B, device, 0.5), 1.0, -1.0) * _u(B, 1, device, 2.0, 6.0) * sd
        stuck = y.gather(1, t0.long().clamp(0, L - 1)).expand(B, L)
        quiet = lo + (y - lo) * torch.where(_bern(B, device, 0.7), _u(B, 1, device, 0.1, 0.7), _u(B, 1, device, 1.3, 2.2))
        new = torch.where(kind == 0, outage, torch.where(kind == 1, spike, torch.where(kind == 2, stuck, quiet)))
        y = torch.where(act & inside, new, y)
    return y


# real long cycles, in steps (the yearly ones three times: daily data is where the evidence is)
_LONG_CAL = [12, 24, 48, 52, 52, 96, 168, 336, 364, 365, 365, 365, 672, 1008, 2016]


def f_long_cycle(B, L, device, _debug=False):
    """LONG CYCLE SEEN ONCE OR TWICE: what happened at this place one long period ago happens again.

    Why: daily data has a yearly cycle and usually 1-4 years of history. Every seasonal pattern in
    training so far was seen at least three times (_pick_period needs three cycles, f_repeat four),
    so the model never had to use "the same dates last year" from ONE earlier occurrence. Measured
    on five daily fev-bench tasks: when the history grows from 273 to 400 days Chronos-2's error
    drops 16%, ours 3%; with under a year of history we are level with it on three of the five.

    Here the generated window holds 2.2 - 4.5 cycles of a long period P (any whole number of steps;
    for 35% of rows a real one when it fits 2-6 times: 365 / 364 days, 52 weeks, 168 hours, ...);
    the model sees a crop of half to all of it, so 1.1 - 6 cycles. A series is the sum of
      a short cycle inside the long one (60% of rows; period 7 for half of them, else 2-30: the
        week in daily data), either any shape or flat with 1-2 different days,
      a smooth shape of the long cycle (70%; three harmonics: the seasons),
      1-8 DATED EVENTS per cycle at fixed places: a dip or a peak 1 to ~P/15 steps wide, a quarter
        with an opposite move just before (the rush before a holiday); 30% of the events are
        CLOSURES that pull the series down to a floor instead of by a fixed amount,
      a SEASON (35%): one stretch of every cycle with its own level and its own short-cycle size.
    35% of rows are EXACT: every cycle is the same. The rest get events whose size changes a little
    from cycle to cycle, 15% of events that MOVE by a few steps (Easter), a wandering level and
    noise. All rows may have growth (level trend, size of everything).
    The family is in PASSTHROUGH_FAMILIES: no seasonality / noise / corruption layer touches it.
    _debug=True also returns (P, exact, the part that must be identical one cycle later, closures)."""
    if L < 64:
        y = f_repeat(B, L, device)
        return (y, None, None, None, None) if _debug else y
    t = torch.arange(L, device=device).unsqueeze(0)                   # (1, L)
    zero = torch.zeros(B, 1, device=device)
    one = torch.ones(B, 1, device=device)
    P = torch.round(L / torch.exp(_u(B, 1, device, math.log(2.2), math.log(4.5)))).clamp(min=12.0)   # (B, 1), whole steps
    cal = [p for p in _LONG_CAL if 2.0 <= L / p <= 6.0]             # real long cycles that fit this window 2-6 times
    if cal:
        P = torch.where(_bern(B, device, 0.35), _pick(B, [float(p) for p in cal], device), P)
    Pl = P.long()
    tt = t + torch.floor(_u(B, 1, device, 0.0, 1.0) * P).long()       # (B, L): random phase in the cycle
    NC = 8                                                            # static bound on the cycle index (at most 6)
    cyc = torch.div(tt, Pl, rounding_mode="floor").clamp(max=NC - 1)  # which cycle
    pos = (tt - torch.div(tt, Pl, rounding_mode="floor") * Pl).float()   # place inside the cycle, in steps
    exact = _bern(B, device, 0.35)                                    # nothing random from cycle to cycle

    # 1. the short cycle (the week): a table of ps values
    ps = torch.where(_bern(B, device, 0.5), torch.full((B, 1), 7, device=device), torch.randint(2, 31, (B, 1), device=device))
    ps = torch.minimum(ps, torch.div(Pl, 6, rounding_mode="floor").clamp(min=2))
    k32 = torch.arange(32, device=device).unsqueeze(0)
    stab = torch.randn(B, 32, device=device)
    off_days = (k32 >= (ps - torch.randint(1, 3, (B, 1), device=device)).clamp(min=1)) & (k32 < ps)
    stab = torch.where(_bern(B, device, 0.4), torch.where(off_days, _u(B, 1, device, -2.5, -0.5), zero) + 0.1 * stab, stab)
    inp = (k32 < ps).float()
    mu = (stab * inp).sum(1, keepdim=True) / ps.float()
    sdv = (((stab - mu) ** 2 * inp).sum(1, keepdim=True) / ps.float()).sqrt()
    stab = (stab - mu) / (sdv + 1e-6)
    short = stab.gather(1, torch.remainder(t + torch.randint(0, 31, (B, 1), device=device), ps))    # (B, L)
    a_s = torch.where(_bern(B, device, 0.6) & (Pl >= 24), _u(B, 1, device, 0.3, 1.5), zero)

    # 2. the smooth shape of the long cycle (the seasons)
    decay = _u(B, 1, device, 0.5, 2.0)
    smooth = torch.zeros(B, L, device=device)
    for k in range(1, 4):
        ang = 2 * math.pi * k * pos / P
        smooth = smooth + torch.full((1, 1), float(k), device=device) ** (-decay) * (
            torch.randn(B, 1, device=device) * torch.cos(ang) + torch.randn(B, 1, device=device) * torch.sin(ang))
    smooth = smooth / (smooth.std(1, keepdim=True) + 1e-6)
    a_y = torch.where(_bern(B, device, 0.7), _u(B, 1, device, 0.3, 1.5), zero)

    # 3. dated events: the same thing at the same place of every cycle
    NE = 8
    n_ev = torch.randint(1, NE + 1, (B, 1), device=device)
    ev = torch.zeros(B, L, device=device)                             # events that add or subtract
    dip = torch.zeros(B, L, device=device)                            # closures: share of the way down to the floor
    for j in range(NE):
        on_j = n_ev > j
        c = torch.floor(_u(B, 1, device, 0.0, 1.0) * P)               # where in the cycle it starts
        wk = torch.randint(0, 5, (B, 1), device=device)               # 40% one step, 40% 2-7 steps, 20% a few % of the cycle
        w = torch.where(wk <= 1, one, torch.where(wk <= 3, torch.randint(2, 8, (B, 1), device=device).float(),
                                                  torch.floor(_u(B, 1, device, 0.02, 0.07) * P).clamp(min=2.0)))
        w = torch.minimum(w, torch.floor(P / 6.0).clamp(min=1.0))
        mv = torch.where(_bern(B, device, 0.15) & ~exact, torch.floor(_u(B, 1, device, 0.01, 0.04) * P), zero)
        shift = torch.round(mv * torch.randn(B, NC, device=device)).gather(1, cyc)        # a date that moves (Easter)
        size = (1.0 + torch.where(_bern(B, device, 0.5) & ~exact, _u(B, 1, device, 0.05, 0.3), zero)
                * torch.randn(B, NC, device=device)).clamp(0.3, 1.7).gather(1, cyc)       # not the same size every cycle
        d = torch.remainder(pos - c - shift + 0.5 * P, P) - 0.5 * P   # steps since the start of the event (circular)
        box = ((d >= 0) & (d < w)).float()
        bump = torch.exp(-0.5 * ((d - 0.5 * (w - 1.0)) / (0.3 * w + 0.2)) ** 2) * ((d >= -w) & (d < 2.0 * w)).float()
        kern = torch.where(_bern(B, device, 0.6) | (w <= 2), box, bump)
        lead = ((d >= -torch.ceil(1.5 * w)) & (d < 0)).float()        # the steps just before it
        m = torch.where(_bern(B, device, 0.55), -one, one) * _u(B, 1, device, 1.0, 5.0)
        closure = _bern(B, device, 0.3)
        add = m * kern + torch.where(_bern(B, device, 0.25), -0.3 * m, zero) * lead
        ev = ev + (on_j & ~closure).float() * size * add
        dip = torch.maximum(dip, (on_j & closure).float() * _u(B, 1, device, 0.5, 1.0) * kern)

    # 4. a season: one stretch of every cycle with its own level and short-cycle size (summer)
    in_s = (torch.remainder(pos - torch.floor(_u(B, 1, device, 0.0, 1.0) * P), P)
            < torch.floor(_u(B, 1, device, 0.04, 0.3) * P).clamp(min=2.0)) & _bern(B, device, 0.35)
    s_lvl = torch.where(in_s, _u(B, 1, device, -2.0, 2.0), zero)
    s_amp = torch.where(in_s, _u(B, 1, device, 0.2, 1.8), one)

    recur = a_y * smooth + ev + s_lvl                                  # identical one cycle later in the exact rows
    y = a_s * s_amp * short + recur
    tn = t.float() / max(L - 1, 1) - 0.5
    y = y * (1.0 + torch.where(_bern(B, device, 0.3), _u(B, 1, device, -0.6, 0.6), zero) * tn)        # everything grows / shrinks
    y = y + torch.where(_bern(B, device, 0.4), torch.randn(B, 1, device=device) * _u(B, 1, device, 0.5, 2.5), zero) * tn
    lv = _ar1_filter(B, L, device, _u(B, 1, device, 0.9, 0.995))
    lv_sd = torch.where(_bern(B, device, 0.4) & ~exact, torch.exp(_u(B, 1, device, math.log(0.05), math.log(0.6))), zero)
    y = y + lv_sd * lv / (lv.std(1, keepdim=True) + 1e-6)             # a level that wanders
    y = y + torch.where(exact, zero, torch.exp(_u(B, 1, device, math.log(0.01), math.log(0.5)))) * torch.randn(B, L, device=device)
    floor_ = y.min(1, keepdim=True).values - _u(B, 1, device, 0.2, 1.5)
    y = floor_ + (y - floor_) * (1.0 - dip)                           # closures pull it down to the floor
    y = _one_off(y, device)
    return (y, P, exact, recur, dip) if _debug else y


# name -> (fn, is_count, is_nonneg, allow_additive_layers, v1 weight)
FAMILIES = {
    "gp_kernel":        (f_gp_kernel,            False, False, True,  0.10),
    "power_law":        (f_power_law,            False, False, True,  0.08),
    "random_walk":      (f_random_walk,          False, False, True,  0.07),
    "ar_process":       (f_ar_process,           False, False, True,  0.07),
    "mean_revert":      (f_mean_revert,          False, False, True,  0.05),
    "stochastic_vol":   (f_stochastic_vol,       False, False, True,  0.08),
    "regime_switch":    (f_regime_switch,        False, False, True,  0.05),
    "level_shift":      (f_level_shift,          False, False, True,  0.05),
    "piecewise_trend":  (f_piecewise_trend,      False, False, True,  0.05),
    "logistic_growth":  (f_logistic_growth,      False, False, True,  0.03),
    "damped_osc":       (f_damped_osc,           False, False, True,  0.02),
    "diurnal_envelope": (f_diurnal_envelope,     False, True,  False, 0.05),
    "spike_train":      (f_spike_train,          False, True,  False, 0.04),
    "burst_onoff":      (f_burst_onoff,          False, True,  False, 0.04),
    "nonlinear_ar":     (f_nonlinear_ar,         False, False, True,  0.03),
    "trend_season":     (f_trend_season_classic, False, False, True,  0.08),
    "flat_sparse":      (f_flat_sparse,          False, False, True,  0.04),
    "intermittent":     (f_intermittent_latent,  True,  True,  True,  0.05),
    "count":            (f_count_latent,         True,  True,  True,  0.05),
    "multiscale":       (f_multiscale_mix,       False, False, True,  0.04),
    "short_macro_trend": (f_short_macro_trend,   False, False, False, 0.05),
    # ---- clean-corpus families (weight 0 in v1, so v1 is unchanged)
    "ets":              (f_ets,                  False, False, False, 0.0),
    "ar_p":             (f_ar_p,                 False, False, True,  0.0),
    "tsi":              (f_tsi,                  False, False, False, 0.0),
    "flat_hold":        (f_flat_hold,            False, False, False, 0.0),
    "cumulative":       (f_cumulative,           False, True,  False, 0.0),
    # ---- clean2 additions (weight 0 in v1 and clean, so both are unchanged)
    "aggregated":       (f_aggregated,           False, True,  False, 0.0),
    "bursty_load":      (f_bursty_load,          False, True,  False, 0.0),
    "daily_weekly":     (f_daily_weekly,         False, False, False, 0.0),
    "epidemic":         (f_epidemic,             False, True,  False, 0.0),
    "sarima":           (f_sarima,               False, False, False, 0.0),
    # ---- fine-tune addition
    "repeat":           (f_repeat,               False, False, False, 0.0),
    "repeat_week":      (f_repeat_week,          False, False, False, 0.0),
    "long_cycle":       (f_long_cycle,           False, False, False, 0.0),
}

DEFAULT_WEIGHTS = {k: v[4] for k, v in FAMILIES.items() if v[4] > 0}

# Chronos-2-style mix: textbook components (KernelSynth, ETS, AR(p), TSI) dominate; the
# realism families stay but at small weights; rough/zero-heavy families are cut back.
CLEAN_WEIGHTS = {
    "gp_kernel": 0.12, "ets": 0.16, "ar_p": 0.10, "tsi": 0.10, "random_walk": 0.05,
    "ar_process": 0.03, "mean_revert": 0.03, "stochastic_vol": 0.04, "regime_switch": 0.02,
    "level_shift": 0.03, "piecewise_trend": 0.04, "logistic_growth": 0.03, "damped_osc": 0.01,
    "diurnal_envelope": 0.03, "spike_train": 0.02, "burst_onoff": 0.02, "nonlinear_ar": 0.01,
    "trend_season": 0.02, "flat_sparse": 0.01, "flat_hold": 0.02, "intermittent": 0.02,
    "count": 0.03, "multiscale": 0.02, "short_macro_trend": 0.05, "cumulative": 0.02,
}

# families whose defining property (exact flatness, monotonicity) the noise / support /
# corruption layers would destroy: they skip those layers and only get rescaled
PASSTHROUGH_FAMILIES = {"flat_hold", "cumulative", "bursty_load", "epidemic", "repeat", "repeat_week", "long_cycle"}

GEN_PROFILES = {
    "v1": dict(weights=DEFAULT_WEIGHTS, support_probs=(0.2, 0.2, 0.2, 0.2, 0.2),
               out_p=0.25, miss_p=0.20, quant_p=0.15, quant_step=(0.05, 0.5),
               clean_p=0.0, fix_degenerate=True),
    # clean: identity/positive dominate the support map, hard clipping 20% -> 5%, zero-floor
    # 20% -> 12%; half the rows skip corruption entirely; outliers 25% -> 8%, imputed runs
    # 20% -> 6%, quantisation 15% -> 6% with a 5x finer step. Constant rows are left constant
    # (the model must learn flat -> flat; arcsinh + the std floor make that numerically safe).
    "clean": dict(weights=CLEAN_WEIGHTS, support_probs=(0.45, 0.25, 0.12, 0.13, 0.05),
                  out_p=0.08, miss_p=0.06, quant_p=0.06, quant_step=(0.01, 0.1),
                  clean_p=0.5, fix_degenerate=False),
}

# ---- clean2 = clean + the switchable additions (section 3c)
GEN_ADDITIONS_ALL = dict(agg=True, periods=True, epidemic=True, repeat=False, daytype=False,
                         yearly=False, oneoff=False,
                         bursty=False, weekday=False, breaks=False, tv_season=False, sarima=False)
# family -> (switch, weight). CLEAN_WEIGHTS sums to 1.0; these add 0.23 on top (renormalised
# by _allocate), so the full clean2 mix is ~19% new families, ~81% the v2 corpus.
CLEAN2_FAMILY_WEIGHTS = {
    "aggregated":   ("agg",      0.08),
    "sarima":       ("sarima",   0.05),
    "daily_weekly": ("weekday",  0.05),
    "bursty_load":  ("bursty",   0.03),
    "epidemic":     ("epidemic", 0.02),
    "repeat":       ("repeat",   0.15),     # 0.15 / 1.25 = 12% of generated series
    "repeat_week":  ("daytype",  0.10),     # with repeat + weekday + daytype on: repeat 10.7%, repeat_week 7.1%,
                                            # daily_weekly 3.6%, the v5 corpus 78.6%
    "long_cycle":   ("yearly",   0.15),     # with yearly on as well: long_cycle 9.7%, repeat 9.7%, repeat_week 6.5%,
}                                           # daily_weekly 3.2%, the v5 corpus 71.0%


def make_clean2_profile(additions=None, period_ext_share=0.15):
    a = dict(GEN_ADDITIONS_ALL)
    for k, v in (additions or {}).items():
        assert k in a, f"unknown gen_addition {k!r}; valid: {sorted(a)}"
        a[k] = bool(v)
    w = dict(CLEAN_WEIGHTS)
    for fam, (key, wt) in CLEAN2_FAMILY_WEIGHTS.items():
        if a[key]:
            w[fam] = wt
    pr = dict(GEN_PROFILES["clean"])
    pr.update(weights=w, period_bank=PERIOD_BANK_EXT if a["periods"] else PERIOD_BANK,
              period_ext_share=period_ext_share if a["periods"] else None,
              breaks_p=0.2 if a["breaks"] else 0.0, tv_season=a["tv_season"], additions=a,
              one_off_p=0.3 if a["oneoff"] else 0.0)
    return pr


GEN_PROFILES["clean2"] = make_clean2_profile()


def resolve_profile(cfg, key="corpus"):
    """Profile name for cfg[key]; (re)builds clean2 from cfg['gen_additions']."""
    name = cfg.get(key) or cfg.get("corpus", "v1")
    if name == "clean2":
        GEN_PROFILES["clean2"] = make_clean2_profile(cfg.get("gen_additions"),
                                                     cfg.get("period_ext_share", None))
    return name


# ============================ stochastic layers ============================

def _L_seasonality(x, allow_add, device, p=0.68):
    """1-3 nested seasonal components with NON-SINUSOIDAL shapes.

    Each component is K harmonics of a base period with power-law-decaying random
    coefficients. The decay exponent is randomized per series, which is what produces the
    variety of real seasonal shapes -- slow decay gives sharp spiky peaks (commute traffic),
    fast decay gives smooth single humps (temperature). A single sine wave, which is what
    the original generator used, produces none of this shape diversity.

    Nested periods (e.g. 24 and 24*7) are how real data carries intraday AND day-of-week
    structure simultaneously -- electricity load is the canonical case.
    """
    B, L = x.shape
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    gate = _bern(B, device, p) & allow_add.unsqueeze(-1)

    total = torch.zeros_like(x)
    tv = _TV_SEASON
    if tv:
        tv_am, tv_ph = _seasonal_drift(B, L, device)
    base_P = _pick_period(B, L, device)
    n_extra = torch.randint(0, 3, (B, 1), device=device)      # 0-2 nested extra periods

    for slot in range(3):
        if slot == 0:
            P = base_P
            active = torch.ones(B, 1, device=device, dtype=torch.bool)
        else:
            # Nested longer cycle (x7 = weekly-on-daily, x12/x30/x52 = annual-on-*).
            # It is DROPPED, not clamped, when it doesn't fit the window. Clamping was a
            # real bug: base_P=24 x30=720 clamped to L/3=341.33 is a fractional,
            # non-canonical "period" that is not a cycle at all -- it injected
            # quasi-periodic structure that diluted the genuine base seasonality, and it
            # was why measured seasonal coverage was stuck near 25% even with the gate
            # forced fully on.
            P = base_P * _pick(B, [7.0, 12.0, 30.0, 52.0], device)
            active = (n_extra >= slot) & (P * 3.0 <= L)
        decay = _u(B, 1, device, 0.3, 1.6)                     # harmonic shape sharpness
        comp = torch.zeros_like(x)
        for k in range(1, 5):
            # harmonic amplitude ~ k^-decay: small decay -> sharp spiky seasonal shape,
            # large decay -> smooth single hump. Randomizing `decay` per series is what
            # gives the variety of real seasonal profiles.
            amp_k = torch.pow(torch.full((1, 1), float(k), device=device), -decay)
            ca = torch.randn(B, 1, device=device) * amp_k
            cb = torch.randn(B, 1, device=device) * amp_k
            ang = 2 * math.pi * k * t / P
            if tv:
                ang = ang + k * tv_ph
            comp = comp + ca * torch.cos(ang) + cb * torch.sin(ang)
        # Amplitude relative to the base process. Too small and the seasonal term is
        # swamped by a random-walk base in the differenced spectrum. Nested cycles get a
        # smaller amplitude than the primary one, which is how real multi-seasonal data
        # behaves -- the day-of-week effect modulates the intraday cycle rather than
        # competing with it on equal terms.
        amp = _u(B, 1, device, 0.8, 4.0) if slot == 0 else _u(B, 1, device, 0.3, 1.2)
        total = total + torch.where(active, comp * amp, torch.zeros_like(comp))

    if tv:
        total = total * tv_am
    sd = x.std(dim=1, keepdim=True) + 1e-6
    additive = x + total * sd
    # multiplicative seasonality: amplitude grows with level (retail/energy standard)
    mult = x * (1.0 + 0.5 * total / (total.std(dim=1, keepdim=True) + 1e-6))
    use_mult = _bern(B, device, 0.3)
    seasoned = torch.where(use_mult, mult, additive)
    return torch.where(gate, seasoned, x)


def _L_trend(x, allow_add, device, p=0.45):
    """Linear / exponential / piecewise-linear trend on top of the latent process."""
    B, L = x.shape
    t = torch.linspace(0, 1, L, device=device).unsqueeze(0)
    gate = _bern(B, device, p) & allow_add.unsqueeze(-1)
    sd = x.std(dim=1, keepdim=True) + 1e-6

    kind = torch.randint(0, 3, (B, 1), device=device)
    lin = t * _u(B, 1, device, -5.0, 5.0)
    expo = (torch.exp((t * _u(B, 1, device, 0.5, 3.0)).clamp(max=_EXP_CLAMP)) - 1.0)
    expo = expo * _u(B, 1, device, -1.5, 1.5)
    piece = (_piecewise_constant(B, L, device, 1.0, 4.0) * _u(B, 1, device, 0.02, 0.3)).cumsum(1)
    trend = torch.where(kind == 0, lin, torch.where(kind == 1, expo, piece))
    return torch.where(gate, x + trend * sd, x)


def _L_calendar(x, allow_add, device, p=0.3):
    """Business-day and holiday effects.

    Real daily data is not smoothly seasonal -- it has a hard 5/2 weekday-weekend split
    (restaurant, sales, traffic) and irregular single-day holiday shocks. Crucially,
    COVID-style reporting artifacts (deaths reported in weekday batches with weekend
    dips) look exactly like this, and covid_deaths was your single worst GIFT-Eval config
    at MASE 38.8 -- a model that has never seen a weekday/weekend reporting pattern has
    no way to represent it.
    """
    B, L = x.shape
    gate = _bern(B, device, p) & allow_add.unsqueeze(-1)
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    sd = x.std(dim=1, keepdim=True) + 1e-6

    pts_per_day = _pick(B, [1.0, 24.0, 48.0, 96.0], device)
    day_idx = torch.div(t, pts_per_day, rounding_mode="floor").long().clamp_(0, L - 1)
    dow = day_idx % 7
    weekend = (dow >= 5).float()
    weekend_effect = weekend * _u(B, 1, device, -1.2, 0.3)

    hol_day = (torch.rand(B, L, device=device) < 0.02).float() * _u(B, 1, device, -2.0, 2.0)
    hol = torch.gather(hol_day, 1, day_idx)

    return torch.where(gate, x + (weekend_effect + hol) * sd, x)


def _L_noise(x, is_count, device):
    """Observation noise: homoskedastic Gaussian, heteroskedastic (variance scaling with
    level -- the multiplicative-error regime), or heavy-tailed via a log-normal scale
    mixture (kurtosis well above Gaussian, as in financial and price-spike data).

    Skipped for count rows, where integer-valuedness is added later and continuous noise
    on top would destroy it.
    """
    B, L = x.shape
    sd = x.std(dim=1, keepdim=True) + 1e-6
    kind = torch.randint(0, 3, (B, 1), device=device)
    # Noise magnitude is deliberately modest (mostly <=8% of signal sd, occasionally up to
    # 25%). An earlier draft used up to 35% and it measurably WHITENED the corpus -- median
    # spectral slope came out at -0.73 and only 31% of rows were highly persistent, whereas
    # real corpora are dominated by strongly negative slopes and near-unit-root persistence.
    # Observation noise should perturb a series, not dominate its spectrum.
    big_noise = _bern(B, device, 0.15)
    sigma = torch.where(big_noise, _u(B, 1, device, 0.08, 0.25),
                        _u(B, 1, device, 0.005, 0.08)) * sd

    gauss = torch.randn(B, L, device=device) * sigma
    level = (x - x.mean(1, keepdim=True)).abs() / sd
    het = torch.randn(B, L, device=device) * sigma * (0.3 + level)
    heavy = torch.randn(B, L, device=device) * sigma * torch.exp(
        (torch.randn(B, L, device=device) * _u(B, 1, device, 0.4, 1.1)).clamp(-_EXP_CLAMP, _EXP_CLAMP))

    noise = torch.where(kind == 0, gauss, torch.where(kind == 1, het, heavy))
    return torch.where(is_count.unsqueeze(-1), x, x + noise)


def _L_support(x, is_count, is_nonneg, device, probs=(0.2, 0.2, 0.2, 0.2, 0.2)):
    """Map the latent process onto a realistic SUPPORT.

    Returns (x, is_count_out, is_nonneg_out). Branches:
      identity     -- unconstrained real (financial returns, temperature anomalies)
      positive     -- exp() or softplus: strictly positive, multiplicative errors
      zero-floor   -- relu: non-negative WITH a real mass at exactly zero (demand)
      counts       -- Poisson / negative-binomial integers (births, admissions, deaths)
      intermittent -- Bernoulli-masked integer counts, mostly zeros (car_parts)
      bounded      -- sigmoid-squashed into [lo, hi], or hard-clipped (censoring,
                      utilization %, sensor saturation, stockouts)
    """
    B, L = x.shape
    xs = _standardize(x)

    # --- continuous branches
    positive = torch.exp((xs * _u(B, 1, device, 0.2, 1.0)).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    zero_floor = torch.relu(xs + _u(B, 1, device, -1.0, 0.8))
    lo = _u(B, 1, device, -1.0, 0.0)
    hi = lo + _u(B, 1, device, 0.5, 3.0)
    bounded = lo + (hi - lo) * torch.sigmoid(xs * _u(B, 1, device, 0.5, 2.5))
    censored = torch.clamp(xs, min=lo, max=hi)

    # --- count branches (intensity must be positive)
    intensity = torch.exp((xs * _u(B, 1, device, 0.3, 1.0)
                           + _u(B, 1, device, -1.0, 3.0)).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    intensity = torch.nan_to_num(intensity, nan=1.0, posinf=math.exp(_EXP_CLAMP), neginf=0.0)
    counts = torch.poisson(intensity)
    # negative binomial = Poisson with gamma-mixed intensity (overdispersion, very common
    # in real count data where Poisson's variance==mean assumption is violated)
    overdisp = torch.exp((torch.randn(B, 1, device=device) * 0.5).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    negbin = torch.poisson(intensity * overdisp)
    demand_p = _u(B, 1, device, 0.05, 0.5)
    interm = torch.poisson(intensity * 0.6) * (torch.rand(B, L, device=device) < demand_p).float()

    kind = torch.multinomial(torch.tensor(probs, device=device, dtype=torch.float32), B,
                             replacement=True).unsqueeze(1)
    cont = torch.where(kind == 0, xs,
           torch.where(kind == 1, positive,
           torch.where(kind == 2, zero_floor,
           torch.where(kind == 3, bounded, censored))))
    cont_nonneg = (kind == 1) | (kind == 2)

    cnt_kind = torch.randint(0, 3, (B, 1), device=device)
    cnt = torch.where(cnt_kind == 0, counts, torch.where(cnt_kind == 1, negbin, interm))

    want_count = is_count.unsqueeze(-1)
    out = torch.where(want_count, cnt, cont)
    out_is_count = want_count.squeeze(-1)
    out_is_nonneg = (want_count | cont_nonneg | is_nonneg.unsqueeze(-1)).squeeze(-1)
    # families with a structural zero floor (solar night, bursts) keep their own values
    keep_own = is_nonneg.unsqueeze(-1) & ~want_count
    out = torch.where(keep_own, x, out)
    return out, out_is_count, out_is_nonneg


def _L_corruption(x, is_count, is_nonneg, device, out_p=0.25, miss_p=0.2, quant_p=0.15,
                  quant_step=(0.05, 0.5), clean_p=0.0):
    """Measurement reality: additive outliers, missing runs (forward-filled, which is what
    an imputed real series actually looks like), and quantization/rounding.

    Missing data is forward-filled rather than left as NaN because the model's attention
    has no padding-mask support -- so this teaches robustness to the FLAT-LINED RUNS that
    imputation leaves behind, which is exactly the artifact present in GIFT-Eval's
    *_with_missing datasets.
    """
    B, L = x.shape
    sd = x.std(dim=1, keepdim=True) + 1e-6
    dirty = ~_bern(B, device, clean_p)          # clean gate: these rows get NO corruption

    # additive outliers
    out_gate = _bern(B, device, out_p) & dirty
    spike_mask = (torch.rand(B, L, device=device) < (_u(B, 1, device, 1.0, 12.0) / max(L, 1))).float()
    spikes = spike_mask * torch.randn(B, L, device=device) * sd * _u(B, 1, device, 3.0, 12.0)
    x = torch.where(out_gate, x + spikes, x)

    # missing runs -> forward fill (flat-lined segments)
    miss_gate = _bern(B, device, miss_p) & dirty
    run_start = torch.rand(B, L, device=device) < (_u(B, 1, device, 1.0, 8.0) / max(L, 1))
    run_len = _segment_ids(B, L, device, 1.0, 8.0)          # reuse segmentation for run ids
    in_run = run_start.cumsum(1) % 2 == 1
    observed = ~(in_run & (run_len % 3 == 0))
    observed[:, 0] = True
    filled = _forward_fill(x, observed)
    x = torch.where(miss_gate, filled, x)

    # quantization / rounding to a coarse grid (real sensors and reported figures)
    quant_gate = _bern(B, device, quant_p) & ~is_count.unsqueeze(-1) & dirty
    step = sd * _u(B, 1, device, quant_step[0], quant_step[1])
    x = torch.where(quant_gate, torch.round(x / step) * step, x)

    x = torch.where(is_nonneg.unsqueeze(-1), torch.relu(x), x)
    x = torch.where(is_count.unsqueeze(-1), torch.round(torch.relu(x)), x)
    return x


def _L_scale(x, is_count, is_nonneg, device):
    """Random affine rescaling spanning several orders of magnitude.

    Real corpora span from car_parts (values ~1) to electricity/W (values ~1e5) -- your
    own GIFT-Eval CSV showed MSE from 1.3 to 6.8e10. The model instance-normalizes, so
    this mainly exercises that normalization path and guards against any latent
    scale-dependence.

    Two exclusions, both load-bearing:
      - Offsets are suppressed for non-negative rows, so the meaningful zero floor survives.
      - COUNT rows are left completely untouched. A continuous rescale would turn integers
        back into arbitrary reals and silently destroy integer-valuedness -- which is one
        of the realism properties this whole file exists to produce. Real count data is
        integer at its native scale, so it simply does not get rescaled.
    """
    B, L = x.shape
    scale = torch.exp((torch.randn(B, 1, device=device) * 2.0).clamp(-_EXP_CLAMP, _EXP_CLAMP))
    offset = torch.where(is_nonneg.unsqueeze(-1),
                         torch.zeros(B, 1, device=device),
                         torch.randn(B, 1, device=device) * 5.0)
    scaled = x * scale + offset
    return torch.where(is_count.unsqueeze(-1), x, scaled)


def _finalize(x, fix_degenerate=True):
    """Guard rails: strip non-finite values, clamp magnitude, and replace degenerate
    (zero-variance) series -- which carry no training signal and divide-by-epsilon in
    instance norm -- with a small amount of noise."""
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).clamp(-_CLAMP, _CLAMP)
    if not fix_degenerate:
        return x
    sd = x.std(dim=1, keepdim=True)
    degenerate = sd < 1e-6
    return torch.where(degenerate, x + torch.randn_like(x) * 0.1, x)


# ============================ top-level API ============================

def _allocate(B, weights):
    """Fixed per-family row counts for a batch. Deterministic given (B, weights), which
    keeps tensor shapes static across steps -- important so torch.compile / CUDA graphs
    don't recompile every step."""
    names = [n for n in weights if weights[n] > 0]
    tot = sum(weights[n] for n in names)
    counts, acc = [], 0
    for n in names[:-1]:
        c = int(round(B * weights[n] / tot))
        counts.append((n, c))
        acc += c
    counts.append((names[-1], max(0, B - acc)))
    return counts


@torch.no_grad()
def generate_batch(B, L, device, weights=None, apply_layers=True, profile="v1"):
    """Generate a batch of diverse synthetic series.

    Parameters
    ----------
    B : int     batch size (rows)
    L : int     series length (should be ctx_len + pred_len for training)
    device      torch device
    weights     optional {family_name: weight}; defaults to DEFAULT_WEIGHTS
    apply_layers : set False to get raw base families with no layering (for debugging /
                   ablating how much the layer stack actually contributes)

    Returns (B, L) float32 on `device`. Rows are shuffled so families are not contiguous
    in the batch (which would otherwise correlate with position in a DDP shard).
    """
    global _PERIOD_BANK_ACTIVE, _PERIOD_EXT_SHARE, _TV_SEASON, _ONE_OFF_P
    pr = GEN_PROFILES[profile]
    weights = weights or pr["weights"]
    _PERIOD_BANK_ACTIVE = pr.get("period_bank", PERIOD_BANK)
    _PERIOD_EXT_SHARE = pr.get("period_ext_share", None)
    _TV_SEASON = bool(pr.get("tv_season", False))
    _ONE_OFF_P = float(pr.get("one_off_p", 0.0))
    xs, cnt_f, nn_f, add_f, pt_f = [], [], [], [], []
    for name, n in _allocate(B, weights):
        if n <= 0:
            continue
        fn, is_cnt, is_nn, allow_add, _ = FAMILIES[name]
        xs.append(fn(n, L, device))
        cnt_f.append(torch.full((n,), is_cnt, device=device, dtype=torch.bool))
        nn_f.append(torch.full((n,), is_nn, device=device, dtype=torch.bool))
        add_f.append(torch.full((n,), allow_add, device=device, dtype=torch.bool))
        pt_f.append(torch.full((n,), name in PASSTHROUGH_FAMILIES, device=device, dtype=torch.bool))

    x = torch.cat(xs, dim=0)
    passthrough = torch.cat(pt_f, dim=0).unsqueeze(-1)
    # guard: a single non-finite row (numerically extreme parameter draw in ANY family) must
    # never reach torch.poisson, which hard-asserts on GPU and kills the whole run
    bad_rows = ~torch.isfinite(x).all(dim=1, keepdim=True)
    x = torch.where(bad_rows, torch.randn_like(x).cumsum(1), x)
    is_count = torch.cat(cnt_f, dim=0)
    is_nonneg = torch.cat(nn_f, dim=0)
    allow_add = torch.cat(add_f, dim=0)

    if apply_layers:
        x_raw, nn_raw = x, is_nonneg
        x = _L_seasonality(x, allow_add, device)
        x = _L_trend(x, allow_add, device)
        x = _L_calendar(x, allow_add, device)
        if pr.get("breaks_p", 0.0) > 0:
            x = _L_breaks(x, device, pr["breaks_p"])
        x = _L_noise(x, is_count, device)
        x, is_count, is_nonneg = _L_support(x, is_count, is_nonneg, device, pr["support_probs"])
        x = _L_corruption(x, is_count, is_nonneg, device, pr["out_p"], pr["miss_p"], pr["quant_p"],
                          pr["quant_step"], pr["clean_p"])
        x = torch.where(passthrough, x_raw, x)
        is_nonneg = torch.where(passthrough.squeeze(-1), nn_raw, is_nonneg)
        x = _L_scale(x, is_count, is_nonneg, device)

    x = _finalize(x, pr["fix_degenerate"])
    return x[torch.randperm(x.shape[0], device=device)]


# ================= 3d. GROUPS: multivariate / covariate data for group attention =================
# Chronos-2 trained its group attention on synthetic "multivariatizers" applied to univariate
# base series; this is the same idea. A BLOCK is `V` aligned series (lanes) of equal length.
# With probability group_prob the first g lanes (g in 2..V) are turned into a RELATED group of
# one of five kinds; all other lanes stay unrelated single series (each its own group of one).
#
#   mix        every variate = own series + random sparse mix of the others (shared factors:
#              stores of one chain, sensors of one plant). All targets.
#   leadlag    variate i = own series + lagged copy of a driver variate (lag 1..128 steps):
#              the driver's PAST predicts the others' future. All targets.
#   known_cov  1-2 targets driven by covariates whose FUTURE is known (promotions, prices,
#              calendar flags, weather forecasts): additive or multiplicative, linear or
#              nonlinear effects -- including V-SHAPED ones (load is high when it is hot AND
#              when it is cold) and, for a quarter of these groups, an ON/OFF GATE (a 'store
#              open' flag: the target is exactly zero while it is off). ~40% of the other
#              covariates are irrelevant DECOYS.
#   past_cov   same, but the covariates are only observed in the past and act with a lag.
#   unrelated  independent series declared as one group with random roles: nothing to learn
#              from each other -- the model must learn to ignore variates that do not help.

GROUP_KINDS = ("mix", "leadlag", "known_cov", "past_cov", "unrelated")
# v4 used (0.30, 0.15, 0.30, 0.10, 0.15). On fev-bench the gain from groups came almost entirely
# from KNOWN covariates on single-target tasks (Chronos-2: -16% error on those, v4: -4%), so
# known-covariate groups now make up half of all groups.
GROUP_KIND_PROBS = (0.15, 0.10, 0.50, 0.10, 0.15)
_GROUP_KIND_PROBS_V5 = GROUP_KIND_PROBS
_COV_SMALL_GROUP_P = 0.0      # share of covariate groups limited to 2-4 variates
_COV_DOMINANT_P = 0.0         # share of covariate groups whose target is almost entirely its covariates
_SECOND_GROUP_P = 0.0         # chance of a SECOND related group in the upper half of a block's lanes
_COV_R3 = False               # round-3 covariate data (see _mv_covariates); off for the holdout


def set_group_options(cfg=None):
    """Group-data settings of the run that is about to generate (None = the v5 defaults). Module
    globals, like the active period bank: generation is single-threaded."""
    global GROUP_KIND_PROBS, _COV_SMALL_GROUP_P, _COV_DOMINANT_P, _SECOND_GROUP_P, _COV_R3
    cfg = cfg or {}
    GROUP_KIND_PROBS = tuple(cfg.get("group_kind_probs") or _GROUP_KIND_PROBS_V5)
    assert len(GROUP_KIND_PROBS) == len(GROUP_KINDS)
    _COV_SMALL_GROUP_P = float(cfg.get("cov_small_group_p") or 0.0)
    _COV_DOMINANT_P = float(cfg.get("cov_dominant_p") or 0.0)
    _SECOND_GROUP_P = float(cfg.get("second_group_p") or 0.0)
    _COV_R3 = bool(cfg.get("cov_r3") or False)


def _covariate_signal(base, device):
    """base (n, L) standardized series -> covariate values (n, L), its effect input (n, L) and
    is_binary (n, 1). Types: continuous (the base series itself: weather-like), binary events
    (iid days or on/off runs: promotions, holidays), piecewise-constant level (price), and a
    periodic calendar flag (e.g. 'weekend')."""
    n, L = base.shape
    t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
    kind = torch.randint(0, 4, (n, 1), device=device)
    iid = torch.rand(n, L, device=device) < _u(n, 1, device, 0.01, 0.2)
    seg = _segment_ids(n, L, device, 2.0, 20.0)
    runs = torch.gather(torch.rand(n, _MAX_SEG, device=device) < _u(n, 1, device, 0.1, 0.5), 1, seg)
    binary = torch.where(_bern(n, device, 0.5), iid, runs).float()
    price = _piecewise_constant(n, L, device, 1.0, 10.0)
    P = _pick_period(n, L, device, min_cycles=2.0)
    cal = (torch.remainder(t + torch.floor(_u(n, 1, device, 0.0, 1.0) * P), P)
           < _u(n, 1, device, 0.1, 0.5) * P).float()
    raw = torch.where(kind == 0, base, torch.where(kind == 1, binary, torch.where(kind == 2, price, cal)))
    is_bin = (kind == 1) | (kind == 3)
    return raw, torch.where(is_bin, raw, _standardize(raw)), is_bin


def _mv_mix(z, act, mu, sd, device):
    n, V, L = z.shape
    W = torch.randn(n, V, V, device=device) * (torch.rand(n, V, V, device=device) < 0.5).float() \
        * _u(n, 1, device, 0.3, 1.5).unsqueeze(-1)
    W = W * act.unsqueeze(2).float() * act.unsqueeze(1).float() * (1.0 - torch.eye(V, device=device))
    y = z * _u(n, V, device, 0.3, 1.0).unsqueeze(-1) + torch.bmm(W, z)
    return y * sd + mu, torch.zeros(n, V, dtype=torch.long, device=device)


def _mv_leadlag(z, act, mu, sd, device):
    n, V, L = z.shape
    d = torch.exp(_u(n, V, device, 0.0, math.log(128.0))).long()
    idx = (torch.arange(L, device=device).view(1, 1, L) - d.unsqueeze(-1)).clamp_(min=0)
    drv = z[:, :1, :].expand(n, V, L).gather(2, idx)
    sgn = torch.where(torch.rand(n, V, device=device) < 0.5, 1.0, -1.0)
    y = z * _u(n, V, device, 0.2, 1.0).unsqueeze(-1) + (sgn * _u(n, V, device, 0.5, 2.0)).unsqueeze(-1) * drv
    lane0 = (torch.arange(V, device=device) == 0).view(1, V, 1)
    y = torch.where(lane0, z, y)
    return y * sd + mu, torch.zeros(n, V, dtype=torch.long, device=device)


def _mv_covariates(z, act, mu, sd, device, known):
    """Targets (lanes 0..n_tgt-1) driven by covariates (the remaining active lanes).

    ROUND 3 (_COV_R3, off for the holdout; with it off this function draws exactly the random
    numbers it always did):
      look-alikes   35% of groups: the target and 60% of the other lanes get a periodic component
                    with the SAME period and different shapes. A covariate that follows the target's
                    cycle is then no evidence that it drives the target.
      weak drivers  35% of the relevant covariates have an effect of 0.03-0.3 instead of 0.3-2.0.
      joint effect  30% of groups: the nonlinearity is applied to the weighted SUM of the covariates.
      rare flags    see the gate below."""
    n, V, L = z.shape
    lane = torch.arange(V, device=device).unsqueeze(0)
    g = act.sum(1, keepdim=True)
    n_tgt = torch.where((torch.rand(n, 1, device=device) < 0.25) & (g >= 3), 2, 1)
    is_cov = act & (lane >= n_tgt)
    if _COV_R3:
        Psh = _pick_period(n, L, device).unsqueeze(-1)                             # (n, 1, 1)
        tL = torch.arange(L, device=device, dtype=torch.float32).view(1, 1, L)
        wave = torch.zeros(n, V, L, device=device)
        for k in (1, 2, 3):
            ang = 2 * math.pi * k * tL / Psh
            wave = wave + (torch.randn(n, V, 1, device=device) * torch.cos(ang)
                           + torch.randn(n, V, 1, device=device) * torch.sin(ang)) / k
        wave = wave / (wave.std(2, keepdim=True) + 1e-6)
        shared = act & ((lane < n_tgt) | (torch.rand(n, V, device=device) < 0.6)) & _bern(n, device, 0.35)
        z = z + (_u(n, V, device, 0.5, 2.0) * shared.float()).unsqueeze(-1) * wave
    raw, e, is_bin = _covariate_signal(z.reshape(n * V, L), device)
    raw, e, is_bin = raw.view(n, V, L), e.view(n, V, L), is_bin.view(n, V)
    fn = torch.randint(0, 5, (n, V, 1), device=device)       # effect shape per covariate
    vshape = (e - _u(n, V, device, -1.0, 1.0).unsqueeze(-1)).abs()        # high at both extremes
    ge = torch.where(fn == 0, e, torch.where(fn == 1, torch.relu(e), torch.where(
        fn == 2, torch.tanh(2.0 * e), torch.where(fn == 3, (e > 0.5).float(), vshape))))
    # lag of the effect: known covariates act (almost always) at once; past-only ones act later
    if known:
        d = torch.where(torch.rand(n, V, device=device) < 0.8, 0, torch.randint(1, 9, (n, V), device=device))
    else:
        d = torch.exp(_u(n, V, device, 0.0, math.log(96.0))).long()
    idx = (torch.arange(L, device=device).view(1, 1, L) - d.unsqueeze(-1)).clamp_(min=0)
    ge = ge.gather(2, idx)
    relevant = (is_cov & (torch.rand(n, V, device=device) < 0.6)).float()          # the rest are decoys
    sgn = torch.where(torch.rand(n, 2, V, device=device) < 0.5, 1.0, -1.0)
    beta = sgn * _u(n * 2, V, device, 0.3, 2.0).view(n, 2, V) * relevant.unsqueeze(1)
    if _COV_R3:
        weak = torch.rand(n, 1, V, device=device) < 0.35
        beta = torch.where(weak, sgn * torch.exp(_u(n * 2, V, device, math.log(0.03), math.log(0.3))).view(n, 2, V)
                           * relevant.unsqueeze(1), beta)
    eff = torch.bmm(beta, ge)                                                      # (n, 2, L)
    if _COV_R3:
        lin = torch.bmm(beta, e.gather(2, idx))                                    # weighted sum of the (lagged) covariates
        ls = lin.std(2, keepdim=True)
        s_ = lin / (ls + 1e-6)
        gk = torch.randint(0, 5, (n, 1, 1), device=device)
        G = torch.where(gk == 0, torch.relu(s_), torch.where(gk == 1, torch.tanh(1.5 * s_), torch.where(
            gk == 2, s_.abs(), torch.where(gk == 3, (s_ > 0.7).float(), torch.expm1(s_.clamp(max=3.0))))))
        G = (G - G.mean(2, keepdim=True)) / (G.std(2, keepdim=True) + 1e-6) * ls
        eff = torch.where(_bern(n, device, 0.3).unsqueeze(-1), G, eff)
    eff = F.pad(eff, (0, 0, 0, V - 2)) if V > 2 else eff[:, :V]
    is_tgt = (act & (lane < n_tgt)).unsqueeze(-1)
    # COVARIATE-DOMINANT targets: the target's own dynamics shrink to 5-30%, so it is almost a
    # pure function of its covariates (a price set by forecast load) -- history alone cannot
    # forecast it, the covariates' future values can.
    dom = _bern(n, device, _COV_DOMINANT_P) if _COV_DOMINANT_P > 0 else torch.zeros(n, 1, dtype=torch.bool, device=device)
    own = torch.where(dom, _u(n, 1, device, 0.05, 0.3), torch.ones(n, 1, device=device)).unsqueeze(-1)
    add = (z * own + eff) * sd + mu
    mult = (z * sd + mu) * (1.0 + 0.5 * torch.tanh(eff))
    tgt = torch.where((_bern(n, device, 0.7) | dom).unsqueeze(-1), add, mult)
    cov = torch.where(is_bin.unsqueeze(-1), raw, raw * sd + mu)
    if known:
        # ON/OFF GATE: the first covariate lane becomes a known 0/1 'open' flag and the targets
        # are exactly zero while it is off -- closed days at a fixed place in a cycle (a weekly
        # rest day) or irregular closed runs. No multiplicative effect above can produce this.
        t = torch.arange(L, device=device, dtype=torch.float32).unsqueeze(0)
        Pg = _pick(n, [7.0, 7.0, 24.0, 168.0], device)
        off_len = 1.0 + torch.floor(_u(n, 1, device, 0.0, 0.3) * Pg)
        periodic = torch.remainder(t + torch.floor(_u(n, 1, device, 0.0, 1.0) * Pg), Pg) < off_len
        runs = torch.gather(torch.rand(n, _MAX_SEG, device=device) < _u(n, 1, device, 0.05, 0.25), 1,
                            _segment_ids(n, L, device, 2.0, 12.0))
        if _COV_R3:
            # RARE SHUTDOWN FLAGS (round 3). Closed days are now one of: the weekly rest day, irregular
            # runs, RARE days only (0.3-4% of the steps, each lasting 1-3 steps: holidays), or the rest
            # day PLUS rare days. While closed the target is zero (75%) or a fraction of itself. The
            # flag is shown as 'open' (1 = normal) or as 'holiday' (1 = closed; for 30% of those a
            # code 1-3, like a holiday type), in ANY covariate lane, for 40% of the groups.
            rs = torch.rand(n, L, device=device) < torch.exp(_u(n, 1, device, math.log(0.003), math.log(0.04)))
            rl = torch.randint(1, 4, (n, 1), device=device)
            rare = rs.clone()
            rare[:, 1:] |= rs[:, :-1] & (rl >= 2)
            rare[:, 2:] |= rs[:, :-2] & (rl >= 3)
            gk2 = torch.randint(0, 4, (n, 1), device=device)
            closed = torch.where(gk2 == 0, periodic, torch.where(gk2 == 1, runs, torch.where(gk2 == 2, rare, periodic | rare)))
            keep = torch.where(_bern(n, device, 0.75), torch.zeros(n, 1, device=device), _u(n, 1, device, 0.05, 0.5))
            factor = torch.where(closed, keep, torch.ones(n, 1, device=device)).unsqueeze(1)          # (n, 1, L)
            code = (1 + torch.remainder(_segment_ids(n, L, device, 3.0, 20.0), 3)).float()
            shown = torch.where(_bern(n, device, 0.5), (~closed).float(),
                                closed.float() * torch.where(_bern(n, device, 0.3), code, torch.ones_like(code)))
            gated = _bern(n, device, 0.4) & (g > n_tgt)                                               # (n, 1)
            gl = n_tgt + torch.floor(torch.rand(n, 1, device=device) * (g - n_tgt).clamp(min=1).float()).long()
            gate_lane = (is_cov & (lane == gl) & gated).unsqueeze(-1)
            tgt = torch.where(gated.unsqueeze(-1), tgt * factor, tgt)
            cov = torch.where(gate_lane, shown.unsqueeze(1).expand(n, V, L), cov)
        else:
            open_f = (~torch.where(_bern(n, device, 0.6), periodic, runs)).float().unsqueeze(1)   # (n, 1, L)
            gated = _bern(n, device, 0.25) & (g > n_tgt)                                          # (n, 1)
            gate_lane = (is_cov & (lane == n_tgt) & gated).unsqueeze(-1)
            tgt = torch.where(gated.unsqueeze(-1), tgt * open_f, tgt)
            cov = torch.where(gate_lane, open_f.expand(n, V, L), cov)
    y = torch.where(is_tgt, tgt, cov)
    role = torch.where(is_cov, ROLE_KNOWN if known else ROLE_PAST, ROLE_TARGET)
    return y, role


def _mv_unrelated(z, act, mu, sd, device):
    n, V, L = z.shape
    role = torch.multinomial(torch.tensor([0.4, 0.3, 0.3]), n * V, replacement=True).view(n, V).to(device)
    role[:, 0] = ROLE_TARGET
    return z * sd + mu, role


def _group_pass(x, is_group, g, kind, device):
    """Turn lanes [0, g) of the blocks flagged is_group into ONE related group of the given kind.
    x (N, V, L) -> (out (N, V, L), role (N, V), act (N, V) = lanes that belong to the group)."""
    N, V, L = x.shape
    lane = torch.arange(V, device=device).unsqueeze(0).expand(N, V)
    role = torch.zeros(N, V, dtype=torch.long, device=device)
    act = is_group.unsqueeze(1) & (lane < g.unsqueeze(1))
    mu = x.mean(2, keepdim=True)
    sd = x.std(2, keepdim=True) + 1e-6
    out = x.clone()
    for k, name in enumerate(GROUP_KINDS):
        idx = (is_group & (kind == k)).nonzero().squeeze(1)
        if idx.numel() == 0:
            continue
        zk, ak, mk, sk = (x[idx] - mu[idx]) / sd[idx], act[idx], mu[idx], sd[idx]
        if name == "mix":
            y, r = _mv_mix(zk, ak, mk, sk, device)
        elif name == "leadlag":
            y, r = _mv_leadlag(zk, ak, mk, sk, device)
        elif name == "unrelated":
            y, r = _mv_unrelated(zk, ak, mk, sk, device)
        else:
            y, r = _mv_covariates(zk, ak, mk, sk, device, known=(name == "known_cov"))
        y = torch.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0).clamp(-_CLAMP, _CLAMP)
        out[idx] = torch.where(ak.unsqueeze(-1), y, x[idx])
        role[idx] = torch.where(ak, r, torch.zeros_like(r))
    return out, role, act


def _multivariatize(x, group_prob, device):
    """x (N, V, L) independent series -> (x, role (N, V), lgid (N, V)): the lanes of a related
    group share one lgid (0 for the group in lanes [0, g); V//2 for an optional second group in
    the upper half); every other lane keeps lgid = its lane index (a group of one)."""
    N, V, L = x.shape
    role = torch.zeros(N, V, dtype=torch.long, device=device)
    lane = torch.arange(V, device=device).unsqueeze(0).expand(N, V)
    if V < 2 or group_prob <= 0:
        return x, role, lane.clone()
    probs = torch.tensor(GROUP_KIND_PROBS)
    is_group = torch.rand(N, device=device) < group_prob
    g = torch.randint(2, V + 1, (N,), device=device)
    kind = torch.multinomial(probs, N, replacement=True).to(device)
    if _COV_SMALL_GROUP_P > 0:      # real covariate tasks have 1-4 covariates, not up to 7
        small = (torch.rand(N, device=device) < _COV_SMALL_GROUP_P) & ((kind == 2) | (kind == 3))
        g = torch.where(small, torch.randint(2, min(V, 4) + 1, (N,), device=device), g)
    out, role, act = _group_pass(x, is_group, g, kind, device)
    lgid = torch.where(act, torch.zeros_like(lane), lane)
    h = V // 2
    if _SECOND_GROUP_P > 0 and h >= 2:
        # a block has V lanes but a covariate group has one or two targets, so one group per block
        # caps covariate-driven targets at ~6% of all series. When lanes h..V-1 are free (no first
        # group, or one of size <= h) they can hold a second, independent group: the same routine
        # is run on the block rolled by h lanes.
        free = (~is_group) | (g <= h)
        is2 = free & (torch.rand(N, device=device) < _SECOND_GROUP_P)
        g2 = torch.randint(2, V - h + 1, (N,), device=device)
        kind2 = torch.multinomial(probs, N, replacement=True).to(device)
        o2, r2, a2 = _group_pass(out.roll(-h, dims=1), is2, g2, kind2, device)
        out, r2, a2 = o2.roll(h, dims=1), r2.roll(h, dims=1), a2.roll(h, dims=1)
        role = torch.where(a2, r2, role)
        lgid = torch.where(a2, torch.full_like(lane, h), lgid)
    return out, role, lgid


@torch.no_grad()
def generate_blocks(N, V, L, device, weights=None, profile="v1", group_prob=0.0):
    """N blocks of V aligned series of length L: x (N, V, L), role (N, V), lgid (N, V)."""
    # per-family rounding in _allocate can return a few rows more than asked for small batches
    x = generate_batch(N * V, L, device, weights, profile=profile)
    while x.shape[0] < N * V:
        x = torch.cat([x, generate_batch(N * V, L, device, weights, profile=profile)], 0)
    return _multivariatize(x[:N * V].view(N, V, L), group_prob, device)



# ============ 4. PACKING: variable-length sampling + row packing ============

def _loguniform_int(shape, lo, hi, gen=None):
    """Integers log-uniform on [lo, hi] (CPU)."""
    lo, hi = max(1, int(lo)), max(1, int(hi))
    if hi <= lo:
        return torch.full(shape, lo, dtype=torch.long)
    u = torch.rand(shape, generator=gen)
    v = torch.exp(math.log(lo) + u * (math.log(hi + 1) - math.log(lo)))
    return v.floor().long().clamp_(lo, hi)


class PackedBatcher:
    """Produces fixed-shape packed batches of variable-length series.

    Layout (sampled on CPU, cheap):
      Each of R rows is W = max_ctx points = P patches. Per row we draw up to M context
      lengths c (points), each occupying n = ceil(c / patch_len) patches. Segments are laid
      left-to-right; the first one that overflows is truncated to exactly fill the row, and
      if all M fit the last one is extended to fill it. So rows tile exactly -- no row-level
      padding at all. The only padding is the LEFT padding inside each series' first patch
      (n*patch_len - c points), which is marked mask=False.

    Content (generated on GPU):
      Each series is drawn from a per-bucket pool generated at length bucket_edge + pred_len,
      randomly cropped to c + pred_len, and scattered into its slot in the row.

    Shapes are identical every step (R, W), (R, P), (R*M, pred_len), (R*M,) -- so
    torch.compile never recompiles.
    """

    def __init__(self, cfg, device, weights=None, seed=None):
        self.pl = cfg["patch_len"]
        self.W = cfg["max_ctx"]
        assert self.W % self.pl == 0, "max_ctx must be a multiple of patch_len"
        self.P = self.W // self.pl
        self.R = cfg["pack_rows"]
        self.M = cfg["max_segs_per_row"]
        self.S = self.R * self.M
        self.decoder = cfg.get("decoder", "pool") == "causal"
        # group mode: rows come in BLOCKS of V lanes sharing one layout. The layout is sampled
        # for Rb = R / V block rows; cell (block row, slot) holds V aligned series.
        self.group = self.decoder and bool(cfg.get("group_attn", False))
        self.V = int(cfg.get("group_lanes", 8)) if self.group else 1
        self.group_prob = float(cfg.get("group_prob", 0.5)) if self.group else 0.0
        assert self.R % self.V == 0, "pack_rows must be a multiple of group_lanes"
        self.Rb = self.R // self.V
        self.Sb = self.Rb * self.M
        self.profile = resolve_profile(cfg)
        # causal decoder: a series is ONE sequence; every patch predicts the next, so no
        # separate target block is generated, and a series needs >= 2 patches to carry loss
        self.pred = 0 if self.decoder else cfg["pred_len"]
        self.min_ctx = max(1, int(cfg["min_ctx"]))
        if self.decoder:
            self.min_ctx = max(self.min_ctx, cfg["patch_len"] + 1)
        self.cpm_prob = float(cfg.get("cpm_prob", 0.5))
        self.cpm_max = int(cfg.get("cpm_max_patches", 16))
        self.cpm_min = int(cfg.get("cpm_min_patches", 1))
        self.cpm_suffix = float(cfg.get("cpm_suffix_frac", 0.5))
        self.cpm_long_p = float(cfg.get("cpm_long_p", 0.0) or 0.0)
        self.cpm_long_max = int(cfg.get("cpm_long_max", 32))
        self.long_frac = float(cfg["long_row_frac"])
        self.long_min = min(int(cfg["long_min_ctx"]), self.W)
        edges = sorted({min(int(e), self.W) for e in cfg["len_bucket_edges"]} | {self.W})
        edges = [e for e in edges if e >= self.min_ctx] or [self.W]
        self.edges = torch.tensor(edges, dtype=torch.long)
        self.pool_pts = float(cfg["gen_pool_points"])
        self.pool_max_rows = int(cfg["gen_pool_max_rows"])
        self.weights = weights
        self.device = device
        self.gen = torch.Generator().manual_seed(seed) if seed is not None else None
        self.pool = {}          # bucket index -> [buffer (N, Lb), read pointer]
        self.gen_seconds = 0.0

    # ---- layout ---------------------------------------------------------------
    def sample_layout(self):
        """Layout of the Rb BLOCK rows (every lane of a block uses the same one)."""
        R, M, P, pl = self.Rb, self.M, self.P, self.pl
        c = _loguniform_int((R, M), self.min_ctx, self.W, self.gen)
        long_rows = torch.rand(R, generator=self.gen) < self.long_frac
        c[:, 0] = torch.where(long_rows, _loguniform_int((R,), self.long_min, self.W, self.gen), c[:, 0])

        n = (c + pl - 1) // pl
        prev = n.cumsum(1) - n
        n_eff = torch.where(prev < P, torch.minimum(n, P - prev), torch.zeros_like(n))
        c_eff = torch.minimum(c, n_eff * pl)
        deficit = P - n_eff.sum(1)                  # > 0 only if all M fit with room left
        n_eff[:, -1] += deficit
        c_eff[:, -1] = torch.where(deficit > 0, n_eff[:, -1] * pl, c_eff[:, -1])
        start = n_eff.cumsum(1) - n_eff             # first patch of each segment
        valid = n_eff > 0

        seg_patch = torch.repeat_interleave(torch.arange(R * M), n_eff.reshape(-1)).view(R, P)
        vi = valid.reshape(-1).nonzero().squeeze(1)
        return dict(seg_patch=seg_patch, slot_valid=valid.reshape(-1), slot=vi,
                    row=vi // M, start=start.reshape(-1)[vi], n=n_eff.reshape(-1)[vi],
                    c=c_eff.reshape(-1)[vi])

    # ---- contiguous patch masking (causal decoder) ------------------------------
    def _cpm_blank(self, lay):
        """(R, W) bool: points hidden from the INPUT (still used as targets). For a fraction
        cpm_prob of series with >= 2 patches, one span of k in [1, cpm_max] patches is hidden,
        never the first patch; with prob cpm_suffix it is the series' LAST k patches -- exactly
        the inference situation of forecasting k patches past the end of the context."""
        R, P, pl, S = self.Rb, self.P, self.pl, self.Sb
        n = torch.zeros(S, dtype=torch.long).index_put_((lay["slot"],), lay["n"])
        st = torch.zeros(S, dtype=torch.long).index_put_((lay["slot"],), lay["start"])
        g = self.gen
        on = (torch.rand(S, generator=g) < self.cpm_prob) & (n >= 2)
        cmax, cmin = torch.full_like(n, self.cpm_max), torch.full_like(n, self.cpm_min)
        if self.cpm_long_p > 0:      # the extra random draw happens only when used: the holdout's stream is untouched
            half = torch.div(n - 1, 2, rounding_mode="floor")        # a long span never hides more than half a series
            lng = (torch.rand(S, generator=g) < self.cpm_long_p) & (half > self.cpm_max)
            cmax = torch.where(lng, torch.minimum(half, torch.full_like(n, self.cpm_long_max)), cmax)
            cmin = torch.where(lng, torch.full_like(n, self.cpm_max + 1), cmin)
        kmax = torch.minimum(n - 1, cmax).clamp(min=1)
        kmin = torch.minimum(kmax, cmin).clamp(min=1)
        k = kmin + (torch.rand(S, generator=g) * (kmax - kmin + 1).float()).floor().long()
        k = torch.minimum(k, kmax)
        suffix = torch.rand(S, generator=g) < self.cpm_suffix
        rnd = 1 + (torch.rand(S, generator=g) * (n - k).clamp(min=1).float()).floor().long()
        b0 = torch.where(suffix, n - k, torch.minimum(rnd, n - k))
        sp = lay["seg_patch"]
        rel = torch.arange(P).expand(R, P) - st[sp]
        blank_patch = on[sp] & (rel >= b0[sp]) & (rel < (b0 + k)[sp])
        return blank_patch.repeat_interleave(pl, dim=1)

    # ---- generation pools --------------------------------------------------------
    def _take(self, bi, k):
        """k blocks from bucket bi: x (k, V, Lb), role (k, V), lgid (k, V)."""
        E = int(self.edges[bi])
        Lb = E + self.pred
        V = self.V
        buf, ptr = self.pool.get(bi, (None, 0))
        if buf is None or buf[0].shape[0] - ptr < k:
            nb = max(k, -(-256 // V), min(self.pool_max_rows // V, int(self.pool_pts // (Lb * V))))
            t0 = time.time()
            buf = generate_blocks(nb, V, Lb, self.device, self.weights, profile=self.profile,
                                  group_prob=self.group_prob)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.gen_seconds += time.time() - t0
            ptr = 0
        out = tuple(t[ptr:ptr + k] for t in buf)
        self.pool[bi] = (buf, ptr + k)
        return out

    # ---- full batch -------------------------------------------------------------
    @torch.no_grad()
    def next_batch(self, layout=None):
        lay = layout or self.sample_layout()
        dev, W, pred, V, pl, P = self.device, self.W, self.pred, self.V, self.pl, self.P
        X = torch.zeros(self.R * W, device=dev)
        MK = torch.zeros(self.R * W, dtype=torch.bool, device=dev)
        Y = torch.zeros(self.S, pred, device=dev)
        lane = torch.arange(V, device=dev)
        ROLE = torch.zeros(self.Sb, V, dtype=torch.long, device=dev)
        LG = lane.expand(self.Sb, V).clone()

        c, row, start, n, slot = lay["c"], lay["row"], lay["start"], lay["n"], lay["slot"]
        # flat index of the first real point of lane v of each cell: row (block*V + v)
        first_pt = ((row.unsqueeze(1) * V + torch.arange(V).unsqueeze(0)) * W
                    + ((start + n) * pl - c).unsqueeze(1))                       # (K, V)
        bucket = torch.bucketize(c, self.edges)               # smallest edge >= c
        for bi in bucket.unique().tolist():
            sel = (bucket == bi).nonzero().squeeze(1)
            k = sel.numel()
            E = int(self.edges[bi])
            gx, grole, glg = self._take(bi, k)                 # (k, V, E + pred), (k, V), (k, V)
            cs = c[sel].to(dev)
            st = (torch.rand(k, device=dev) * (E - cs + 1).float()).floor().long()  # ONE crop per block
            j = torch.arange(E, device=dev).view(1, 1, E)
            vals = gx.gather(2, (st.view(k, 1, 1) + j).clamp_(max=E + pred - 1).expand(k, V, E))
            ok = (j < cs.view(k, 1, 1)).expand(k, V, E)
            dst = first_pt[sel].to(dev).unsqueeze(-1) + j
            X[dst[ok]] = vals[ok]
            MK[dst[ok]] = True
            sl = slot[sel].to(dev)
            ROLE[sl] = grole
            LG[sl] = glg
            if pred:
                tidx = (st + cs).view(k, 1, 1) + torch.arange(pred, device=dev).view(1, 1, pred)
                Y[(sl.unsqueeze(1) * V + lane.unsqueeze(0)).reshape(-1)] = \
                    gx.gather(2, tidx.expand(k, V, pred)).reshape(k * V, pred)

        n_ctx = int(c.sum()) * V; n_tgt = int(c.numel()) * V * pred
        sp = lay["seg_patch"].to(dev)                                            # (Rb, P) cell ids
        seg_full = (sp.unsqueeze(1) * V + lane.view(1, V, 1)).reshape(self.R, P)
        out = {}
        if self.decoder:
            role_p = ROLE[sp].permute(0, 2, 1).reshape(self.R, P)                # (R, P)
            gid = (sp.unsqueeze(1) * V + LG[sp].permute(0, 2, 1)).reshape(self.R, P)
            blank = self._cpm_blank(lay).to(dev).repeat_interleave(V, dim=0)     # same span for a cell
            blank = blank & ~(role_p == ROLE_KNOWN).repeat_interleave(pl, dim=1)  # known: never hidden
            out = dict(blank=blank, lanes=V, gid=gid, role=role_p,
                       slot_loss=(ROLE != ROLE_KNOWN).reshape(-1))
        return dict(out, x=X.view(self.R, W), mask=MK.view(self.R, W), y=Y,
                    seg_patch=seg_full, slot_valid=lay["slot_valid"].to(dev).repeat_interleave(V),
                    slot_ctx_len=torch.zeros(self.Sb, dtype=torch.long).index_put_(
                        (slot,), c).to(dev).repeat_interleave(V),
                    points=n_ctx + n_tgt, n_series=int(c.numel()) * V)


def packed_forward(model, b):
    if "blank" in b:            # causal decoder batch
        return model(b["x"], mask=b["mask"], seg_patch=b["seg_patch"],
                     slot_valid=b["slot_valid"], blank=b["blank"], lanes=b.get("lanes", 1),
                     gid=b.get("gid"), role=b.get("role"), slot_loss=b.get("slot_loss"))
    return model(b["x"], target=b["y"], mask=b["mask"],
                 seg_patch=b["seg_patch"], slot_valid=b["slot_valid"])


def layout_stats(batcher, n_layouts=20):
    """Realised length distribution of the sampler (CPU only, no generation)."""
    cs, per_row = [], []
    for _ in range(n_layouts):
        lay = batcher.sample_layout()
        cs.append(lay["c"])
        per_row.append(lay["slot_valid"].view(batcher.Rb, batcher.M).sum(1).float())
    c = torch.cat(cs).float()
    pr = torch.cat(per_row)
    pts_per_step = (float(c.sum()) / n_layouts + pr.sum().item() / n_layouts * batcher.pred) * batcher.V
    return c, pr, pts_per_step


# ================= 7. INFERENCE (left-padded, matches training) =================

@torch.no_grad()
def forecast_left_padded(model, series_list, horizon, device, max_ctx=8192, batch_size=256):
    """Point + quantile forecasts for a list of 1-D series, preprocessed EXACTLY like
    training: keep the last <= max_ctx observed points, LEFT-pad to the next patch boundary,
    mask the padding. Series are bucketed by padded length so no batch mixes lengths.
    horizon > pred_len -> autoregressive rollout on the median.

    Returns point (N, horizon), quant (N, n_q, horizon). Replace forecast_batch() in the
    fev eval with this -- the old flooring throws away data and silently falls back to
    last-value repeat for series shorter than one patch.
    """
    model.eval()
    pl, nq, q_mid = model.patch_len, model.n_q, model.n_q // 2
    is_dec = getattr(model, "is_decoder", False)
    N = len(series_list)
    point = np.zeros((N, horizon), dtype=np.float32)
    quant = np.zeros((N, nq, horizon), dtype=np.float32)
    amp_on = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_on and torch.cuda.is_bf16_supported()) else torch.float16

    def pad_left(y):
        y = y[-max_ctx:]
        Lp = max(pl, ((len(y) + pl - 1) // pl) * pl)
        v = np.zeros(Lp, dtype=np.float32); m = np.zeros(Lp, dtype=bool)
        v[Lp - len(y):] = y; m[Lp - len(y):] = True
        return v, m

    buckets = {}
    for i, y in enumerate(series_list):
        y = np.asarray(y, dtype=np.float32)
        y = y[np.isfinite(y)]
        if len(y) == 0:
            y = np.zeros(1, dtype=np.float32)
        v, m = pad_left(y)
        buckets.setdefault(len(v), []).append((i, v, m))

    for _, items in buckets.items():
        for c0 in range(0, len(items), batch_size):
            chunk = items[c0:c0 + batch_size]
            idx = [i for i, _, _ in chunk]
            cur = torch.from_numpy(np.stack([v for _, v, _ in chunk])).to(device)
            msk = torch.from_numpy(np.stack([m for _, _, m in chunk])).to(device)
            rem, qs = horizon, []
            while rem > 0:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_on):
                    if is_dec:   # all horizon patches from ONE pass over blanked patches
                        step = min(rem, model.max_pass_patches * pl)
                        qp = model.predict_horizon(cur, msk, step).float()
                    else:
                        qp = model(cur, mask=msk)["quantile_preds"].float()   # (b, pred, nq)
                take = min(qp.shape[1], rem)
                qs.append(qp[:, :take])
                rem -= take
                if rem > 0:     # roll forward on the median, re-left-pad to a patch boundary
                    med = qp[:, :take, q_mid]
                    cur = torch.cat([cur, med], 1); msk = torch.cat([msk, torch.ones_like(med, dtype=torch.bool)], 1)
                    keep = min(max_ctx, ((cur.shape[1] + pl - 1) // pl) * pl)
                    if cur.shape[1] >= keep:
                        cur, msk = cur[:, -keep:], msk[:, -keep:]
                    else:
                        padn = keep - cur.shape[1]
                        cur = torch.cat([torch.zeros(cur.shape[0], padn, device=device), cur], 1)
                        msk = torch.cat([torch.zeros(msk.shape[0], padn, dtype=torch.bool, device=device), msk], 1)
            qf = torch.cat(qs, 1).cpu().numpy()                             # (b, horizon, nq)
            for bi, i in enumerate(idx):
                quant[i] = qf[bi].T
                point[i] = qf[bi, :, q_mid]
    return point, quant

def _lane_array(a, n):
    """Last n points of a variate -> (values, observed). Leading NaNs stay unobserved; NaNs
    inside the series are forward-filled (lanes of a group must stay aligned in time, so
    missing points cannot be dropped the way the univariate path drops them)."""
    a = np.asarray(a, dtype=np.float32)[-n:]
    if len(a) < n:
        a = np.concatenate([np.full(n - len(a), np.nan, np.float32), a])
    ok = np.isfinite(a)
    if not ok.any():
        return np.zeros(n, np.float32), np.zeros(n, bool)
    idx = np.where(ok, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    first = int(np.argmax(ok))
    v = a[idx]
    obs = np.arange(n) >= first
    v[~obs] = 0.0
    return v.astype(np.float32), obs


@torch.no_grad()
def forecast_groups(model, groups, horizon, device, max_ctx=8192, rows=256, max_variates=None):
    """Forecast GROUPS with group attention, fully batched and packed exactly like training.

    groups: list of dicts, one per item:
        targets : list of 1-D arrays (the series to forecast; all get a forecast)
        past    : list of 1-D arrays (covariates observed only up to the forecast start)
        known   : list of (context 1-D array, future 1-D array of length `horizon`)
      All arrays of a group are aligned in time and end at the forecast start.
    Returns a list (one entry per group) of (point (n_targets, horizon),
    quant (n_targets, n_q, horizon)).

    A group never exceeds `max_variates` lanes (default: the largest group seen in training):
    covariates are kept first (known before past, at most max-1 with one target, max//2 with
    several), and the targets are split into chunks that each see those covariates.
    A group with a single variate takes the ordinary univariate path (NaNs dropped)."""
    model.eval()
    pl, nq, q_mid = model.patch_len, model.n_q, model.n_q // 2
    hp = -(-horizon // pl)
    cap = int(max_variates or getattr(model, "group_lanes", 8))
    res = [None] * len(groups)
    if hp > getattr(model, "max_pass_patches", 32) or not getattr(model, "group_attn", False):
        for gi, gr in enumerate(groups):      # rollout / univariate model: targets one by one
            res[gi] = forecast_left_padded(model, gr["targets"], horizon, device, max_ctx=max_ctx)
        return res

    # ---- units: (group, target indices, lanes) with lanes = list of (values, observed, role)
    units = []
    for gi, gr in enumerate(groups):
        tg, past, known = gr["targets"], gr.get("past", []), gr.get("known", [])
        nt = len(tg)
        res[gi] = (np.zeros((nt, horizon), np.float32), np.zeros((nt, nq, horizon), np.float32))
        n_cov = min(len(known) + len(past), cap - 1 if nt == 1 else cap // 2)
        kn, ps = known[:n_cov], past[:max(0, n_cov - len(known))]
        if nt == 1 and n_cov == 0:            # plain univariate: drop NaNs, as in training
            y = np.asarray(tg[0], dtype=np.float32); y = y[np.isfinite(y)]
            y = (y if len(y) else np.zeros(1, np.float32))[-max_ctx:]
            units.append((gi, [0], [(y, np.ones(len(y), bool), ROLE_TARGET, None)], len(y)))
            continue
        Lc = int(min(max_ctx, max(len(np.asarray(t)) for t in tg)))
        cov_lanes = []
        for cx, fut in kn:
            v, o = _lane_array(cx, Lc)
            f = np.asarray(fut, dtype=np.float32)[:horizon]
            fo = np.isfinite(f)
            if (~fo).any():                   # forward-fill the known future from the last value
                last = v[-1] if o[-1] else 0.0
                for i in range(len(f)):
                    if fo[i]:
                        last = f[i]
                    else:
                        f[i] = last
            cov_lanes.append((v, o, ROLE_KNOWN, f))
        for cx in ps:
            v, o = _lane_array(cx, Lc)
            cov_lanes.append((v, o, ROLE_PAST, None))
        per = max(1, cap - len(cov_lanes))
        for c0 in range(0, nt, per):
            ids = list(range(c0, min(nt, c0 + per)))
            lanes = [(*_lane_array(tg[i], Lc), ROLE_TARGET, None) for i in ids] + cov_lanes
            units.append((gi, ids, lanes, Lc))

    amp_on = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_on and torch.cuda.is_bf16_supported()) else torch.float16
    by_v = {}
    for u in units:
        by_v.setdefault(len(u[2]), []).append(u)
    for V, us in by_v.items():
        nc = np.array([max(1, -(-u[3] // pl)) for u in us])
        need = nc + hp
        P = int(need.max())
        order = np.argsort(-need, kind="stable")       # first-fit-decreasing into block rows
        brow, bstart, fill = np.zeros(len(us), int), np.zeros(len(us), int), []
        # only rows that can still take the SMALLEST unit are kept in `open_rows`; the old loop
        # rescanned every row for every series (quadratic in the number of series), which made
        # short-context batches slower than Chronos-2 for no reason
        min_need, open_rows = int(need.min()), []
        for i in order:
            for r in open_rows:
                if fill[r] + need[i] <= P:
                    brow[i], bstart[i] = r, fill[r]; fill[r] += need[i]
                    if P - fill[r] < min_need:
                        open_rows.remove(r)
                    break
            else:
                brow[i], bstart[i] = len(fill), 0; fill.append(int(need[i]))
                if P - need[i] >= min_need:
                    open_rows.append(len(fill) - 1)
        nb_all, nb = len(fill), max(1, rows // V)
        for b0 in range(0, nb_all, nb):
            sel = np.nonzero((brow >= b0) & (brow < b0 + nb))[0]
            Bk = int(min(nb, nb_all - b0)); R = Bk * V; W = P * pl
            x = np.zeros((R, W), np.float32); obs = np.zeros((R, W), bool)
            seg = np.full((R, P), -1, np.int64); gid = np.full((R, P), -1, np.int64)
            role = np.zeros((R, P), np.int64)
            for k, i in enumerate(sel):
                _, _, lanes, Lc = us[i]
                st, b = bstart[i], brow[i] - b0
                e_ctx = (st + nc[i]) * pl              # end of context / start of horizon
                for v, (val, o, rl, fut) in enumerate(lanes):
                    r = b * V + v
                    x[r, e_ctx - Lc:e_ctx] = val; obs[r, e_ctx - Lc:e_ctx] = o
                    if rl == ROLE_KNOWN:               # known future values fill the horizon patches
                        x[r, e_ctx:e_ctx + len(fut)] = fut; obs[r, e_ctx:e_ctx + len(fut)] = True
                    seg[r, st:st + need[i]] = k * V + v
                    gid[r, st:st + need[i]] = k
                    role[r, st:st + need[i]] = rl
            # unused tails: one dummy segment (and group) per row
            dummy = len(sel) * V + np.arange(R)[:, None].repeat(P, 1)
            gid = np.where(seg < 0, dummy, gid); seg = np.where(seg < 0, dummy, seg)
            S = len(sel) * V + R
            xt, ot = torch.from_numpy(x).to(device), torch.from_numpy(obs).to(device)
            sg = torch.from_numpy(seg).to(device)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_on):
                seg_pt, seg_start_pt, attn = model._packed_geometry(sg, S)
                q, mu_e, sd_e, _ = model._core(xt, ot, seg_pt, seg_start_pt, S, attn_mask=attn, lanes=V,
                                               gid=torch.from_numpy(gid).to(device),
                                               role=torch.from_numpy(role).to(device))
            allq = (model._inv_tf(q.float()) * sd_e[..., None, None] + mu_e[..., None, None]).cpu().numpy()
            for i in sel:
                gi, ids, lanes, _ = us[i]
                p_last = bstart[i] + nc[i] - 1
                for v, ti in enumerate(ids):
                    f = allq[(brow[i] - b0) * V + v, p_last:p_last + hp].reshape(hp * pl, nq)[:horizon]
                    res[gi][1][ti] = f.T; res[gi][0][ti] = f[:, q_mid]
    return res



# ======================= equivalence check =======================

@torch.no_grad()
def _verify_decoder(model, b, cfg, n_check, tol):
    """Causal decoder, per block cell (V aligned lanes):
      (1) packed == alone: the cell's forecasts at EVERY position inside the packed batch equal
          those of the same V lanes run on their own;
      (2) causality: changing every lane's future never changes an earlier forecast (a known
          covariate may see exactly one patch ahead -- that is its job -- and no further);
      (3) group isolation: a lane that is a group of one gives the same forecasts when run
          completely alone -- other lanes of the block cannot leak into it."""
    out = packed_forward(model, b)
    allq = model._inv_tf(out["_q"]) * out["_sd_e"][..., None, None] + out["_mu_e"][..., None, None]
    pl, M, V = cfg["patch_len"], cfg["max_segs_per_row"], b.get("lanes", 1)
    cells = b["slot_valid"].view(-1, V)[:, 0].nonzero().squeeze(1)
    cells = cells[torch.randperm(cells.numel(), device=cells.device)[:max(4, n_check // V)]]
    obs_in = b["mask"] & ~b["blank"]
    gid_all, role_all = b.get("gid"), b.get("role")

    def rel(a, r):
        return ((a - r).abs().amax(dim=(1, 2, 3)) / (r.abs().amax(dim=(1, 2, 3)) + 1e-6)).max().item()

    worst = worst_causal = worst_iso = 0.0
    n_series = n_grouped = 0
    for cell in cells.tolist():
        r0 = (cell // M) * V
        cols = (b["seg_patch"][r0] == cell * V).nonzero().squeeze(1)
        c0, c1 = int(cols[0]), int(cols[-1]) + 1
        xs, ms = b["x"][r0:r0 + V, c0 * pl:c1 * pl], obs_in[r0:r0 + V, c0 * pl:c1 * pl]
        gs = gid_all[r0:r0 + V, c0:c1] if gid_all is not None else None
        rs = role_all[r0:r0 + V, c0:c1] if role_all is not None else None
        alone = model(xs, mask=ms, lanes=V, gid=gs, role=rs)["quantile_preds_all"]
        ref = allq[r0:r0 + V, c0:c1]
        worst = max(worst, rel(alone, ref))
        n_series += V
        if c1 - c0 >= 3:                       # perturb the future, earlier outputs must not move
            cut = (c1 - c0) // 2
            xp = xs.clone(); xp[:, (cut + 1) * pl:] += 1e3 * torch.randn_like(xp[:, (cut + 1) * pl:])
            if rs is not None:                 # a known covariate's NEXT patch is visible by design
                kn = rs[:, 0] == ROLE_KNOWN
                xp[kn, (cut + 1) * pl:(cut + 2) * pl] = xs[kn, (cut + 1) * pl:(cut + 2) * pl]
            pert = model(xp, mask=ms, lanes=V, gid=gs, role=rs)["quantile_preds_all"]
            worst_causal = max(worst_causal, rel(pert[:, :cut + 1], alone[:, :cut + 1]))
        if gs is not None:
            g0 = gs[:, 0]
            single = (g0.unsqueeze(0) == g0.unsqueeze(1)).sum(1) == 1
            n_grouped += int((~single).sum())
            for v in single.nonzero().squeeze(1).tolist()[:2]:
                solo = model(xs[v:v + 1], mask=ms[v:v + 1], lanes=1, gid=gs[v:v + 1],
                             role=rs[v:v + 1])["quantile_preds_all"]
                worst_iso = max(worst_iso, rel(solo, ref[v:v + 1]))
    ok = worst < tol and worst_causal < tol and worst_iso < tol
    print(f"causal-decoder packing equivalence: {n_series} series ({n_grouped} in related groups), "
          f"worst rel diff {worst:.2e}; causality (future perturbed) worst {worst_causal:.2e}; "
          f"group isolation worst {worst_iso:.2e} -> {'PASS' if ok else 'FAIL'}")
    return ok

@torch.no_grad()
def verify_packing_equivalence(cfg, device, n_check=64, tol=2e-3):
    """Every series forecast inside a packed row must equal the same series forecast
    alone (left-padded, unpacked). Runs in fp32 with dropout off. If this fails, something
    leaks across series boundaries and packed training is invalid."""
    lanes = int(cfg.get("group_lanes", 8)) if (cfg.get("group_attn") and cfg.get("decoder") == "causal") else 1
    small = dict(cfg, pack_rows=min(cfg["pack_rows"], max(8, 2 * lanes)))
    torch.manual_seed(123)
    model = build_model(small).to(device).eval()
    if getattr(model, "group_attn", False):
        # make the group pathway and the role embeddings LARGE here, so any leak between
        # series, lanes or groups shows up clearly in the check
        for blk in model.blocks:
            nn.init.normal_(blk.group.out_proj.weight, std=0.05)
        nn.init.normal_(model.role_emb.weight, std=0.5)
    b = PackedBatcher(small, device, seed=7).next_batch()
    if "blank" in b:
        return _verify_decoder(model, b, small, n_check, tol)
    packed = packed_forward(model, b)["quantile_preds"]
    slots = b["slot_valid"].nonzero().squeeze(1)
    slots = slots[torch.randperm(slots.numel(), device=slots.device)[:n_check]]
    pl, W = small["patch_len"], small["max_ctx"]
    worst = 0.0
    for s in slots.tolist():
        r = s // small["max_segs_per_row"]
        cols = (b["seg_patch"][r] == s).nonzero().squeeze(1)
        p0, p1 = int(cols[0]) * pl, (int(cols[-1]) + 1) * pl
        alone = model(b["x"][r:r + 1, p0:p1], mask=b["mask"][r:r + 1, p0:p1])["quantile_preds"][0]
        ref = packed[s]
        rel = ((alone - ref).abs().max() / (ref.abs().max() + 1e-6)).item()
        worst = max(worst, rel)
    ok = worst < tol
    print(f"packing equivalence: {len(slots)} series checked, worst relative diff = {worst:.2e} "
          f"-> {'PASS' if ok else 'FAIL'}")
    return ok


# ======================= 5. INSPECTION ==============================

def _acf_at(xc, den, lag):
    return (xc[:, lag:] * xc[:, :-lag]).sum(1) / den


def seasonal_strength(x, periods=tuple(int(p) for p in PERIOD_BANK)):
    """Local-PEAK detection of the autocorrelation at lag P.

    Two simpler measures were tried and both failed, in opposite directions:
      raw acf(P)            -> a random walk has high acf at EVERY lag, so ~87% of any
                               persistent corpus reads as "seasonal" (false positives)
      acf(P) on diff(x)     -> differencing scales a period-P cycle by ~2*pi/P but white
                               noise by a constant, so long periods are annihilated: a
                               clean period-48 sine with 30% noise scored 0.05 (false negs)
    Peak-vs-neighbours is robust to both: generic persistence gives a smooth, locally
    linear acf (difference ~0) while a real cycle gives a local maximum.
    """
    xc = x - x.mean(1, keepdim=True)
    den = (xc ** 2).sum(1) + 1e-9
    L = x.shape[1]
    best = torch.zeros(x.shape[0], device=x.device)
    for p in periods:
        d = max(1, int(round(p / 6.0)))
        if L <= 2 * p + d + 2:
            continue
        peak = _acf_at(xc, den, p) - 0.5 * (_acf_at(xc, den, p - d) + _acf_at(xc, den, p + d))
        best = torch.maximum(best, peak)
    return best


def spectral_slope(x):
    """Slope of log power vs log frequency: ~0 white, ~-1 pink, ~-2 brown/random-walk."""
    B, L = x.shape
    xc = x - x.mean(1, keepdim=True)
    P = (torch.fft.rfft(xc, dim=1).abs() ** 2)[:, 1:]
    f = torch.fft.rfftfreq(L)[1:].to(x.device)
    lf = torch.log(f + 1e-12).unsqueeze(0)
    lp = torch.log(P + 1e-12)
    lfc = lf - lf.mean(1, keepdim=True)
    lpc = lp - lp.mean(1, keepdim=True)
    return (lfc * lpc).sum(1) / ((lfc ** 2).sum(1) + 1e-9)


def inspect(n=512, L=1024, plot=True, profile="v1"):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = generate_batch(n, L, dev, profile=profile)
    sd = x.std(dim=1)
    d = x.diff(dim=1)
    dz = (d - d.mean(1, keepdim=True)) / (d.std(1, keepdim=True) + 1e-9)
    kurt = (dz ** 4).mean(dim=1) - 3.0
    h = L // 2
    vr = (x[:, h:].std(1) + 1e-9) / (x[:, :h].std(1) + 1e-9)
    ls_ = (x[:, h:].mean(1) - x[:, :h].mean(1)).abs() / (sd + 1e-9)
    zfrac = (x == 0).float().mean(dim=1)
    is_int = ((x - x.round()).abs().max(dim=1).values < 1e-6)
    # repeated NON-ZERO values = forward-fill imputation artifact. Repeated zeros are
    # structural (solar night, zero-floor demand) and integer data repeats by discreteness,
    # so both are excluded or this metric massively overstates the artifact rate.
    flat = ((d.abs() < 1e-12) & (x[:, 1:].abs() > 1e-12)).float().mean(dim=1)
    flat = torch.where(is_int, torch.zeros_like(flat), flat)
    acf1 = _acf_at(x - x.mean(1, keepdim=True), ((x - x.mean(1, keepdim=True)) ** 2).sum(1) + 1e-9, 1)

    rows = [
        ("integer-valued (counts)",          is_int.float().mean(),            "~5-15% in Monash/GIFT"),
        (">30% exact zeros (intermittent)",  (zfrac > .3).float().mean(),      "~5-15% (car_parts, solar night)"),
        ("strictly non-negative",            (x.min(1).values >= 0).float().mean(), "~50-70%"),
        ("excess kurtosis > 3 (fat tails)",  (kurt > 3).float().mean(),        "high in financial/spiky"),
        ("var ratio >2 or <0.5 (heterosked)",((vr > 2) | (vr < .5)).float().mean(), "GP samples give ~0%"),
        ("level shift > 1 sd",               (ls_ > 1).float().mean(),         "GP samples give ~0%"),
        (">10% flat-lined (imputed)",        (flat > .1).float().mean(),       "the *_with_missing sets"),
        ("|lag-1 acf| > 0.9 (persistent)",   (acf1.abs() > .9).float().mean(), "most real series"),
        ("seasonal strength > 0.3",          (seasonal_strength(x) > .3).float().mean(), "~40-60%"),
        ("spectral slope < -1 (non-white)",  (spectral_slope(x) < -1).float().mean(),   "should dominate"),
    ]
    print(f"=== synthetic corpus audit (profile={profile}, n={n}, len={L}) ===")
    print(f"{'property':36s} {'fraction':>9s}   reference")
    print("-" * 88)
    for name, v, ref in rows:
        print(f"{name:36s} {float(v):9.3f}   {ref}")
    print("-" * 88)
    lsd = torch.log10(sd.clamp(min=1e-9))
    print(f"{'log10 scale spread (p5..p95)':36s} {float(lsd.quantile(.05)):5.2f} .. {float(lsd.quantile(.95)):.2f}")
    print(f"{'median spectral slope':36s} {float(spectral_slope(x).median()):9.2f}   -1 pink / -2 random-walk")
    print(f"{'degenerate (zero-var) rows':36s} {int((sd < 1e-6).sum()):9d}   must be 0")
    print(f"{'non-finite values':36s} {int((~torch.isfinite(x)).sum()):9d}   must be 0")

    if plot:
        import matplotlib.pyplot as plt
        k = 24
        xs = x[torch.randperm(x.shape[0])[:k]].float().cpu().numpy()
        fig, axes = plt.subplots(6, 4, figsize=(18, 12))
        for i, ax in enumerate(axes.ravel()):
            ax.plot(xs[i], lw=.8); ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle("generate_batch() -- random sample", fontsize=14)
        fig.tight_layout(); plt.show()
    return x


def inspect_packing(cfg, n_layouts=20):
    """What the sampler actually produces: series per row, context-length distribution by
    COUNT and by POINTS (points = where the compute goes), and fill efficiency."""
    bt = PackedBatcher(cfg, torch.device("cpu"))
    c, pr, pps = layout_stats(bt, n_layouts)
    bands = [(1, 32), (33, 128), (129, 512), (513, 2048), (2049, 8192)]
    print(f"=== packing layout audit ({n_layouts} layouts x {bt.R} rows, row = {bt.W} pts) ===")
    print(f"series per row: mean {pr.mean():.1f}  min {int(pr.min())}  max {int(pr.max())}  "
          f"(cap {bt.M})   series per step ~{pr.sum()/n_layouts*bt.V:,.0f}")
    pad = ((c / bt.pl).ceil() * bt.pl - c).sum() / (bt.Rb * bt.W * n_layouts)
    print(f"left-padding share of row points: {float(pad):.2%}   (only padding in the batch)")
    if bt.V > 1:
        adev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        b = PackedBatcher(dict(cfg, pack_rows=min(cfg["pack_rows"], 8 * bt.V), gen_pool_points=5e5,
                               gen_pool_max_rows=1024), adev, seed=5).next_batch()
        b = {k: (t.cpu() if torch.is_tensor(t) else t) for k, t in b.items()}
        v = b["slot_valid"]
        role, gid = b["role"], b["gid"]
        first = torch.zeros(b["slot_valid"].numel(), dtype=torch.long).scatter_(
            0, b["seg_patch"].reshape(-1), gid.reshape(-1))                 # group id of every series
        size = torch.bincount(first[v])[first[v]]
        srole = torch.zeros_like(first).scatter_(0, b["seg_patch"].reshape(-1), role.reshape(-1))[v]
        print(f"groups: blocks of {bt.V} lanes; {float((size > 1).float().mean()):.0%} of series sit in a related "
              f"group (mean size {float(size[size > 1].float().mean()):.1f}); roles: "
              f"target {float((srole == ROLE_TARGET).float().mean()):.0%}  past covariate "
              f"{float((srole == ROLE_PAST).float().mean()):.0%}  known covariate "
              f"{float((srole == ROLE_KNOWN).float().mean()):.0%}")
    print(f"real points per step ~{human(pps)} (context + targets)")
    print(f"{'context length':>16s} {'% of series':>12s} {'% of ctx points':>16s}")
    tot = c.sum()
    for lo, hi in bands:
        m = (c >= lo) & (c <= hi)
        print(f"{f'{lo}-{hi}':>16s} {m.float().mean():12.1%} {(c[m].sum()/tot):16.1%}")


# ======================= 6. TRAINING ================================

def human(n):
    for u in ["", "K", "M", "B", "T"]:
        if abs(n) < 1000:
            return f"{n:.2f}{u}"
        n /= 1000
    return f"{n:.2f}P"


def build_val_batches(cfg, device, n_batches, seed=1234):
    """Fixed packed holdout, identical every eval and across runs. Uses its own RNG seed
    and its own generation pools, and restores the global RNG so it never perturbs the
    training stream."""
    st = torch.random.get_rng_state()
    cst = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    torch.manual_seed(seed)
    vcfg = dict(cfg, corpus=cfg.get("val_corpus") or cfg.get("corpus", "v1"),
                group_prob=cfg.get("val_group_prob", cfg.get("group_prob", 0.5)), cpm_long_p=0.0)
    bt = PackedBatcher(vcfg, device, seed=seed)
    batches = [bt.next_batch() for _ in range(n_batches)]
    torch.random.set_rng_state(st)
    if cst is not None:
        torch.cuda.set_rng_state_all(cst)
    return batches


VAL_BANDS = [(1, 64), (65, 512), (513, 2048), (2049, 10 ** 9)]


@torch.no_grad()
def evaluate_packed(model, batches, amp_dtype, amp_on, device):
    """Overall holdout loss + loss by context-length band, so a regression on short or
    long series is visible instead of averaged away."""
    model.eval()
    losses, lens = [], []
    for b in batches:
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_on):
            out = packed_forward(model, b)
        v = b["slot_valid"]
        losses.append(out["per_series_loss"][v].float())
        lens.append(b["slot_ctx_len"][v])
    model.train()
    l, c = torch.cat(losses), torch.cat(lens)
    fin = torch.isfinite(l)
    l, c = l[fin], c[fin]
    bands = {}
    for lo, hi in VAL_BANDS:
        m = (c >= lo) & (c <= hi)
        bands[f"{lo}-{hi if hi < 10**9 else 'max'}"] = float(l[m].mean()) if m.any() else float("nan")
    return float(l.mean()) if l.numel() else float("nan"), bands


@torch.no_grad()
def group_gain(model, batches, amp_dtype, amp_on, device):
    """How much the model USES the other variates: mean loss of the series that sit in a related
    group, with group attention ON vs CUT (same weights, same data, every lane made a group of
    one). cut > on means the model is getting information from sibling targets / covariates.
    Returns (loss_on, loss_cut), or (nan, nan) for a univariate model."""
    model.eval()
    on, cut = [], []
    for b in batches:
        if b.get("lanes", 1) == 1 or "gid" not in b:
            continue
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_on):
            l_on = packed_forward(model, b)["per_series_loss"].float()
            l_cut = packed_forward(model, dict(b, gid=b["seg_patch"]))["per_series_loss"].float()
        S = b["slot_valid"].numel()
        g = torch.zeros(S, dtype=torch.long, device=device).scatter_(0, b["seg_patch"].reshape(-1), b["gid"].reshape(-1))
        v = b["slot_valid"]
        size = torch.zeros(int(g.max()) + 1, device=device).index_add_(0, g[v], torch.ones(int(v.sum()), device=device))
        m = v & (size[g] > 1) & torch.isfinite(l_on) & torch.isfinite(l_cut)
        on.append(l_on[m]); cut.append(l_cut[m])
    model.train()
    if not on or sum(t.numel() for t in on) == 0:
        return float("nan"), float("nan")
    return float(torch.cat(on).mean()), float(torch.cat(cut).mean())


@torch.no_grad()
def horizon_split(model, batches, amp_dtype, amp_on, device):
    """Holdout loss of forecasts made from real history (one-patch) vs from hidden placeholder
    patches (multi-patch: the 2nd, 3rd, ... patch of a horizon). Returns (one, multi)."""
    model.eval()
    one, multi = [], []
    for b in batches:
        if "blank" not in b:
            continue
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_on):
            out = packed_forward(model, b)
        one.append(out["_per_one"].float()); multi.append(out["_per_multi"].float())
    model.train()

    def mean(xs):
        if not xs:
            return float("nan")
        v = torch.cat(xs)
        v = v[torch.isfinite(v)]
        return float(v.mean()) if v.numel() else float("nan")
    return mean(one), mean(multi)


def group_probe(cfg, steps=3000, rows=None, every=250):
    """SHORT THROWAWAY RUN (nothing is saved) that answers one question before a long run:
    does the model start to USE the other variates of a group? Prints, every `every` steps,
    the loss of grouped series with group attention on vs cut. A gap that opens and keeps
    growing means the group pathway is being learned. Uses the real model and the real data,
    with fewer rows per step so it takes minutes."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    V = int(cfg.get("group_lanes", 8))
    c = dict(cfg, pack_rows=rows or min(cfg["pack_rows"], 16 * V), compile_model=False)
    amp_on = device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_on and torch.cuda.is_bf16_supported()) else torch.float16
    torch.manual_seed(c["seed"])
    model = build_model(c).to(device).train()
    batcher = PackedBatcher(c, device, seed=c["seed"] + 1)
    val = build_val_batches(dict(c, val_corpus=None), device, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=c["lr"], fused=(device.type == "cuda"))
    print(f"group probe: {sum(p.numel() for p in model.parameters())/1e6:.1f}M params, {c['pack_rows']} rows/step, "
          f"{steps} steps, lanes={V}, group_prob={c.get('group_prob')}, corpus={c.get('corpus')}")
    t0, pts, lw, nw = time.time(), 0.0, 0.0, 0
    for step in range(1, steps + 1):
        for gp in opt.param_groups:
            gp["lr"] = c["lr"] * min(1.0, step / 100)
        b = batcher.next_batch()
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_on):
            loss = packed_forward(model, b)["loss"]
        if not torch.isfinite(loss):
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c["grad_clip"])
        opt.step()
        pts += b["points"]; lw += loss.item(); nw += 1
        if step % every == 0 or step == steps:
            g_on, g_cut = group_gain(model, val, amp_dtype, amp_on, device)
            print(f"  step {step:5d}  train loss {lw/max(nw,1):.4f}  |  grouped series: loss with groups ON "
                  f"{g_on:.4f}  CUT {g_cut:.4f}  -> gain {(g_cut - g_on) / max(g_cut, 1e-9):+.1%}  |  "
                  f"{pts/max(time.time()-t0,1e-9)/1e6:.2f}M pts/s  {human(pts)} pts", flush=True)
            lw, nw = 0.0, 0
    return model


# ======================= 5b. REPEAT PROBE ==============================
# Noise-free exactly repeating series; score = share of the pattern the forecast captures:
# 1 - squared error / squared error of a flat line at the context mean (1 = perfect, 0 = flat
# line, below 0 = worse; clipped at -1). "list" = periods in the generator's period bank.
PROBE_PERIODS = [7, 10, 12, 16, 24, 32, 48, 50, 64, 70, 96, 100, 120, 128, 168, 200]
PROBE_SHAPES = ["sine", "sawtooth", "square", "spike", "template"]
PROBE_CTX, PROBE_H = 2048, 96
LAST_VAL = {}          # run name -> metrics of its latest validation


def _probe_set(n_per, seed=0):
    rng = np.random.default_rng(seed)
    meta, ctx, fut = [], [], []
    for shape in PROBE_SHAPES:
        for P in PROBE_PERIODS:
            for _ in range(n_per):
                t = np.arange(PROBE_CTX + PROBE_H) + int(rng.integers(0, P))
                u = (t % P) / P
                if shape == "sine":
                    y = np.sin(2 * np.pi * u)
                elif shape == "sawtooth":
                    y = 2 * u - 1
                elif shape == "square":
                    y = np.where(u < rng.uniform(0.2, 0.5), 1.0, -1.0)
                elif shape == "spike":
                    y = np.where(t % P == 0, 1.0, 0.0)
                else:
                    w = rng.standard_normal(P); w = (w - w.mean()) / (np.abs(w).max() + 1e-9)
                    y = w[t % P]
                s = rng.uniform(0.5, 5.0) * y + rng.normal(0.0, 3.0)
                ctx.append(s[:PROBE_CTX].astype(np.float32)); fut.append(s[PROBE_CTX:]); meta.append((shape, P))
    return meta, ctx, np.asarray(fut)


@torch.no_grad()
def repeat_probe(model, device, n_per=4, table=False):
    """-> dict(list=, other=, by_shape={shape: (list, other)}). Leaves the model in eval mode."""
    meta, ctx, fut = _probe_set(n_per)
    med, _ = forecast_left_padded(model, ctx, PROBE_H, device, max_ctx=PROBE_CTX)
    flat = np.array([c.mean() for c in ctx])[:, None]
    r = np.clip(1.0 - ((med - fut) ** 2).sum(1) / np.maximum(((flat - fut) ** 2).sum(1), 1e-12), -1.0, 1.0)
    r[fut.std(1) < 1e-9] = np.nan
    shp = np.array([m[0] for m in meta]); per = np.array([m[1] for m in meta])
    inl = np.isin(per, PERIOD_BANK_EXT)
    out = dict(list=float(np.nanmean(r[inl])), other=float(np.nanmean(r[~inl])),
               by_shape={s: (float(np.nanmean(r[(shp == s) & inl])), float(np.nanmean(r[(shp == s) & ~inl])))
                         for s in PROBE_SHAPES})
    if table:
        cell = lambda m: f"{np.nanmean(r[m]):9.2f}" if np.isfinite(r[m]).any() else f"{'-':>9s}"
        print(f"{'period':>7s} {'in list':>8s} " + " ".join(f"{s:>9s}" for s in PROBE_SHAPES))
        for P in PROBE_PERIODS:
            print(f"{P:7d} {'yes' if P in PERIOD_BANK_EXT else '-':>8s} "
                  + " ".join(cell((shp == s) & (per == P)) for s in PROBE_SHAPES))
    return out


def probe_line(pr):
    return (f"repeat probe (1 = perfect): generator's periods {pr['list']:.2f} | other periods {pr['other']:.2f}   ["
            + "  ".join(f"{s} {a:.2f}/{b:.2f}" for s, (a, b) in pr["by_shape"].items()) + "]")


WEEK_PPD = [1, 12, 24, 48, 96]        # steps per day (1 = daily data with a day-of-week pattern)
WEEK_KINDS = ["peaks", "template"]


def _week_set(n_per, noise=0.0, seed=0):
    """Noise-free weekday / weekend series. The context ends on a Thursday night; the horizon is
    Friday, Saturday, Sunday (a whole week for daily data). -> per steps-per-day: contexts, the
    clean future, a mask of the weekend points, and the kind of each series."""
    rng = np.random.default_rng(seed)
    out = {}
    for ppd in WEEK_PPD:
        weeks = {1: 52, 12: 24, 24: 12, 48: 6, 96: 6}[ppd]
        H = 7 if ppd == 1 else 3 * ppd
        n_ctx = (weeks * 7 + 4) * ppd
        t = np.arange(n_ctx + H)
        day = t // ppd; tod = t % ppd; dow = day % 7
        u = (tod + 0.5) / ppd
        ctx, fut, kinds = [], [], []
        for kind in WEEK_KINDS:
            for _ in range(n_per):
                if ppd == 1:
                    lv = np.array([1, 1, 1, 1, 1, 0.45, 0.3]) if kind == "peaks" else rng.standard_normal(7)
                    y = lv[dow].astype(np.float64)
                elif kind == "peaks":
                    c1, c2, c3 = 0.33 + rng.uniform(-0.05, 0.05), 0.72 + rng.uniform(-0.05, 0.05), 0.55 + rng.uniform(-0.05, 0.05)
                    wd = np.exp(-0.5 * ((u - c1) / 0.06) ** 2) + 0.8 * np.exp(-0.5 * ((u - c2) / 0.08) ** 2)
                    we = 0.6 * np.exp(-0.5 * ((u - c3) / 0.12) ** 2)
                    y = np.where(dow >= 5, we, wd)
                else:
                    a, b = rng.standard_normal(ppd), 0.6 * rng.standard_normal(ppd) - 0.3
                    y = np.where(dow >= 5, b[tod], a[tod])
                s = rng.uniform(0.5, 5.0) * (y - y.mean()) / (y.std() + 1e-9) + rng.normal(0.0, 3.0)
                c = s[:n_ctx] + noise * s.std() * rng.standard_normal(n_ctx)
                ctx.append(c.astype(np.float32)); fut.append(s[n_ctx:]); kinds.append(kind)
        out[ppd] = (ctx, np.asarray(fut), (dow[n_ctx:] >= 5), kinds)
    return out


@torch.no_grad()
def week_probe(model, device, n_per=4, noise=0.0, table=False):
    """Share of a weekday / weekend pattern the model continues (1 = perfect, 0 = flat line).
    -> dict(all=, weekend=, yesterday=): over the whole horizon, over the weekend points only, and
    what repeating the last day scores on the weekend points. Leaves the model in eval mode."""
    def cap(pred, fut, flat, m):
        r = 1.0 - ((pred - fut) ** 2)[:, m].sum(1) / np.maximum(((flat - fut) ** 2)[:, m].sum(1), 1e-12)
        return np.clip(r, -1.0, 1.0)
    rows, A, W, Y = [], [], [], []
    for ppd, (ctx, fut, wk, kinds) in _week_set(n_per, noise).items():
        med, _ = forecast_left_padded(model, ctx, fut.shape[1], device, max_ctx=8192)
        flat = np.array([c.mean() for c in ctx])[:, None]
        yest = np.stack([np.resize(c[-ppd:], fut.shape[1]) for c in ctx])
        a, w, y = cap(med, fut, flat, np.ones_like(wk)), cap(med, fut, flat, wk), cap(yest, fut, flat, wk)
        A.append(a); W.append(w); Y.append(y)
        rows.append((ppd, a.mean(), w.mean(), y.mean()))
    out = dict(all=float(np.concatenate(A).mean()), weekend=float(np.concatenate(W).mean()),
               yesterday=float(np.concatenate(Y).mean()))
    if table:
        print(f"{'steps/day':>9s} {'all':>7s} {'weekend':>8s} {'yesterday':>10s}")
        for ppd, a, w, y in rows:
            print(f"{ppd:9d} {a:7.2f} {w:8.2f} {y:10.2f}")
        print(f"{'mean':>9s} {out['all']:7.2f} {out['weekend']:8.2f} {out['yesterday']:10.2f}")
    return out


def week_line(wk, wkn):
    return (f"week probe (1 = perfect): weekend {wk['weekend']:.2f} | whole horizon {wk['all']:.2f} | "
            f"noisy context: weekend {wkn['weekend']:.2f}   [repeating yesterday scores {wk['yesterday']:.2f}]")


YEAR_SETS = [(52, 0), (120, 7), (365, 7)]      # (long period, short period inside it; 0 = none)
YEAR_CYCLES = [1.4, 2.4]                        # cycles of history the model is shown


def _year_set(n_per, noise=0.0, seed=0):
    """Series with ONE dated event per long cycle whose next occurrence falls inside the horizon (and two
    other dated events that never touch the horizon). With 1.4 cycles of history that event was seen once
    before, with 2.4 twice. -> list of (P, cycles, contexts, the future, the future WITHOUT the event,
    mask of the event's steps)."""
    rng = np.random.default_rng(seed)
    out = []
    for P, ps in YEAR_SETS:
        w = max(1, P // 50)                      # width of the event: 1, 2, 7 steps
        H = max(8, min(64, P // 6))              # horizon: 8, 20, 60 steps
        for cyc in YEAR_CYCLES:
            n_ctx = int(round(cyc * P))
            t = np.arange(n_ctx + H)
            ctx, fut, base, mask = [], [], [], []
            for _ in range(n_per):
                y = 0.6 * np.sin(2 * np.pi * (t + int(rng.integers(0, P))) / P)
                if ps:
                    y = y + 0.7 * rng.standard_normal(ps)[(t + int(rng.integers(0, ps))) % ps]
                te = n_ctx + int(rng.integers(1, H - w))                  # where the event starts in the horizon
                ev, oth = np.zeros(len(t)), np.zeros(len(t))
                m = float(rng.choice([-1.0, 1.0])) * rng.uniform(2.0, 4.0)
                for a in range(te % P - P, len(t), P):
                    ev[max(a, 0):max(a + w, 0)] += m
                for _k in range(2):
                    off = int(rng.integers(H + w, P - H - w))
                    m2 = float(rng.choice([-1.0, 1.0])) * rng.uniform(1.5, 3.0)
                    for a in range((te - off) % P - P, len(t), P):
                        oth[max(a, 0):max(a + w, 0)] += m2
                full, nob = y + oth + ev, y + oth
                mu_, sd_ = full[:n_ctx].mean(), full[:n_ctx].std() + 1e-9
                sc_, sh_ = rng.uniform(0.5, 5.0), rng.normal(0.0, 3.0)
                f2, b2 = sc_ * (full - mu_) / sd_ + sh_, sc_ * (nob - mu_) / sd_ + sh_
                ctx.append((f2[:n_ctx] + noise * sc_ * rng.standard_normal(n_ctx)).astype(np.float32))
                fut.append(f2[n_ctx:]); base.append(b2[n_ctx:])
                mk = np.zeros(H, bool); mk[te - n_ctx:te - n_ctx + w] = True
                mask.append(mk)
            out.append((P, cyc, ctx, np.asarray(fut), np.asarray(base), np.asarray(mask)))
    return out


@torch.no_grad()
def year_probe(model, device, n_per=4, noise=0.0, table=False):
    """Does the model bring back an event it saw one long cycle ago?
    event: on the steps of the event, 1 - squared error / squared error of the same series WITHOUT the
    event (1 = the event is forecast exactly, 0 = as if it were not there, below 0 = worse).
    rest: the other steps of the horizon, against a flat line at the context mean.
    -> dict(event=, one=, two=, rest=): all sets, event seen once before, seen twice. Leaves the model in eval mode."""
    rows, R = [], []
    for P, cyc, ctx, fut, base, mask in _year_set(n_per, noise):
        med, _ = forecast_left_padded(model, ctx, fut.shape[1], device, max_ctx=8192)
        e = np.clip(1.0 - (((med - fut) ** 2) * mask).sum(1) / np.maximum((((base - fut) ** 2) * mask).sum(1), 1e-12), -1.0, 1.0)
        flat = np.array([c.mean() for c in ctx])[:, None]
        r = np.clip(1.0 - (((med - fut) ** 2) * ~mask).sum(1) / np.maximum((((flat - fut) ** 2) * ~mask).sum(1), 1e-12), -1.0, 1.0)
        rows.append((P, cyc, float(e.mean()), float(r.mean()))); R.append(r)
    out = dict(event=float(np.mean([e for _, _, e, _ in rows])), rest=float(np.concatenate(R).mean()),
               one=float(np.mean([e for _, c, e, _ in rows if c < 2])), two=float(np.mean([e for _, c, e, _ in rows if c >= 2])))
    if table:
        print(f"{'period':>7s} {'cycles seen':>12s} {'event':>7s} {'rest':>7s}")
        for P, cyc, e, r in rows:
            print(f"{P:7d} {cyc:12.1f} {e:7.2f} {r:7.2f}")
        print(f"{'mean':>7s} {'':12s} {out['event']:7.2f} {out['rest']:7.2f}")
    return out


def year_line(yr, yrn):
    return (f"year probe (1 = perfect): event seen once before {yr['one']:.2f} | seen twice {yr['two']:.2f} | "
            f"noisy context {yrn['event']:.2f} | rest of the horizon {yr['rest']:.2f}")


def _flag_set(n_per, seed=0):
    """A weekly pattern around 10 that is ZERO on rare closed days (about 2.5% of the history, at least four),
    with a known flag that names them; the 28-step horizon has two closed days. Two versions of the flag:
    'open' (1 = normal, 0 = closed) and 'holiday' (1 = closed)."""
    rng = np.random.default_rng(seed)
    n_ctx, H = 420, 28
    out = {}
    for pol in ("open", "holiday"):
        groups, fut, base, mask = [], [], [], []
        for _ in range(n_per):
            t = np.arange(n_ctx + H)
            lv = 10.0 + rng.uniform(1.0, 3.0) * np.clip(rng.standard_normal(7), -2.0, 2.0)[(t + int(rng.integers(0, 7))) % 7]
            closed = rng.random(n_ctx + H) < 0.025
            closed[rng.choice(n_ctx - 30, 4, replace=False)] = True
            closed[n_ctx - 7:] = False
            closed[n_ctx + rng.choice(np.arange(2, H - 1), 2, replace=False)] = True
            y = np.where(closed, 0.0, lv)
            flag = (~closed if pol == "open" else closed).astype(np.float32)
            groups.append(dict(targets=[y[:n_ctx].astype(np.float32)], past=[], known=[(flag[:n_ctx], flag[n_ctx:])]))
            fut.append(y[n_ctx:]); base.append(lv[n_ctx:]); mask.append(closed[n_ctx:])
        out[pol] = (groups, np.asarray(fut), np.asarray(base), np.asarray(mask))
    return out


@torch.no_grad()
def flag_probe(model, device, n_per=6):
    """Does the model use a RARE shutdown flag? On the closed days of the horizon: 1 - squared error /
    squared error of the open-day pattern (1 = forecasts zero there, 0 = ignores the flag).
    -> dict(open=, holiday=, hidden=); hidden = the same forecast without the flag. Leaves the model in eval mode."""
    def cap(med, fut, base, mask):
        return float(np.clip(1.0 - (((med - fut) ** 2) * mask).sum(1)
                             / np.maximum((((base - fut) ** 2) * mask).sum(1), 1e-12), -1.0, 1.0).mean())
    res = {}
    for pol, (groups, fut, base, mask) in _flag_set(n_per).items():
        r = forecast_groups(model, groups, fut.shape[1], device)
        res[pol] = cap(np.stack([np.asarray(p[0])[0] for p in r]), fut, base, mask)
        if pol == "open":
            med0, _ = forecast_left_padded(model, [g_["targets"][0] for g_ in groups], fut.shape[1], device, max_ctx=8192)
            res["hidden"] = cap(med0, fut, base, mask)
    return res


def flag_line(fl):
    return (f"flag probe (1 = perfect): 'open' flag {fl['open']:.2f} | 'holiday' flag {fl['holiday']:.2f}   "
            f"[without the flag: {fl['hidden']:.2f}]")


def find_resume(cfg):
    """Checkpoint to continue from: out_dir/last.pt if this session already wrote one, else
    cfg['resume_from'] ('auto' = the last.pt under /kaggle/input with the most points seen)."""
    own = os.path.join(cfg["out_dir"], "last.pt")
    if os.path.exists(own):
        return own
    src = cfg.get("resume_from")
    if not src:
        return None
    if src != "auto":
        assert os.path.exists(src), f"resume_from={src!r} does not exist"
        return src
    import glob
    best, best_pts = None, -1.0
    for f in glob.glob("/kaggle/input/**/last.pt", recursive=True):
        try:
            pts = float(torch.load(f, map_location="cpu", weights_only=False).get("points_seen", 0.0))
        except Exception as e:
            print(f"  (skipping unreadable checkpoint {f}: {type(e).__name__})")
            continue
        print(f"  found {f}: {human(pts)} points seen")
        if pts > best_pts:
            best, best_pts = f, pts
    return best


def train(cfg):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    amp = cfg["amp"]
    if amp == "auto":
        amp = "off" if device.type != "cuda" else ("bf16" if torch.cuda.is_bf16_supported() else "fp16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    amp_on = amp_dtype is not None and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=(amp == "fp16" and device.type == "cuda"))
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"

    model = build_model(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())

    batcher = PackedBatcher(cfg, device, seed=cfg["seed"] + 1)
    _, per_row, pts_per_step = layout_stats(PackedBatcher(cfg, torch.device("cpu"), seed=99), 20)
    est_steps = max(100, int(cfg["target_points"] / pts_per_step))

    print(f"device={gpu}  amp={amp}  params={n_par/1e6:.1f}M  compile={cfg['compile_model']}  "
          f"decoder={cfg.get('decoder', 'pool')}  corpus={cfg.get('corpus', 'v1')}  "
          f"mask_channel={cfg['mask_channel']}  arcsinh={cfg['arcsinh']}  "
          + (f"group_attn=ON lanes={batcher.V} group_prob={batcher.group_prob}" if batcher.group
             else "group_attn=off"))
    print(f"target={human(cfg['target_points'])} points  ~{est_steps:,} steps  "
          f"(~{human(pts_per_step)} real pts/step, ~{per_row.mean():.1f} series/row)")
    print(f"rows={cfg['pack_rows']} x {cfg['max_ctx']} pts  ctx {cfg['min_ctx']}..{cfg['max_ctx']}  "
          f"long_row_frac={cfg['long_row_frac']}  "
          + (f"horizon: next patch everywhere + CPM spans {cfg.get('cpm_min_patches', 1)}-{cfg['cpm_max_patches']} "
             f"patches on {cfg.get('cpm_prob', 0.5):.0%} of series ({cfg.get('cpm_suffix_frac', 0.5):.0%} at the end)"
             + (f", spans up to {cfg.get('cpm_long_max', 32)} patches on {cfg['cpm_long_p']:.0%} of long series"
                if cfg.get('cpm_long_p') else "")
             if cfg.get('decoder') == 'causal' else f"pred_len={cfg['pred_len']}"))
    print(f"multi-patch loss share={cfg.get('horizon_loss_share')}  ema_decay={cfg.get('ema_decay')}  "
          f"milestone every {human(cfg.get('milestone_points') or 0)} points")

    raw = model
    if cfg["compile_model"]:
        model = torch.compile(model, mode="default")

    opt = torch.optim.AdamW(raw.parameters(), lr=cfg["lr"], fused=(device.type == "cuda"))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=est_steps,
                                                pct_start=cfg["warmup_pct"])

    # running average of the weights (EMA). Kept beside the raw weights; never used for training.
    ema_decay = cfg.get("ema_decay")
    ema = {k: v.detach().clone() for k, v in raw.state_dict().items()} if ema_decay else None
    ema_pairs = [(ema[k], p_) for k, p_ in raw.named_parameters()] if ema else []

    start_step, points_seen, base_pts = 0, 0.0, 0.0
    ck_path = find_resume(cfg) if cfg["resume"] else None
    if ck_path:
        print(f"continuing from {ck_path}")
    else:
        print("no earlier checkpoint of this run: starting it from its base weights" if cfg.get("finetune_from")
              else "no earlier checkpoint found: starting from scratch")
    if ck_path:
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        raw.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        try:
            sched.load_state_dict(ck["sched"])
        except Exception:
            pass
        start_step = ck.get("step", 0); points_seen = ck.get("points_seen", 0.)
        base_pts = float(ck.get("base_points_seen", 0.0))
        if ema is not None:
            src = ck.get("ema") or ck["model"]
            for k in ema:
                ema[k].copy_(src[k])
        print(f"resumed @ step {start_step:,}, {human(points_seen)} points already seen")
        if sched.total_steps != est_steps:
            print(f"  budget changed ({sched.total_steps:,} -> {est_steps:,} steps): rebuilding LR schedule")
            sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=est_steps,
                                                        pct_start=cfg["warmup_pct"])
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                for _ in range(min(start_step, est_steps - 1)):
                    sched.step()

    elif cfg.get("finetune_from"):
        # FINE-TUNE: every weight comes from the base checkpoint; the optimizer, the LR schedule and
        # the point counter start fresh (a plain resume would continue the base run's schedule)
        ck0 = torch.load(cfg["finetune_from"], map_location=device, weights_only=False)
        raw.load_state_dict(ck0["model"])
        base_pts = float(ck0.get("points_seen", 0.0)) + float(ck0.get("base_points_seen", 0.0) or 0.0)
        print(f"fine-tune from {cfg['finetune_from']} ({human(base_pts)} points, step {int(ck0.get('step', 0)):,}): "
              f"all weights loaded, fresh optimizer and LR schedule")
        del ck0

    elif cfg.get("init_from"):
        # warm start from a univariate checkpoint: every shared weight is copied; the group
        # layers and role embeddings are new and start at zero, so step 0 == that checkpoint
        ck0 = torch.load(cfg["init_from"], map_location=device, weights_only=False)
        missing, unexpected = raw.load_state_dict(ck0["model"], strict=False)
        bad = [k for k in missing if ".group." not in k and "role_emb" not in k and not k.startswith("look_")]
        assert not bad and not unexpected, f"init_from does not match this architecture: {bad[:5]} {unexpected[:5]}"
        zero_group_layers(raw)
        print(f"warm start from {cfg['init_from']}: {len(missing)} new (group) tensors, group output starts at zero")

    set_group_options(None)                 # the holdout always uses the v5 group mix ...
    val_batches = build_val_batches(cfg, device, cfg["val_batches"])
    set_group_options(cfg)                  # ... training uses this run's

    def save(tag_paths, note):
        ck = dict(model=raw.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(),
                  step=step, points_seen=points_seen, cfg=cfg, synthetic_only=True,
                  packed=True, note=note, run_name=cfg.get("run_name"), base_points_seen=base_pts)
        if ema is not None:
            ck["ema"] = ema
        for p in tag_paths:       # write to a temp file, then rename: never a half-written checkpoint
            tmp = os.path.join(cfg["out_dir"], p + ".tmp")
            torch.save(ck, tmp)
            os.replace(tmp, os.path.join(cfg["out_dir"], p))

    # milestone checkpoints: ckpt_100B.pt, ckpt_200B.pt, ... kept forever
    ms = float(cfg.get("milestone_points") or 0)
    next_ms = (math.floor(points_seen / ms) + 1) * ms if ms > 0 else float("inf")

    def do_val(tag):
        m = {}
        vl, bands = evaluate_packed(model, val_batches, amp_dtype, amp_on, device)
        bs = "  ".join(f"ctx {k}: {v:.4f}" for k, v in bands.items())
        print(f"  [{tag}] holdout loss = {vl:.4f}   |  {bs}", flush=True)
        m.update(holdout=vl, **{f"ctx {k}": v for k, v in bands.items()})
        if batcher.decoder:
            l_one, l_multi = horizon_split(model, val_batches, amp_dtype, amp_on, device)
            print(f"  [{tag}] one-patch loss {l_one:.4f}  |  multi-patch loss {l_multi:.4f}", flush=True)
            m.update(one_patch=l_one, multi_patch=l_multi)
        if batcher.group:
            g_on, g_cut = group_gain(model, val_batches, amp_dtype, amp_on, device)
            gg = (g_cut - g_on) / max(g_cut, 1e-9)
            print(f"  [{tag}] grouped series: loss with groups ON {g_on:.4f}  CUT {g_cut:.4f}  "
                  f"-> group gain {gg:+.1%}", flush=True)
            m.update(group_gain=gg)
        if cfg.get("probe_at_val") and batcher.decoder:
            pr = repeat_probe(raw, device)
            wk, wkn = week_probe(raw, device), week_probe(raw, device, noise=0.3)
            yr, yrn = year_probe(raw, device), year_probe(raw, device, noise=0.2)
            fl = flag_probe(raw, device)
            raw.train()
            print(f"  [{tag}] {probe_line(pr)}", flush=True)
            print(f"  [{tag}] {week_line(wk, wkn)}", flush=True)
            print(f"  [{tag}] {year_line(yr, yrn)}", flush=True)
            print(f"  [{tag}] {flag_line(fl)}", flush=True)
            m.update(probe_list=pr["list"], probe_other=pr["other"], week_end=wk["weekend"],
                     week_all=wk["all"], week_end_noisy=wkn["weekend"], year_once=yr["one"], year_twice=yr["two"],
                     year_noisy=yrn["event"], flag_open=fl["open"], flag_holiday=fl["holiday"])
        LAST_VAL[cfg.get("run_name", "run")] = m
        return m

    model.train()
    step = start_step
    t_win = time.time(); pts_win = loss_win = gen_win = 0.; n_win = 0; n_bad = 0; ser_win = 0
    hist = []

    max_hours, stopped = cfg.get("max_hours"), False
    while points_seen < cfg["target_points"] and step < cfg["max_steps"]:
        if max_hours and step % 20 == 0 and (time.time() - _T0) / 3600 >= max_hours:
            stopped = True
            break
        step += 1

        tg = time.time()
        b = batcher.next_batch()
        if device.type == "cuda":
            torch.cuda.synchronize()
        gen_win += time.time() - tg

        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_on):
            out = packed_forward(model, b)
            loss = out["loss"]

        if not torch.isfinite(loss):
            n_bad += 1; opt.zero_grad(set_to_none=True); continue

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg["grad_clip"])
        scaler.step(opt); scaler.update()
        if sched.last_epoch < sched.total_steps - 1:
            sched.step()
        if ema is not None:
            d = min(ema_decay, (1.0 + step) / (10.0 + step))      # short memory in the first steps
            with torch.no_grad():
                for e_, p_ in ema_pairs:
                    e_.lerp_(p_, 1.0 - d)

        points_seen += b["points"]; pts_win += b["points"]; ser_win += b["n_series"]
        loss_win += loss.item(); n_win += 1

        if step % cfg["print_every"] == 0:
            el = time.time() - t_win
            pps = pts_win / max(el, 1e-9)
            eta = (cfg["target_points"] - points_seen) / max(pps, 1e-9) / 3600
            avg_loss = loss_win / max(n_win, 1)
            hist.append((step, avg_loss))
            print(f"step {step:,}  loss={avg_loss:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
                  f"series/step={ser_win/max(n_win,1):,.0f}  "
                  f"{human(points_seen)}/{human(cfg['target_points'])} "
                  f"({points_seen/cfg['target_points']:5.1%})  ~{pps/1e6:.2f}M pts/s  "
                  f"gen={gen_win/max(el,1e-9):4.1%}  ETA {eta:.1f}h  bad={n_bad}", flush=True)
            if step - start_step == cfg["print_every"] and device.type == "cuda":
                print(f"  peak GPU memory so far: {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB of "
                      f"{torch.cuda.get_device_properties(0).total_memory / 2 ** 30:.0f} GB", flush=True)
            t_win = time.time(); pts_win = loss_win = gen_win = 0.; n_win = 0; ser_win = 0

        if step % cfg["val_every"] == 0:
            do_val(f"val @ {step:,}")

        if step % cfg["ckpt_every"] == 0:
            save(["last.pt"], "Packed synthetic-only checkpoint; not a real-corpus model.")
            print(f"  saved @ {step:,} ({human(points_seen)} pts)", flush=True)

        if points_seen >= next_ms:
            name = f"ckpt_{int(round(next_ms / 1e9))}B.pt"
            save([name, "last.pt"], f"Milestone checkpoint at {human(next_ms)} points (run still in progress).")
            print(f"  MILESTONE saved: {name}  (step {step:,}, lr {sched.get_last_lr()[0]:.2e})", flush=True)
            next_ms += ms

    if stopped:
        save(["last.pt"], "Stopped at the session time limit; add this Output as Input and run again.")
        left = cfg["target_points"] - points_seen
        print(f"\nSTOPPED at the {max_hours} h session limit: {step:,} steps, {human(points_seen)} of "
              f"{human(cfg['target_points'])} points ({points_seen / cfg['target_points']:.1%}). last.pt saved in "
              f"{cfg['out_dir']}.\nTO CONTINUE: let this version finish saving, open the notebook, Add Input -> "
              f"this notebook's Output, and run again ({human(left)} points left).")
        return hist
    final_val = do_val("FINAL")
    name = cfg.get("run_name", "run")
    slim = os.path.join(os.path.dirname(cfg["out_dir"].rstrip("/")), name + ".pt")
    torch.save(dict(model=raw.state_dict(), cfg=cfg, step=step, points_seen=points_seen, packed=True,
                    synthetic_only=True, run_name=name, base_points_seen=base_pts,
                    base_ckpt=os.path.basename(str(cfg.get("finetune_from"))), final_val=final_val,
                    note=f"Fine-tune '{name}': {human(points_seen)} points on top of a {human(base_pts)}-point base."),
               slim + ".tmp")
    os.replace(slim + ".tmp", slim)
    shutil.rmtree(cfg["out_dir"], ignore_errors=True)      # the resumable checkpoint is no longer needed
    print(f"\nDone: {name}. {step:,} steps, {human(points_seen)} points. -> {slim} "
          f"({os.path.getsize(slim) / 2 ** 20:.0f} MB, weights only)")
    return hist


def load_packed_checkpoint(path, device):
    """Rebuild a model from a checkpoint written by this file (mask_channel etc. restored
    from the saved cfg). Use this in the fev eval instead of constructing the model by hand."""
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["cfg"]
    model = build_model(dict(cfg, dropout=0.0)).to(device)
    model.load_state_dict(ck["model"])
    return model.eval(), cfg, ck
