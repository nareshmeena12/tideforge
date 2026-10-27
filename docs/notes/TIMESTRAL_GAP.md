# Timestral (Mistral, 2026-10-15) vs Fluxion-Tide: what they do that we don't

Source: the blog post plus the public inference code (github.com/mistralai/timestral, commit a534bdb), read in
full on 2026-10-17. Their sizes are 5M / 28M / 123M / 209M. The 28M beats Chronos-2 on GIFT-Eval and TIME; we
are at TIME MASE 0.680 vs Chronos-2 0.662.

## Differences, by what it costs us to copy them
| # | Theirs | Ours | Cost |
|---|---|---|---|
| 1 | "clampsym" at inference (on by default; their benchmark numbers include it): forecast x and -x, reflect the second (-Q(1-p)), average; clamp at 0 if the context is all >= 0 | none | inference only |
| 2 | Categorical covariates target-encoded per series ((mean + sum_c) / (1 + count_c)) | pd.factorize codes (impose a fake order) | inference only |
| 3 | 21 quantiles 0.01..0.99, median-centred monotone head (median free, cumsum softplus both ways) | 9 quantiles, monotone from the lowest | fine-tune (head) |
| 4 | Known-future covariate rows see the WHOLE window (bidirectional time attention); firewall: covariates never read targets, known rows never read causal rows | lane attention at the same patch only; known lanes causal; no firewall | fine-tune (masks only) |
| 5 | DHF: [Enc ; Dec] sequence; learned forecast-query tokens read the causal context; dec-dec causal within a block of K = 32; the full history stays as context, true gaps stay flagged | contiguous patch masking: horizon = blanked patches, which also removes context | re-pretrain |
| 6 | Advancing refine frame: horizon patch j denormalised with the context stats updated by the block's earlier medians (as in training) | the whole horizon in the last context patch's frame | inference + training consistency |
| 7 | Block: RMSNorm, SwiGLU, sigmoid-gated attention (Qwen-style), partial RoPE (half the head dims), no biases; schedule mixes time and time+variate layers | GELU FFN, full RoPE, group attention in every layer | re-pretrain |
| 8 | Two-stage: synthetic pre-training at ctx 8192, then real + synthetic post-training at ctx 32768 | synthetic only, ctx 8192 | data decision (breaks synthetic-only) |

Shared already: causal per-patch normalisation + asinh, [values, missing] patch embedding, monotone quantiles,
multivariate synthesizer, patch 32 (check theirs).

## Plan, cheapest first
1. Measure before building: score Timestral-28M / 5M with our TIME + fev harness, with clampsym on vs off, plus
   a speed run. This splits their lead into inference tricks vs model.
2. Add clampsym and categorical target encoding to our inference; rescore ft3_main on TIME + fev.
3. Fine-tune (~20-50B points): 21 quantiles + known-future bidirectional rows + firewall.
4. A new pretrain run with DHF + refine frame + modern block (+32K context stage). This is the real bet.
5. Real-data post-training: only as a separate variant, so the synthetic-only claim is kept.

Paper impact: their 28M takes the "small, synthetic-first, beats Chronos-2" position. What we still have:
synthetic ONLY (they post-train on real data), and speed (to be measured against theirs).
