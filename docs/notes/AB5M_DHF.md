# 5M A/B: CPM (our recipe) vs DHF (Decoupled Horizon Forecasting), 2026-10-17

Scripts: ab5m_cpm_vs_dhf.py (training, both arms), eval_ab5m_time_fev.py (TIME + fev-bench scoring + head-to-head).
(Not in the tidemark repo; ask the owner for them. They define tideforge's 5M recipe.)

## RESULT (100B points per arm, both scored univariate)
| | TIME MASE (norm.) | TIME CRPS (norm.) | fev SQL skill |
|---|---|---|---|
| ab5m_cpm | 0.735 | 0.617 | 0.388 |
| ab5m_dhf | 0.772 | 0.638 | 0.369 |
| Timestral-5M | 0.652 | 0.544 | - |
| Fluxion 30M univariate | 0.687 | 0.575 | - |
| Chronos-2 (fev, official run) | 0.662 | 0.556 | 0.469 |

- DHF / CPM on TIME: MASE 1.050 [1.034, 1.068], DHF better on 21% of tasks; CRPS 1.035 [1.022, 1.049].
  Worst on short horizons (1.065), least on long (1.024).
- DHF / CPM on fev: SQL error 1.031 [1.018, 1.046], DHF better on 32% of tasks.
- Verdict: DHF AS IMPLEMENTED LOSES, significantly, on both benchmarks, at equal points (and it used ~2x the
  compute). Do not adopt it.
- Probes (repeat, week, year, flag) were mixed and within noise: DHF slightly better at year (one-off recall),
  CPM slightly better at repeat / week / flags.
- Recipe gap: our 5M CPM is 12.8% behind Timestral-5M on TIME MASE (wins 1% of tasks).

## Likely reasons (untested)
1. The first horizon patch: at inference DHF reads it from a Dec token, but only ~1/32 of Dec tokens are trained
   at horizon 1. CPM reads it from the densely trained Enc stream. Fits the "worst on short horizons" pattern.
   Cheap test: hybrid inference (Enc token for patch 1, Dec for the rest) on the trained checkpoint.
2. Anchor frame instead of Timestral's advancing refine frame.
3. Hyperparameters (LR, horizon share) chosen for CPM.
4. The group firewall (only affects known-covariate groups; TIME is univariate here).

## Design (only the horizon mechanism differs)
- Model: d256, 4 layers, 4 heads x 64, d_ff 1024, patch 32, dropout 0.1, group attention (8 lanes),
  9 quantiles, arcsinh -> 5.45M parameters (+256 for DHF's forecast-query vector).
- Data: the 300B run's mix. LR 5e-4 peak, cosine over points, 2% warmup, AdamW (0.9, 0.95), clip 1.0.
  100B points per arm, milestone at 50B.
- CPM: horizon patches blanked in the input (the existing CausalPatchDecoder).
- DHF: no blanking. Sequence [Enc ; Dec], one learned forecast-query token per patch. Per series, anchors at
  a0, a0+32, ... with a0 random in [1, min(n-1, 32)]. A Dec token reads Enc < anchor (own series) + earlier
  Dec tokens of its block. Targets in the anchor frame (stats at the end of patch a-1). Loss = 0.5 * Enc dense
  next-patch + 0.5 * Dec.
- Groups in DHF: Enc as the baseline + a firewall (covariates never read targets; known covariates read only
  known). Without it a target's future leaks into its own Dec tokens through a known covariate.
- Checks (CPU): packed == alone 9e-7; future leak 0; inference == training 6e-7; negative control without the
  firewall fails (8.3e-3).
