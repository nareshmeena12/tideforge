# Patching probe (main-track idea), 2026-10-16

STATUS (after probe 5, 2026-10-17): the effect is real on synthetic recall, and the insertion fix works there,
but the fix HURTS real ETT forecasts (probe 3). Trained fixes at 5B points (probe 4) moved neither the gap nor
fev-bench (pf_arch slightly worse). Tide 30M has almost no exact recall of iid motifs even when aligned. Tide
100M (probe 5) recalls a little more (aligned iid NE 0.98 -> 0.905, improving with points seen) and the comb
appears as recall appears (iid gap 0.009 -> 0.079), but it is still far from Chronos-2 (0.24), and the 100M's
fev score is no better than the 30M's. Conclusion: not the lever for fev; keep as an analysis section of the
workshop paper. Main effort moves to covariates.

Zero-shot test of pretrained models (no training). Context: 512 points of AR(1) noise holding one 128-point
segment, with the cue repeated at the end; forecast the 64-point motif. Copy distance = 192 + s + 32k, with
s = 0..31 the offset relative to the patch grid. NE = MSE(with copy) / MSE(control with no copy):
0 = perfect recall, 1 = no help. n = 128 series per cell (Chronos-T5: 64).

## Probe 1 results (NE at s=0 / s=16 / mean over misaligned s; gap with 95% CI)
| Model (patch) | iid | smooth | real (ETTh1) |
|---|---|---|---|
| Chronos-2 (16) | 0.24 / 0.27 / 0.98, gap +0.73 [0.69, 0.76] | 0.02 / 0.02 / 0.05, gap +0.03 | 0.54 / 0.67 / 0.81, gap +0.20 [0.15, 0.26] |
| TimesFM 3.0 (32) | 0.47 / 0.72 / 0.98, gap +0.51 [0.48, 0.54] | 0.08 / 0.41 / 0.42, gap +0.34 [0.31, 0.37] | 0.48 / 0.67 / 0.86, gap +0.38 [0.32, 0.42] |
| Chronos-Bolt (16) | 0.99 everywhere (no recall even when aligned: excluded) | 0.73-0.85 | 0.83-0.98 |
| Chronos-T5 (per point, control) | 0.055-0.062 flat | 0.30-0.38 flat | 0.08-0.11 flat |
| Tide 30M (32), ft3_main (own implementation, n = 128 per s) | 0.981 / 0.981 / 0.990, gap +0.009 [0.00, 0.02] | 0.893 / 0.914 / 0.943, gap +0.050 [0.03, 0.08] | 0.684 / 0.747 / 0.797, gap +0.113 [0.06, 0.16] |
| Tide 100M (32), tide100m_300b (probe 5) | 0.905 / 0.894 / 0.983, gap +0.079 [0.05, 0.11] | 0.850 / 0.863 / 0.929, gap +0.079 [0.05, 0.11] | 0.641 / 0.703 / 0.774, gap +0.133 [0.09, 0.18] |

Pre-registered verdict: iid LIMIT PRESENT (Chronos-2 and TimesFM 3.0; control flat; Chronos-2 dips at exactly
s = 0 and 16). smooth / real: "not shown" under the strict rule (Chronos-2 immune on smooth; on real its aligned
NE of 0.54 misses the 0.5 bar).

Exploratory: on real motifs both models dip at 0 and 16, with smaller dips at 8 and 24. TimesFM's smooth error
grows with the distance to the nearest multiple of 32. The per-point Chronos-T5 recalls real motifs at ~0.09
at every offset (confounded comparison).

## Probe 2: phase-shift test-time augmentation (synthetic), n = 32 per cell
Insert r interpolated points before the last 64 (r = 0..P-1), keep the r with the narrowest 10-90% band.
| Model, motif | base misaligned | select misaligned | base aligned | recovery | control cost |
|---|---|---|---|---|---|
| Chronos-2, iid | 0.983 | 0.276 | 0.255 | 0.97 | 1.004 |
| Chronos-2, real | 0.840 | 0.506 | 0.592 | 1.35 | 0.983 |
| TimesFM 3.0, iid | 0.976 | 0.412 | 0.479 | 1.14 | 0.996 |
| TimesFM 3.0, smooth | 0.474 | 0.101 | 0.070 | 0.92 | 1.029 |
| TimesFM 3.0, real | 0.875 | 0.501 | 0.427 | 0.83 | 0.969 |
Pre-registered rule: PASS on all four (iid and real, both models). Averaging over r does not help.

## Probe 3: cheap selectors (synthetic) and real data (ETT), 2026-10-16
Synthetic: cross-correlating the last 64 points with the history finds the true copy in 100% of series, and
inserting only that shift ("xcorr", 1 forecast) recovers as much as trying all P shifts (Chronos-2 iid
0.983 -> 0.278; TimesFM iid 0.976 -> 0.413; TimesFM real 0.875 -> 0.494). Coarse grids fail on iid.

Real ETT (ETTh1/h2/m1/m2, last 20%, all columns, context 512 -> 64, ~380 windows each). Relative MSE vs base:
| Model | xcorr (1) | xcorr+base (2) | grid4 |
|---|---|---|---|
| Chronos-2 | 1.10-1.39 | 1.006-1.075 | 1.03-1.15 |
| TimesFM 3.0 | 1.14-1.51 | 1.017-1.159 | 1.08-1.34 |
Pre-registered "no harm" (<= 1.01 on every dataset): FAIL for every method. xcorr makes ~50-60% of windows
worse. Median best correlation of the query with the history: 0.63-0.87 (approximate, not exact, repeats).

Likely reason: on periodic real data, inserting r points breaks the time continuity and seasonal phase
between the history and the recent window, and that costs more than patch re-alignment gains. The synthetic
background had no periodicity, so the cost never showed there. Any fix that moves data in time inherits this;
a real fix has to match at sub-patch offsets inside the model.

## Probe 4: trained fixes on Tide 30M, 5B points per arm, 2026-10-17
Four 5B-point fine-tunes of ft3_main, same mix / schedule / seed; only the fix differs:
pf_ctrl (none), pf_data ("offset copy" family, ~10% of series: segments re-appearing at arbitrary point
offsets in realistic backgrounds), pf_arch ("lookback embedding": each token also embeds the previous patch,
zero-initialised), pf_both. Probe as above (own implementation, AR(1) phi 0.6 background, k = 0..3).

| Arm | iid gap | smooth gap | real gap | holdout | repeat probe (other periods) |
|---|---|---|---|---|---|
| ft3_main | +0.009 [0.00, 0.02] | +0.050 [0.03, 0.08] | +0.113 [0.06, 0.16] | 0.2438 | 0.957 |
| pf_ctrl | +0.007 | +0.052 | +0.117 | 0.2459 | 0.959 |
| pf_data | +0.011 | +0.052 | +0.123 | 0.2461 | 0.963 |
| pf_arch | +0.008 | +0.044 | +0.111 | 0.2461 | 0.974 |
| pf_both | +0.008 | +0.047 | +0.125 | 0.2461 | 0.974 |

Result: no arm changes the gap (all inside the control's CI); holdout unchanged. Only side effect: the
lookback embedding raises the repeat probe on unusual periods (0.959 -> 0.974), a sub-patch-alignment skill.
Key observation: Tide's aligned iid NE is 0.98 (Chronos-2 0.24, TimesFM 0.47): it barely recalls a motif seen
once even when aligned, so there is no recall for the patch grid to block. Its weakness is single-occurrence
in-context recall itself (consistent with year_once ~0.09).

### fev-bench of the four arms (SQL skill score, groups on / hidden)
| Model | groups on | hidden |
|---|---|---|
| ft3_main | 43.78 | 42.09 |
| pf_ctrl | 43.71 | 41.98 |
| pf_data | 43.63 | 41.94 |
| pf_both | 43.64 | 41.97 |
| pf_arch | 43.64 | 41.80 |
Paired vs pf_ctrl: pf_arch hidden -0.17 [-0.33, -0.02] (small, real harm); pf_data and pf_both not
distinguishable from zero. ft3_main is +0.12 above pf_ctrl in hidden mode, i.e. the 5B fine-tune on its own
costs a little. Every history band is unchanged (2049+: 0.520-0.522 for all arms vs Chronos-2 0.479).
Verdict: the trained fixes are closed; they help neither the probe nor the benchmark.

## Probe 5: does recall grow with size? Tide 100M and its milestones, 2026-10-17
| Model | iid NE s0 / s16 / mis, gap | smooth NE s0 / mis, gap | real NE s0 / mis, gap |
|---|---|---|---|
| ft3_main (29M, 1.2T) | 0.981 / 0.981 / 0.990, +0.009 [0.00, 0.02] | 0.893 / 0.943, +0.050 | 0.684 / 0.797, +0.113 |
| 100M @100B | 0.940 / 0.925 / 0.985, +0.045 [0.02, 0.07] | 0.809 / 0.910, +0.101 | 0.634 / 0.784, +0.150 |
| 100M @200B | 0.927 / 0.927 / 0.986, +0.059 [0.03, 0.09] | 0.848 / 0.927, +0.078 | 0.650 / 0.779, +0.129 |
| 100M @300B (final) | 0.905 / 0.894 / 0.983, +0.079 [0.05, 0.11] | 0.850 / 0.929, +0.079 | 0.641 / 0.774, +0.133 |

- iid recall appears with size and keeps growing with points seen; the comb appears with it (gap CI now well
  clear of 0). Misaligned NE stays ~0.98 throughout: all new recall is grid-locked.
- Share of the aligned recall benefit lost when misaligned, (mis - s0) / (1 - s0): 100M iid 82% (Chronos-2
  97%, TimesFM 96%); real 37% for both Tide models (Chronos-2 59%, TimesFM 73%). Tide is less grid-locked on
  real motifs but recalls less overall.
- Smooth recall peaks at 100B (0.81) and degrades later (0.85).
- Comb shape: for the 100M on iid, s = 16 dips as deep as s = 0 (0.894 vs 0.905), with smaller dips at 20, 24,
  28 and none at 4, 8, 12. Smooth / real: dips at multiples of 8 for every Tide model.
- Period-phase check (analytic, seasonal layer only: period bank with 15% ext share, nested x7/x12/x30/x52,
  all lags n*P): lags mod 32 land on multiples of 4 for ~79% of the periodic-lag mass (uniform 25%),
  multiples of 8 ~48% (12%), 0 or 16 ~28% (6%), exactly 0 ~15-19% (3%). At the probe's lags (192-319) the
  order is 0 (0.19) > 24 (0.12) > 16 (0.11) > 4, 28, 20, 12, 8 (0.07-0.09).
  Verdict: SUPPORTS the grid of the comb (dips only at multiples of 4 / 8, the residues the data
  over-represents); does NOT explain the depths: 16 dips deepest although 24 has more mass, and 4 / 8 / 12
  have no iid dips although they carry as much mass as 20 / 28. Odd periods (7, 13, 365) and the repeat family
  (log-uniform periods) spread over all residues. A causal test is needed before claiming it in the paper
  (train with periods that avoid multiples of 4, check whether the sub-patch dips disappear).
- fev: the 100M is +0.21 [-0.17, +0.60] over the 30M (not significant) and its >512-history gap to Chronos-2 is
  unchanged, so a recall gain of this size does not show on the benchmark.
