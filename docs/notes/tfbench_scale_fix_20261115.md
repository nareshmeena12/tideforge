# tfbench scorer: the input-scale fix (2026-11-15)

Runs on Perlmutter by the owner at commit 86fbf3f (job `score_rescale`, runbook I4). Fix in bae302f, cache-key fix in
86fbf3f, diag_scale task names in 8468f8c.

## Problem

On the tfbench fast tier (groups OFF, before the fix), a few tasks were orders of magnitude worse than seasonal naive for
both models, and they dominated the geometric mean:

| task | r11 rel MASE | ft3_main rel MASE |
|---|---|---|
| thermobench_thermobench_pure__5min/medium | 65,930 | 68,324 |
| thermobench_thermobench_pure__h/medium | 77,220 | 15,616 |
| datonic_world_development_indicators_default__ys/short | 7,554 | 1,978 |
| juliensimon_goes_xray_flux__min/medium | ~10 | ~10 |

Fast-tier MASE skill before the fix: r11 11.9, ft3_main 17.89.

## Cause

ft3_core's `_causal_stats` floors the model's input std at an absolute 1e-3. Series whose spread is ~1e-3 or less (values
~1e-6: GOES X-ray flux, WDI ratios; near-flat thermobench series) are normalised by the floor instead of their own std, and
the forecast spread comes out ~1000x the series. `diag_scale` showed bf16 = fp32: precision is not the cause, the floor is.

## Fix (scorer only; model weights and third_party unchanged)

`tideforge/score/adapters/ft3.py`: every series (targets, past covariates, known covariates with their future) is mapped in
float64 to (y − median) / MAD before the model (std when the MAD is 0, floored at 1e-8 × |median|, else 1), and the
quantiles are mapped back in float64. The model's own normalisation is affine-invariant apart from the floor and the
float32 input, so ordinary series are unaffected. Prediction shards are float64. `--rescale off` keeps the old path
(results under `<model_id>_raw`); the prediction-cache key carries the variant and SCORER_VERSION 2, so no shard from
before bae302f is reused.

## Results

diag_scale, r11, rel MASE (old path → rescaled):

| task | before | after |
|---|---|---|
| datonic_world_development_indicators_default__ys/short | 7,554 | 0.98 |
| juliensimon_goes_xray_flux__min/medium | 9.6 | 0.46 |
| thermobench_thermobench_pure__5min/medium | 65,930 | 0.64 |
| thermobench_thermobench_pure__h/medium | 77,220 | 0.82 |

fev-bench, ft3_main, rescale on: 43.84 groups ON / 42.09 groups OFF (old scorer 43.78 / 42.09). ON moved +0.06, inside the
~0.1 tolerance; OFF unchanged.

tfbench v1.1.0 fast, groups OFF, ft3_main, rescale on: MASE skill 27.42 (was 17.89), WQL 41.19, SQL 39.60; 0 tasks with
rel MASE > 20.

Not yet reported here: r11's fast-tier skill after the fix, and the full-tier counts of tasks with rel MASE > 20 before /
after (in the job's stats.json files).

## Consequences

- Every tfbench number from before bae302f (old path) is superseded; compare runs only with rescale on.
- fev-bench numbers stay comparable with the old board (≤ 0.06 change).
- Open question, not tested: training uses the same `_causal_stats` floor, so generated series with spread below ~1e-3 are
  also normalised by the floor in training. Whether the generator produces such series, and whether that matters, is a
  generator question for EXPERIMENTS.md, not a scorer one.
