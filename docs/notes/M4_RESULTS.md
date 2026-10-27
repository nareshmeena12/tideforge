# M4 twins (30M, 100B points, Perlmutter 4xA100): FINAL, 2026-10-18

Pilot (5B points, holdout): softmax 0.2698 (lr 3e-4), SWA16 0.2708 (3e-4), SWA8 0.2731 (3e-4), GLA 0.2841 (1.5e-4; tie with 3e-4).
All 9 jobs: exit 0, 100B points, 0 rollbacks. Wall time per twin 43-58 min (GLA slowest). Bench: 36 M points/s per node (GLA 32).

| Twin | Holdout | ctx 1-64 | 65-512 | 513-2048 | 2049+ | repeat gen/other | spike | weekend | year once/twice | holiday flag | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|
| softmax s0 | 0.2487 | 0.499 | 0.315 | 0.169 | 0.102 | 0.97/0.91 | 0.90 | 0.95 | 0.10/0.35 | 0.04 | 0 |
| softmax s1 | 0.2490 | 0.499 | 0.316 | 0.169 | 0.102 | 0.98/0.89 | 0.96 | 0.93 | 0.12/0.41 | 0.06 | 0 |
| SWA16 s0 | 0.2494 | 0.498 | 0.316 | 0.170 | 0.103 | 0.94/0.79 | 0.81 | 0.78 | 0.11/0.32 | 0.04 | 0 |
| SWA16 s1 | 0.2494 | 0.498 | 0.315 | 0.170 | 0.103 | 0.96/0.80 | 0.88 | 0.75 | 0.12/0.37 | 0.07 | 0 |
| SWA8 s0 | 0.2499 | 0.498 | 0.316 | 0.170 | 0.105 | 0.90/0.72 | 0.75 | 0.50 | 0.01/0.17 | 0.03 | 0 |
| SWA8 s1 | 0.2498 | 0.499 | 0.315 | 0.169 | 0.105 | 0.91/0.72 | 0.76 | 0.54 | 0.04/0.20 | 0.05 | 0 |
| GLA s0 | 0.2541 | 0.500 | 0.319 | 0.175 | 0.110 | 0.84/0.69 | 0.56 | -0.07 | -0.02/0.09 | 0.06 | 33 |
| GLA s1 | 0.2545 | 0.500 | 0.320 | 0.176 | 0.111 | 0.86/0.71 | 0.58 | -0.09 | -0.01/0.07 | 0.06 | 34 |

Seeds agree to <= 0.0004 in holdout in every arm.

Findings (holdout and probes):
- Holdout order: softmax 0.2489 < SWA16 0.2494 < SWA8 0.2499 << GLA 0.2543 (GLA 2.2% worse).
- The gap appears only at long context (ctx 2049+: 0.102 / 0.103 / 0.105 / 0.110).
- Weekend probe: softmax 0.94, SWA16 0.77, SWA8 0.52, GLA -0.08. GLA's compressed state, not attention range.
- Year event once/twice: softmax 0.11/0.38, SWA16 0.12/0.35, SWA8 0.03/0.19, GLA ~0/0.08. Holiday flags ~0.05 everywhere (data/budget limit).

## Public benchmarks (owner's pipeline, reference/time_fev_eval.py; univariate, ctx 8192), 2026-10-18

TIME (98 tasks; geometric mean normalised to seasonal naive, lower = better):
| Model | MASE | CRPS |
|---|---|---|
| Timestral-28M | 0.638 | 0.534 |
| Timestral-5M | 0.652 | 0.544 |
| Chronos-2 | 0.662 | 0.556 |
| Toto-2.0-22m | 0.670 | 0.562 |
| TiRex | 0.683 | 0.573 |
| Toto-2.0-4m | 0.691 | 0.582 |
| softmax s1 / s0 | 0.699 / 0.704 | 0.588 / 0.589 |
| SWA16 s1 / s0 | 0.719 / 0.719 | 0.601 / 0.602 |
| SWA8 s1 / s0 | 0.730 / 0.733 | 0.607 / 0.610 |
| GLA s0 / s1 | 0.749 / 0.758 | 0.625 / 0.631 |

fev-bench (100 tasks; SQL skill score, higher = better):
| Model | Skill [95% CI] | Win rate |
|---|---|---|
| Timestral-28M | 0.481 | 0.875 |
| Chronos-2 | 0.469 | 0.805 |
| Timestral-5M | 0.458 | 0.759 |
| Toto-2.0-22m | 0.425 | 0.606 |
| TiRex | 0.423 | 0.621 |
| softmax s0 / s1 | 0.413 / 0.412 | 0.527 / 0.511 |
| SWA16 s1 / s0 | 0.406 / 0.405 | 0.487 / 0.455 |
| Toto-2.0-4m | 0.403 | 0.442 |
| SWA8 s1 / s0 | 0.402 / 0.399 | 0.443 / 0.399 |
| GLA s0 / s1 | 0.380 / 0.379 | 0.276 / 0.291 |

Head-to-heads (geometric-mean error ratio vs softmax, seeds averaged, 95% bootstrap CI over tasks; > 1 = worse than softmax):
| Arm | TIME MASE | TIME short / medium / long | fev-bench | fev short / medium / long |
|---|---|---|---|---|
| GLA | 1.074 [1.050, 1.102] | 1.043 / 1.078 / 1.137 | 1.056 [1.039, 1.077] | 1.013 / 1.066 / 1.095 |
| SWA16 | 1.025 [1.016, 1.035] | 1.019 / 1.029 / 1.033 | 1.012 [1.004, 1.021] | 0.999 / 1.014 / 1.023 |
| SWA8 | 1.043 [1.030, 1.057] | 1.027 / 1.054 / 1.066 | 1.020 [1.009, 1.033] | 1.002 / 1.021 / 1.039 |
Seed noise floor (seed 0 vs seed 1): 0.988-1.007 on TIME MASE, 0.998-1.006 on fev-bench.

Benchmark findings:
- Same order on both public benchmarks as on holdout and probes: softmax > SWA16 > SWA8 >> GLA, both seeds of every arm adjacent.
- GLA is 7.4% worse on TIME and 5.6% worse on fev-bench, CIs well clear of the seed floor; GLA beats softmax on only 17-18% of tasks.
- Every gap grows with horizon: GLA vs softmax on fev-bench 1.3% (short) -> 6.6% (medium) -> 9.5% (long); on TIME 4.3% -> 7.8% -> 13.7%.
- SWA16 costs 1.2% on fev-bench and is tied on short horizons; SWA8 costs 2.0%.
- Our 30M softmax twin (0.413 skill) sits just below TiRex/Toto-22m and above Toto-2.0-4m.
