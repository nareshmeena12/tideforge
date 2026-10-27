# Fluxion-Tide on the TIME benchmark, 2026-10-17

TIME (Qiao et al., ICML 2026, arXiv 2602.12147): 50 fresh datasets, 98 tasks (dataset x short/medium/long),
rolling non-overlapping test windows. MASE and CRPS are divided by Seasonal Naive per task, then a geometric
mean over tasks; lower is better. Scored with the benchmark's own Dataset / saver / leaderboard code
(time_bench_full_colab.py). Our settings follow their Chronos-2 script: context 8192, quantiles 0.1-0.9, and
the variates of a multivariate series forecast together (group attention). Missing history values are filled
one season back. Official baseline results come from Real-TSF/TIME-Output.

| Model | MASE (norm.) | CRPS (norm.) | MASE rank | CRPS rank |
|---|---|---|---|---|
| TimesFM-3 | 0.640 | 0.536 | 3.7 | 3.4 |
| Toto-2.0-2.5B | 0.642 | 0.539 | 4.6 | 4.5 |
| Toto-2.0-313m | 0.644 | 0.542 | 5.0 | 5.0 |
| Toto-2.0-1B | 0.645 | 0.545 | 5.0 | 5.3 |
| Chronos-2 | 0.662 | 0.556 | 6.3 | 6.2 |
| TimesFM-2.5 | 0.669 | 0.567 | 7.6 | 8.4 |
| Toto-2.0-22m | 0.670 | 0.562 | 8.4 | 7.9 |
| **Fluxion-Tide 30M (ft3_main, groups)** | **0.680** | **0.571** | 8.8 | 8.3 |
| TiRex | 0.683 | 0.573 | 9.2 | 9.1 |
| Fluxion-Tide 30M, univariate | 0.687 | 0.575 | 10.3 | 9.4 |
| Toto-2.0-4m | 0.691 | 0.582 | 10.8 | 10.5 |
| Toto | 0.698 | 0.584 | 11.6 | 11.1 |
| Moirai2 | 0.703 | 0.588 | 12.0 | 11.6 |
| TimesFM-2.0 | 0.718 | 0.620 | 13.3 | 13.5 |
| Chronos-bolt | 0.733 | 0.620 | 14.1 | 14.6 |
| Kairos | 0.739 | 0.644 | 14.6 | 15.6 |
| sundial_base | 0.758 | 0.663 | 14.7 | 17.0 |
| TimesFM-1.0 | 0.788 | 0.689 | 17.3 | 17.6 |
| visiontspp_base | 0.805 | 0.671 | 15.9 | 15.3 |
| Moirai_base | 0.826 | 0.666 | 17.5 | 16.0 |
| seasonal_naive | 1.000 | 1.000 | 20.0 | 20.6 |

Reading:
- 8th of 20 models (excluding our own univariate row), ahead of TiRex, Moirai 2, Chronos-Bolt, TimesFM 2.0,
  Sundial and Kairos.
- Relative gap: +2.7% MASE / +2.7% CRPS vs Chronos-2, +6.3% / +6.5% vs TimesFM-3. On fev-bench the gap to
  Chronos-2 in relative error is ~5% (skill 44.0 vs 46.6). On fresh data the gap is about half: consistent with
  less train/test overlap favouring the real-data models, but not proof.
- Size-matched competitor: Toto-2.0-22m is 1.5% better on MASE (0.670 vs 0.680) and Toto-2.0-4m is worse.
  The paper has to address this comparison.
- Group attention helps a little here as well: groups 0.680 vs univariate 0.687 (1%).

## Timestral's inference tricks on our model (2026-10-17)
| Variant | MASE (norm.) | CRPS (norm.) |
|---|---|---|
| plain | 0.680 | 0.571 |
| clip (clamp at 0 if the context is >= 0) | 0.680 | 0.571 |
| sym (average with the reflected forecast of -x; 2x inference) | 0.677 | 0.571 |
| symclip (Timestral's defaults) | 0.677 | 0.570 |
- Total gain: -0.4% MASE, -0.2% CRPS. The gap to Chronos-2 is now +2.3% / +2.5%; Toto-2.0-22m (0.670) is still ahead.
- Conclusion: Timestral's lead does NOT come from inference tricks; it is training (DHF, covariate masks,
  real-data post-training at 32K context). sym costs 2x inference for 0.4%: report it, but it is not worth
  making it the default given speed is our selling point.
- To do: the per-term and per-task Chronos-2 head-to-head, the 100M on TIME, and Timestral-28M scored with our harness.
