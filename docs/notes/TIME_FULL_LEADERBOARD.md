# TIME: full leaderboard with one Fluxion row, 2026-10-17

40 models (all 39 official result sets on Real-TSF/TIME-Output + fluxion_tide_30m plain), 98 common tasks.
Ranks are among these 40, with ties sharing the average rank. Field win % = (N - rank) / (N - 1).
Script: time_leaderboard_full.py. CSVs: MyDrive/fluxion_time/time_board_full_fluxion_tide_30m.csv, time_h2h_*.csv.

## Leaderboard (excerpt)
| # | Model | MASE (norm.) | CRPS (norm.) | MASE win % | CRPS win % |
|---|---|---|---|---|---|
| 1 | Timestral-209M | 0.626 | 0.522 | 89.6 | 90.9 |
| 2 | Timestral-123M | 0.628 | 0.525 | 89.0 | 89.1 |
| 3 | TimeRSI | 0.630 | 0.528 | 80.8 | 79.8 |
| 4 | t0_gamma | 0.637 | 0.534 | 77.6 | 77.7 |
| 5 | Timestral-28M | 0.638 | 0.534 | 81.7 | 82.4 |
| 8 | TimesFM-3 | 0.640 | 0.536 | 77.7 | 78.3 |
| 12 | Timestral-5M | 0.652 | 0.544 | 69.0 | 72.3 |
| 14 | Chronos-2 | 0.662 | 0.556 | 60.3 | 59.5 |
| 16 | Hart-36M | 0.666 | 0.561 | 55.9 | 55.5 |
| 20 | Toto-2.0-22m | 0.670 | 0.562 | 51.1 | 53.4 |
| **23** | **fluxion_tide_30m** | **0.680** | **0.571** | **45.9** | **48.2** |
| 24 | TiRex | 0.683 | 0.573 | 45.8 | 45.5 |
| 30 | Moirai2 | 0.703 | 0.588 | 33.4 | 34.8 |
| 40 | seasonal_naive | 1.000 | 1.000 | 4.1 | 1.4 |

## Head-to-head (fluxion_tide_30m vs, tasks won % MASE / CRPS, ratio ours/theirs MASE)
- Timestral-28M: 9% / 8%, 1.065. Timestral-5M: 11% / 9%, 1.043 -- a 5M model beats our 30M on ~90% of tasks.
- TimesFM-3: 12% / 14%, 1.063. Chronos-2: 24% / 24%, 1.027. Toto-2.0-22m: 40% / 41%, 1.014.
- TimesFM-2.5: 37% / 53%, 1.017 (CRPS: we win more tasks but lose on average -> a few bad tasks).
- TiRex: 52% / 50%, 0.995. Moirai2: 71% / 73%, 0.967. Chronos-Bolt: 87% / 91%, 0.928.

## Reading
- 23rd of 40. Level with TiRex; ahead of every 2024-25 model (Moirai 2, Toto 1, TimesFM 2.0, Chronos-Bolt);
  behind the 2026 generation.
- Against the top models the gap is BROAD, not a few bad tasks: we beat Chronos-2 on only 24% of tasks and
  Timestral-28M on 9%. The "few bad tasks" pattern holds only against TimesFM-2.5 / Toto-2.0-22m.
- Timestral-5M (0.652, 4.3% better, wins ~90% of tasks) proves the gap is the recipe (objective + data), not size.
  Note: Timestral's submitted numbers include its inference tricks (symclip); like for like with our symclip
  (0.677) the 28M gap is ~6%.
- The small-model niche is crowded: Timestral-5M/28M, Hart-36M, Toto-2.0-22m all beat us.
