# Real-data post-training of ft3_main: first result (2026-10-24)

## Run
- Start: ft3_main (29M). 30B points total = 9.96B real + ~20B synthetic (ft3_main's own mix), 1:2 per step.
- Real data: chronos_datasets only, fev / GIFT-Eval overlaps removed (TiRex-2 Table 5 as the kept list, plus 9 extra drops). 21 datasets, 14.52B valid points; 96% weather (weatherbench hourly [11 variables, 12-shard sample for the 8 pressure-level ones], daily, weekly). Non-weather capped at 4 epochs, weather share 78% of the real half. chronos_datasets_extra skipped (needs Kaggle).
- Missing values: interior gaps forward-filled (observed flag set), originally-missing positions masked out of the loss.
- Training: 3,219 steps, 288 rows/GPU x 4 GPUs (96 real + 192 synthetic), peak LR 1.2e-4, 5% warmup, cosine, AdamW wd 0.01, clip 1.0, dropout 0.1, ctx 8192, bf16, Perlmutter interactive node.
- Monitor: synthetic holdout 0.2554 (1B smoke) -> 0.2552 (step 800) -> 0.2540 final. No forgetting on synthetic data. Group gain on the synthetic holdout stayed ~5% (CUT 0.2585 vs ON 0.2452).

## Result (repo scorer, univariate / groups-hidden mode)
| Checkpoint | TIME MASE | TIME CRPS | fev skill x100 | fev win rate x100 |
|---|---|---|---|---|
| ft3_main | 0.6873 | 0.5752 | 42.08 | 41.4 |
| real30b | 0.6844 | 0.5732 | 42.51 | 49.6 |
| change | -0.42% | -0.34% | +0.43 | +8.3 |

## Reading
- Gain is small: TIME -0.4% (the pre-set gate was about -1%), fev skill +0.43 points. fev did not drop (gate met), TIME gate not met.
- Win rate rises 8 points while the aggregate moves little: many tasks improve slightly. Win rates depend on the board contents (ft3_main's own win rate moved from 42.3 to 41.4 once real30b joined the board), so only the within-table comparison counts.
- Not yet scored: groups mode (ft3_main: fev 43.78, TIME 0.680 / 0.571), the 10B / 20B checkpoints, per-category breakdown.
- Chronos-2 TIME 0.662 is still ~3.3% ahead; the real-data pool (3/4 weather) does not close the gap on its own.

## Next
1. Score ckpt_10B / ckpt_20B (trajectory), and the groups-mode scorer on all.
2. Per-task / per-category head-to-head (head_to_head.csv, scores_long.csv): where does real data help?
3. Decide: scale the pool vs. use the category analysis to design synthetic generators.

(2026-10-27, owner: the real-data route gave about +0.6 skill; the decision is to focus on the generator. See EXPERIMENTS.md.)
