# Generator gap analysis (2026-10-25)

Full doc: https://claude.ai/artifact/1oSofxgqSUtoCqpFJx5Tuc. Audit script: tools/gen_audit.py (runs ft3_core's generator on CPU).

## Measured with B0 (2026-10-27, ft3_main's real cfg)
B0 = `v0-b0-af75b1af` (configs/ft3_main_cfg.json): clean2 with agg, periods, epidemic, repeat, daytype, weekday, yearly and
oneoff on; breaks, tv_season, sarima, bursty off; cov_r3 on; cov_small_group_p, cov_dominant_p, second_group_p 0; group_prob
0.7 over 8 lanes. Command: `python tools/gen_audit.py --base b0` (n = 4096 per length, torch seed 0; ~1.5 min on CPU).
- Assumptions vs the earlier measurement below: yearly and cov_r3 were assumed on and ARE on. One difference: oneoff is
  also on in the real cfg (the earlier run had it off). oneoff only sets one_off_p = 0.3 (one-off events inside series);
  it does not change the family weights, so the family mix is the same.
- Family mix: unchanged (exact repetition and day types: repeat 9.5 + long_cycle 9.5 + repeat_week 6.3 + daily_weekly 3.2
  = 28.5% of series; ets 10.1, gp_kernel 7.6, ar_p 6.3, tsi 6.3).
- Seasonality: 43-48% of non-flat series seasonal (L = 128 ... 8192); at L = 2048, 94% of those show >= 8 cycles; 0% have
  fewer than 4 cycles. 67% of detected periods sit at a period-bank value.
- Shape: integer-valued 3%, more than 20% zeros 12-13%; at least one jump 23-50% (rising with L); heavy-tailed differences
  (kurtosis > 10) 15-37%.
- Groups: 44% of lanes in a related group; targets with known covariates 7.6% of lanes, with past covariates 3.4%.
  (The fev covariate gap stands: about 7-8% of lanes are covariate-driven targets.)
- B0 is ft3_main's generator mix, not its training history: ft3_main was fine-tuned from ft2_main (finetune_from), which
  was trained on an earlier mix.

## Measured earlier (2026-10-25: clean2 + CONFIG additions, yearly and cov_r3 assumed on, oneoff off)
- Family mix:
  - 36.6% textbook stationary (ets, gp, ar_p, tsi)
  - 28.5% exact repetition and day types (repeat, long_cycle, repeat_week, daily_weekly)
  - 13.3% trend / level / regime
  - 12.2% sparse / counts / events
  - 8.3% low-frequency
- Seasonality: about 46% of series show clear seasonality; of those, ~94% at L=2048 show 8 or more cycles. Except for long_cycle, every family needs 2-3 cycles by construction.
- Counts: 3% of series are integer-valued; 12% have more than 20% zeros.
- Groups: 7.4% of lanes are targets with known covariates, 2.9% with past covariates.

## fev covariate split (published results)
- Covariate tasks (~42) hold 53-69% of a small model's gap to TimeRSI-1.5 (51.8).
- Toto-2.0-22m scores 46.5 on tasks without covariates and 37.6 on tasks with them. Matching the leader on covariate tasks alone would put it at 49.2.
- ft3_main gains +1.7 points from groups.

## Ranked gaps
1. Covariate realism and volume.
   - Scenario generators for load-weather, price-residual load, sales-promo-open-holiday-price and PV-irradiance.
   - A causal-graph covariate generator as in t0 beta/gamma: a random DAG over base-family nodes, lagged nonlinear edges, parents as covariates, descendants and siblings as decoys. Unablated, so test it on its own.
   - Known covariates given with forecast error.
   - Covariate-driven targets on 20-30% of lanes.
2. Real calendar engine: dates, real holidays, moving festivals, DST, paydays and school terms; given as covariates in about half the rows.
3. Few-cycle / seen-once history: long_cycle at ~15% on the calendar engine, mostly in contexts of 2048+. Band 2049+: 0.520 vs Chronos-2 0.479.
4. Structural breaks: the `breaks` switch is off; turn it on at 0.2 with late breaks.
5. Quasi-periodic cycles: `tv_season` is off; add AM/FM, beats and cycles of irregular length.
6. True missing values (2026-10-27, E0.3 audit: real known covariates often have long missing stretches: 22% of fev's
   known-covariate history values are NaN, up to 46-91% in the retail tasks; groups ON forward-fills them today). G6 should
   also generate covariate lanes with gaps, marked unobserved, not only gaps in targets; G1's covariate scenarios inherit
   it.
   Earlier note: training forward-fills gaps instead of masking them. Inference bug: `forecast_left_padded` deletes NaNs (`y = y[np.isfinite(y)]`), which shifts the phase.
7. Counts, intermittency, product life cycles, stockouts.
8. Hierarchies.
9. ODE dynamics.
10. Financial tails.

## Model-side items
- Known-covariate attention over the whole window (TIMESTRAL_GAP #4).
- Static/categorical inputs.
- Calendar input.
- 32K context and DHF.
- 21 quantiles and the refine frame.

## Next
1. ~~Confirm ft3_main's cfg~~ done 2026-10-27 (see Measured with B0).
2. Run the covariate split on our own per-task fev file.
3. Test the cheap switches one at a time.
4. Then the covariate scenarios, the causal-graph generator and the calendar engine.
