# Generator gap analysis (2026-10-25)

Full doc: https://claude.ai/artifact/1oSofxgqSUtoCqpFJx5Tuc. Audit script: tools/gen_audit.py (runs ft3_core's generator on CPU).

## Measured (clean2 + CONFIG additions, yearly and cov_r3 assumed on)
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
6. True missing values: training forward-fills gaps instead of masking them. Inference bug: `forecast_left_padded` deletes NaNs (`y = y[np.isfinite(y)]`), which shifts the phase.
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
1. Confirm ft3_main's cfg (gen_additions, cov_r3).
2. Run the covariate split on our own per-task fev file.
3. Test the cheap switches one at a time.
4. Then the covariate scenarios, the causal-graph generator and the calendar engine.
