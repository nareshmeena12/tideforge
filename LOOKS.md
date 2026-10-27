# Looks at fev-bench and TIME

Every time a model is scored on fev-bench or TIME, add a row, newest first. Nothing is selected on TIME.

| Date | Experiment | Checkpoint | Benchmark | Why this look | Result |
|---|---|---|---|---|---|
| 2026-10-27 | E0.3 | none (data only, no model scored) | fev-bench + TIME | NaN audit of inference contexts | TIME: 98 tasks ok, 0 failed; tasks with interior NaN 35; interior NaN 5339254 of 705711921 values; NaN after fill 0 in 0 tasks; all-NaN series 0; covariate NaN (ON) 0; fev: 100 tasks ok, 0 failed; tasks with interior NaN 15; interior NaN 1070612 of 350862231 values; NaN after fill 30820 in 1 tasks; all-NaN series 230; covariate NaN (ON) 35998909 |
| 2026-10-24 | E0.2 | ft3main_s0 (groups ON, tidemark run) | fev-bench | Phase 0: reproduce ft3_main (tidemark score_groups, size_real gate; the scorer tideforge wraps) | skill 43.8425 |
| 2026-10-24 | E0.2 | ft3main_s0 (groups ON, tidemark run) | TIME | Phase 0: reproduce ft3_main (tidemark score_groups, size_real gate; the scorer tideforge wraps) | MASE 0.6797, CRPS 0.5712 |
| 2026-10-24 | E0.2 | ft3main_s0 (groups hidden, tidemark run) | fev-bench | Phase 0: reproduce ft3_main (tidemark score_groups, size_real gate; the scorer tideforge wraps) | skill 42.0802 |
| 2026-10-24 | E0.2 | ft3main_s0 (groups hidden, tidemark run) | TIME | Phase 0: reproduce ft3_main (tidemark score_groups, size_real gate; the scorer tideforge wraps) | MASE 0.6873, CRPS 0.5752 |
