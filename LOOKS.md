# Looks at fev-bench and TIME

Every time a model is scored on fev-bench or TIME, add a row, newest first. Nothing is selected on TIME.

| Date | Experiment | Checkpoint | Benchmark | Why this look | Result |
|---|---|---|---|---|---|
| 2026-10-24 | E0.2 | ft3main_s0 (groups ON, tidemark run) | fev-bench | Phase 0: reproduce ft3_main (tidemark score_groups, size_real gate; the scorer tideforge wraps) | skill 43.8425 |
| 2026-10-24 | E0.2 | ft3main_s0 (groups ON, tidemark run) | TIME | Phase 0: reproduce ft3_main (tidemark score_groups, size_real gate; the scorer tideforge wraps) | MASE 0.6797, CRPS 0.5712 |
| 2026-10-24 | E0.2 | ft3main_s0 (groups hidden, tidemark run) | fev-bench | Phase 0: reproduce ft3_main (tidemark score_groups, size_real gate; the scorer tideforge wraps) | skill 42.0802 |
| 2026-10-24 | E0.2 | ft3main_s0 (groups hidden, tidemark run) | TIME | Phase 0: reproduce ft3_main (tidemark score_groups, size_real gate; the scorer tideforge wraps) | MASE 0.6873, CRPS 0.5752 |
