# third_party

`tidemark_e21cbf0/` is a verbatim, read-only copy of the parts of the tidemark repo that tideforge builds on, taken at
commit e21cbf0aae4356c371882ab2c60456a3673f6be4 (2026-10-24). Do not edit it: port code from it into `tideforge/`.

Use it by putting this folder on `sys.path`:

    import sys; sys.path.insert(0, "third_party/tidemark_e21cbf0")
    import tidemark.real.ft3_core as C      # ft3 model, generator, packed batcher, group builder, inference

A full clone of the tidemark repo may also sit in `reference/tidemark/` (untracked) for reading, e.g. the Perlmutter
job generator `tidemark/pretrain/slurm.py`. Running code imports only this folder, never `reference/`.

Contents: tidemark/real (ft3_core, loss-masked model, sampler, store, trainer, score_ft3, score_groups, ...),
tidemark/eval (score_twins, fev), tidemark/pretrain (generators, packing, model, inference, probes, train, ...),
tidemark/models and tidemark/utils (dependencies), tidemark/REAL_NOTES.md, scripts/real, scripts/perlmutter (real-track
scripts), tests (the real-track tests and conftest).
