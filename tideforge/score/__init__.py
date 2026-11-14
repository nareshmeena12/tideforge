"""Fast GPU scorer for tfbench (and any fev task list, e.g. fev-bench): window cache built once through fev's Task API,
predictions pooled across tasks in large batches, fev's own metric classes, group-weighted skill. Entry point:
python -m tideforge.score.run (see run.py)."""
