"""5M softmax recipe (ab5m_cpm), run configs with generator version and switches in the run id, Perlmutter launchers.

Requirements fixed before the T3 port (owner, 2026-10-27):
- The packed batcher maps NaN in generated blocks to "unobserved" (value 0, mask 0) only when NaN is present: on NaN-free
  input it does nothing (no random draws, identical tensors), so v0 batches stay bit-exact with ft3_core's PackedBatcher.
  NaN can only come from a switch at final_blocks (H8, tideforge/gen/_hooks.py).
- tideforge.gen.configure(gc, seed=..., stream=...) is called in every DataLoader worker and DDP rank (gen.WorkerInit).
"""


def thirdparty_ft3():
    """ft3_core from the pinned third_party copy (sys.path set up the way tideforge.eval.thirdparty does)."""
    from tideforge.eval import thirdparty
    return thirdparty.ft3_core()
