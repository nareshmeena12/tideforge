"""Generator extensions: one module per gap block (families, layers, group kinds, post-processing), each registering its
switches with tideforge.gen.switches.register. Importing this package registers all of them; nothing here is active until a
GenConfig lists the switch. See tideforge/gen/switches.py for the rules (own RNG, NaN only at final_blocks, priorities)."""
from tideforge.gen.ext import dummy  # noqa: F401
