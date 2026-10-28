"""Generator extensions: one module per gap block (families, layers, group kinds, post-processing), each registering its
switches with tideforge.gen.switches.register. Importing this package registers all of them; nothing here is active until a
GenConfig lists the switch. See tideforge/gen/switches.py for the rules (own RNG, NaN only at final_blocks, priorities).
g1_lead_noise is not a switch (the batcher applies it in T3)."""
from tideforge.gen.ext import dummy, g3_share, g4_breaks, g6_gaps, g7_integer  # noqa: F401
from tideforge.gen.ext import g1_events, g1_graph, g1_load, g1_price, g1_pv, g1_retail  # noqa: F401
from tideforge.gen.ext import g2_switches  # noqa: F401
