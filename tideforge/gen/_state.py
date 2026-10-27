"""Generator v0: Profile state that generate_batch sets and the families / layers read (module globals in the source).

Ported from third_party/tidemark_e21cbf0/tidemark/real/ft3_core.py L785-L789 (tidemark commit e21cbf0) by tools/port_gen_v0.py; do not edit by hand
(edit and rerun the tool). Edits vs the source: imports only.
Must stay bit-exact with ft3_core (tests/test_gen_parity.py, tests/test_gen_source.py).
"""
# flake8: noqa
import math

import torch
import torch.nn.functional as F

from tideforge.gen.primitives import (  # noqa: F401
    PERIOD_BANK,
)


# >>> ft3_core.py L785-L789
# set by generate_batch() from the active profile (single-threaded generation)
_PERIOD_BANK_ACTIVE = PERIOD_BANK
_PERIOD_EXT_SHARE = None      # None: uniform over the active bank
_TV_SEASON = False
_ONE_OFF_P = 0.0              # share of repeating series that get a one-off anomaly (round 3)
# <<< ft3_core.py
