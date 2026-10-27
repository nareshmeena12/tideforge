"""Generator v0 (port of third_party/tidemark_e21cbf0/tidemark/real/ft3_core.py, sections 3, 3c, 3d) plus switches and new
families (covariate scenarios, causal graph, calendar engine, demand, AM/FM, ...). v0 must stay bit-exact with ft3_core (E0.1).

Like ft3_core, generation uses module state (the active profile, the group options) and torch's global RNG: call configure()
(or resolve_profile + set_group_options) before generating, from one thread."""
from tideforge.gen.batch import _allocate, generate_batch
from tideforge.gen.config import VERSION, GenConfig, b0, load_ft3_cfg
from tideforge.gen.groups import GROUP_KINDS, ROLE_KNOWN, ROLE_PAST, ROLE_TARGET, generate_blocks, set_group_options
from tideforge.gen.profiles import (CLEAN_WEIGHTS, DEFAULT_WEIGHTS, FAMILIES, GEN_ADDITIONS_ALL, GEN_PROFILES,
                                    PASSTHROUGH_FAMILIES, make_clean2_profile, resolve_profile)


def configure(gc: GenConfig, holdout: bool = False) -> str:
    """Set the module state for gc, as ft3 training does (PackedBatcher.__init__ + set_group_options); returns the profile name.
    holdout=True: the holdout stream (val_corpus, v5 group options), as ft3_core.build_val_batches is called in train()."""
    cfg = gc.to_ft3_cfg()
    if holdout:
        set_group_options(None)
        return resolve_profile(dict(cfg, corpus=gc.val_corpus or gc.corpus))
    set_group_options(cfg)
    return resolve_profile(cfg)
