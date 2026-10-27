"""Generator v0 (port of third_party/tidemark_e21cbf0/tidemark/real/ft3_core.py, sections 3, 3c, 3d) plus switches and new
families (covariate scenarios, causal graph, calendar engine, demand, AM/FM, ...). v0 must stay bit-exact with ft3_core (E0.1).

Like ft3_core, generation uses module state (the active profile, the group options, the installed switches) and torch's global
RNG for v0's own draws: call configure() before generating, from one thread, in every process (DataLoader worker, DDP rank;
WorkerInit does it for a DataLoader)."""
from tideforge.gen import _hooks, switches
from tideforge.gen.batch import _allocate, generate_batch
from tideforge.gen.config import VERSION, GenConfig, b0, load_ft3_cfg
from tideforge.gen.groups import GROUP_KINDS, ROLE_KNOWN, ROLE_PAST, ROLE_TARGET, generate_blocks, set_group_options
from tideforge.gen.profiles import (CLEAN_WEIGHTS, DEFAULT_WEIGHTS, FAMILIES, GEN_ADDITIONS_ALL, GEN_PROFILES,
                                    PASSTHROUGH_FAMILIES, make_clean2_profile, resolve_profile)


def configure(gc: GenConfig, holdout: bool = False, seed: int = 0, stream: int = 0) -> str:
    """Set the module state for gc, as ft3 training does (PackedBatcher.__init__ + set_group_options), and install gc's switches;
    returns the profile name. Idempotent: v0's tables, profiles and hooks are restored first, so configure(b0()) after any
    switch config is v0 exactly. seed / stream seed the switches' own generators (stream: one per worker / rank).
    holdout=True: the holdout stream (val_corpus, v5 group options, no switches), as ft3_core.build_val_batches is called in
    train(); call configure(gc) again before generating training data."""
    switches.restore_v0()
    _hooks.reset(seed, stream)
    cfg = gc.to_ft3_cfg()
    if holdout:
        set_group_options(None)
        return resolve_profile(dict(cfg, corpus=gc.val_corpus or gc.corpus))
    set_group_options(cfg)
    name = resolve_profile(cfg)
    switches.install(gc.switches, name)
    return name


class WorkerInit:
    """DataLoader(worker_init_fn=WorkerInit(gc, seed, rank)): configure() in each worker with stream = rank * workers + id.
    torch's global seed per worker is DataLoader's own (base_seed + worker id). Picklable, so it works with spawn."""

    def __init__(self, gc: GenConfig, seed: int = 0, rank: int = 0, workers_per_rank: int = 1):
        self.gc, self.seed, self.rank, self.wpr = gc, int(seed), int(rank), int(workers_per_rank)

    def __call__(self, worker_id: int) -> None:
        configure(self.gc, seed=self.seed, stream=self.rank * self.wpr + int(worker_id))
