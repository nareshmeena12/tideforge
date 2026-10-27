"""Packed batches for the twins.

PORTED AS WRITTEN from reference/tide_v5_100m_300b.py section 4: _loguniform_int,
PackedBatcher, packed_forward, layout_stats. Rows of W = 8,192 points tile exactly with
variable-length series; every series' only padding is the left padding inside its first
patch. In group mode rows come in blocks of `group_lanes` rows sharing one layout.

New here (not in the reference):
  slice_batch   one rank's batch split into micro-batches by block rows (gradient
                accumulation): the data never depends on how many rows fit on a GPU.
  DataStream    the training batches of one rank as a function of (seed, rank, step) only
                (owner, 2026-10-15): generator pools and RNG states restart every
                `block_steps` steps from a seed made of (seed, rank, block); a resume inside a
                block regenerates the block's earlier batches and discards them. The data RNG
                is kept apart from the model's (dropout), so every arm with the same seed sees
                the same data, whether or not a job was requeued.
  holdout       the fixed holdout ("clean" corpus, v5 group mix, no long CPM spans): batch i
                depends only on i, so a rank evaluates batches i = rank, rank + world, ...
                and the all-reduced loss is the same for any number of GPUs.
"""
import math
import time

import torch

from tidemark.pretrain.generators import ROLE_KNOWN, generate_blocks, resolve_profile, set_group_options


def _loguniform_int(shape, lo, hi, gen=None):
    """Integers log-uniform on [lo, hi] (CPU)."""
    lo, hi = max(1, int(lo)), max(1, int(hi))
    if hi <= lo:
        return torch.full(shape, lo, dtype=torch.long)
    u = torch.rand(shape, generator=gen)
    v = torch.exp(math.log(lo) + u * (math.log(hi + 1) - math.log(lo)))
    return v.floor().long().clamp_(lo, hi)


class PackedBatcher:
    """Produces fixed-shape packed batches of variable-length series.

    Layout (sampled on CPU, cheap):
      Each of R rows is W = max_ctx points = P patches. Per row we draw up to M context
      lengths c (points), each occupying n = ceil(c / patch_len) patches. Segments are laid
      left-to-right; the first one that overflows is truncated to exactly fill the row, and
      if all M fit the last one is extended to fill it. So rows tile exactly -- no row-level
      padding at all. The only padding is the LEFT padding inside each series' first patch
      (n*patch_len - c points), which is marked mask=False.

    Content (generated on GPU):
      Each series is drawn from a per-bucket pool generated at length bucket_edge + pred_len,
      randomly cropped to c + pred_len, and scattered into its slot in the row.

    Shapes are identical every step (R, W), (R, P), (R*M, pred_len), (R*M,) -- so
    torch.compile never recompiles.
    """

    def __init__(self, cfg, device, weights=None, seed=None):
        self.pl = cfg["patch_len"]
        self.W = cfg["max_ctx"]
        assert self.W % self.pl == 0, "max_ctx must be a multiple of patch_len"
        self.P = self.W // self.pl
        self.R = cfg["pack_rows"]
        self.M = cfg["max_segs_per_row"]
        self.S = self.R * self.M
        self.decoder = cfg.get("decoder", "pool") == "causal"
        # group mode: rows come in BLOCKS of V lanes sharing one layout. The layout is sampled
        # for Rb = R / V block rows; cell (block row, slot) holds V aligned series.
        self.group = self.decoder and bool(cfg.get("group_attn", False))
        self.V = int(cfg.get("group_lanes", 8)) if self.group else 1
        self.group_prob = float(cfg.get("group_prob", 0.5)) if self.group else 0.0
        assert self.R % self.V == 0, "pack_rows must be a multiple of group_lanes"
        self.Rb = self.R // self.V
        self.Sb = self.Rb * self.M
        self.profile = resolve_profile(cfg)
        # causal decoder: a series is ONE sequence; every patch predicts the next, so no
        # separate target block is generated, and a series needs >= 2 patches to carry loss
        self.pred = 0 if self.decoder else cfg["pred_len"]
        self.min_ctx = max(1, int(cfg["min_ctx"]))
        if self.decoder:
            self.min_ctx = max(self.min_ctx, cfg["patch_len"] + 1)
        self.cpm_prob = float(cfg.get("cpm_prob", 0.5))
        self.cpm_max = int(cfg.get("cpm_max_patches", 16))
        self.cpm_min = int(cfg.get("cpm_min_patches", 1))
        self.cpm_suffix = float(cfg.get("cpm_suffix_frac", 0.5))
        self.cpm_long_p = float(cfg.get("cpm_long_p", 0.0) or 0.0)
        self.cpm_long_max = int(cfg.get("cpm_long_max", 32))
        self.long_frac = float(cfg["long_row_frac"])
        self.long_min = min(int(cfg["long_min_ctx"]), self.W)
        edges = sorted({min(int(e), self.W) for e in cfg["len_bucket_edges"]} | {self.W})
        edges = [e for e in edges if e >= self.min_ctx] or [self.W]
        self.edges = torch.tensor(edges, dtype=torch.long)
        self.pool_pts = float(cfg["gen_pool_points"])
        self.pool_max_rows = int(cfg["gen_pool_max_rows"])
        self.weights = weights
        self.device = device
        self.gen = torch.Generator().manual_seed(seed) if seed is not None else None
        self.pool = {}          # bucket index -> [buffer (N, Lb), read pointer]
        self.gen_seconds = 0.0

    # ---- layout ---------------------------------------------------------------
    def sample_layout(self):
        """Layout of the Rb BLOCK rows (every lane of a block uses the same one)."""
        R, M, P, pl = self.Rb, self.M, self.P, self.pl
        c = _loguniform_int((R, M), self.min_ctx, self.W, self.gen)
        long_rows = torch.rand(R, generator=self.gen) < self.long_frac
        c[:, 0] = torch.where(long_rows, _loguniform_int((R,), self.long_min, self.W, self.gen), c[:, 0])

        n = (c + pl - 1) // pl
        prev = n.cumsum(1) - n
        n_eff = torch.where(prev < P, torch.minimum(n, P - prev), torch.zeros_like(n))
        c_eff = torch.minimum(c, n_eff * pl)
        deficit = P - n_eff.sum(1)                  # > 0 only if all M fit with room left
        n_eff[:, -1] += deficit
        c_eff[:, -1] = torch.where(deficit > 0, n_eff[:, -1] * pl, c_eff[:, -1])
        start = n_eff.cumsum(1) - n_eff             # first patch of each segment
        valid = n_eff > 0

        seg_patch = torch.repeat_interleave(torch.arange(R * M), n_eff.reshape(-1)).view(R, P)
        vi = valid.reshape(-1).nonzero().squeeze(1)
        return dict(seg_patch=seg_patch, slot_valid=valid.reshape(-1), slot=vi,
                    row=vi // M, start=start.reshape(-1)[vi], n=n_eff.reshape(-1)[vi],
                    c=c_eff.reshape(-1)[vi])

    # ---- contiguous patch masking (causal decoder) ------------------------------
    def _cpm_blank(self, lay):
        """(R, W) bool: points hidden from the INPUT (still used as targets). For a fraction
        cpm_prob of series with >= 2 patches, one span of k in [1, cpm_max] patches is hidden,
        never the first patch; with prob cpm_suffix it is the series' LAST k patches -- exactly
        the inference situation of forecasting k patches past the end of the context."""
        R, P, pl, S = self.Rb, self.P, self.pl, self.Sb
        n = torch.zeros(S, dtype=torch.long).index_put_((lay["slot"],), lay["n"])
        st = torch.zeros(S, dtype=torch.long).index_put_((lay["slot"],), lay["start"])
        g = self.gen
        on = (torch.rand(S, generator=g) < self.cpm_prob) & (n >= 2)
        cmax, cmin = torch.full_like(n, self.cpm_max), torch.full_like(n, self.cpm_min)
        if self.cpm_long_p > 0:      # the extra random draw happens only when used: the holdout's stream is untouched
            half = torch.div(n - 1, 2, rounding_mode="floor")        # a long span never hides more than half a series
            lng = (torch.rand(S, generator=g) < self.cpm_long_p) & (half > self.cpm_max)
            cmax = torch.where(lng, torch.minimum(half, torch.full_like(n, self.cpm_long_max)), cmax)
            cmin = torch.where(lng, torch.full_like(n, self.cpm_max + 1), cmin)
        kmax = torch.minimum(n - 1, cmax).clamp(min=1)
        kmin = torch.minimum(kmax, cmin).clamp(min=1)
        k = kmin + (torch.rand(S, generator=g) * (kmax - kmin + 1).float()).floor().long()
        k = torch.minimum(k, kmax)
        suffix = torch.rand(S, generator=g) < self.cpm_suffix
        rnd = 1 + (torch.rand(S, generator=g) * (n - k).clamp(min=1).float()).floor().long()
        b0 = torch.where(suffix, n - k, torch.minimum(rnd, n - k))
        sp = lay["seg_patch"]
        rel = torch.arange(P).expand(R, P) - st[sp]
        blank_patch = on[sp] & (rel >= b0[sp]) & (rel < (b0 + k)[sp])
        return blank_patch.repeat_interleave(pl, dim=1)

    # ---- generation pools --------------------------------------------------------
    def _take(self, bi, k):
        """k blocks from bucket bi: x (k, V, Lb), role (k, V), lgid (k, V)."""
        E = int(self.edges[bi])
        Lb = E + self.pred
        V = self.V
        buf, ptr = self.pool.get(bi, (None, 0))
        if buf is None or buf[0].shape[0] - ptr < k:
            nb = max(k, -(-256 // V), min(self.pool_max_rows // V, int(self.pool_pts // (Lb * V))))
            t0 = time.time()
            buf = generate_blocks(nb, V, Lb, self.device, self.weights, profile=self.profile,
                                  group_prob=self.group_prob)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            self.gen_seconds += time.time() - t0
            ptr = 0
        out = tuple(t[ptr:ptr + k] for t in buf)
        self.pool[bi] = (buf, ptr + k)
        return out

    # ---- full batch -------------------------------------------------------------
    @torch.no_grad()
    def next_batch(self, layout=None):
        lay = layout or self.sample_layout()
        dev, W, pred, V, pl, P = self.device, self.W, self.pred, self.V, self.pl, self.P
        X = torch.zeros(self.R * W, device=dev)
        MK = torch.zeros(self.R * W, dtype=torch.bool, device=dev)
        Y = torch.zeros(self.S, pred, device=dev)
        lane = torch.arange(V, device=dev)
        ROLE = torch.zeros(self.Sb, V, dtype=torch.long, device=dev)
        LG = lane.expand(self.Sb, V).clone()

        c, row, start, n, slot = lay["c"], lay["row"], lay["start"], lay["n"], lay["slot"]
        # flat index of the first real point of lane v of each cell: row (block*V + v)
        first_pt = ((row.unsqueeze(1) * V + torch.arange(V).unsqueeze(0)) * W
                    + ((start + n) * pl - c).unsqueeze(1))                       # (K, V)
        bucket = torch.bucketize(c, self.edges)               # smallest edge >= c
        for bi in bucket.unique().tolist():
            sel = (bucket == bi).nonzero().squeeze(1)
            k = sel.numel()
            E = int(self.edges[bi])
            gx, grole, glg = self._take(bi, k)                 # (k, V, E + pred), (k, V), (k, V)
            cs = c[sel].to(dev)
            st = (torch.rand(k, device=dev) * (E - cs + 1).float()).floor().long()  # ONE crop per block
            j = torch.arange(E, device=dev).view(1, 1, E)
            vals = gx.gather(2, (st.view(k, 1, 1) + j).clamp_(max=E + pred - 1).expand(k, V, E))
            ok = (j < cs.view(k, 1, 1)).expand(k, V, E)
            dst = first_pt[sel].to(dev).unsqueeze(-1) + j
            X[dst[ok]] = vals[ok]
            MK[dst[ok]] = True
            sl = slot[sel].to(dev)
            ROLE[sl] = grole
            LG[sl] = glg
            if pred:
                tidx = (st + cs).view(k, 1, 1) + torch.arange(pred, device=dev).view(1, 1, pred)
                Y[(sl.unsqueeze(1) * V + lane.unsqueeze(0)).reshape(-1)] = \
                    gx.gather(2, tidx.expand(k, V, pred)).reshape(k * V, pred)

        n_ctx = int(c.sum()) * V; n_tgt = int(c.numel()) * V * pred
        sp = lay["seg_patch"].to(dev)                                            # (Rb, P) cell ids
        seg_full = (sp.unsqueeze(1) * V + lane.view(1, V, 1)).reshape(self.R, P)
        out = {}
        if self.decoder:
            role_p = ROLE[sp].permute(0, 2, 1).reshape(self.R, P)                # (R, P)
            gid = (sp.unsqueeze(1) * V + LG[sp].permute(0, 2, 1)).reshape(self.R, P)
            blank = self._cpm_blank(lay).to(dev).repeat_interleave(V, dim=0)     # same span for a cell
            blank = blank & ~(role_p == ROLE_KNOWN).repeat_interleave(pl, dim=1)  # known: never hidden
            out = dict(blank=blank, lanes=V, gid=gid, role=role_p,
                       slot_loss=(ROLE != ROLE_KNOWN).reshape(-1))
        return dict(out, x=X.view(self.R, W), mask=MK.view(self.R, W), y=Y,
                    seg_patch=seg_full, slot_valid=lay["slot_valid"].to(dev).repeat_interleave(V),
                    slot_ctx_len=torch.zeros(self.Sb, dtype=torch.long).index_put_(
                        (slot,), c).to(dev).repeat_interleave(V),
                    points=n_ctx + n_tgt, n_series=int(c.numel()) * V)


def packed_forward(model, b):
    if "blank" in b:            # causal decoder batch
        return model(b["x"], mask=b["mask"], seg_patch=b["seg_patch"],
                     slot_valid=b["slot_valid"], blank=b["blank"], lanes=b.get("lanes", 1),
                     gid=b.get("gid"), role=b.get("role"), slot_loss=b.get("slot_loss"))
    return model(b["x"], target=b["y"], mask=b["mask"],
                 seg_patch=b["seg_patch"], slot_valid=b["slot_valid"])


def layout_stats(batcher, n_layouts=20):
    """Realised length distribution of the sampler (CPU only, no generation)."""
    cs, per_row = [], []
    for _ in range(n_layouts):
        lay = batcher.sample_layout()
        cs.append(lay["c"])
        per_row.append(lay["slot_valid"].view(batcher.Rb, batcher.M).sum(1).float())
    c = torch.cat(cs).float()
    pr = torch.cat(per_row)
    pts_per_step = (float(c.sum()) / n_layouts + pr.sum().item() / n_layouts * batcher.pred) * batcher.V
    return c, pr, pts_per_step



# ============================== new: slicing ================================

def slice_batch(b: dict, M: int, k: int, n: int) -> dict:
    """Part k of n of a packed group batch, cut by BLOCK rows (each block = `lanes` rows that
    share a layout). Series ids, group ids and the per-slot arrays are re-based, so the part
    is an ordinary packed batch; points / n_series are the part's own."""
    if n == 1:
        return b
    V = b["lanes"]
    R = b["x"].shape[0]
    Rb = R // V
    if Rb % n:
        raise ValueError(f"{Rb} block rows cannot be split into {n} equal parts")
    r0, r1 = (Rb // n) * k * V, (Rb // n) * (k + 1) * V
    s0, s1 = (r0 // V) * M * V, (r1 // V) * M * V
    out = dict(b)
    for key in ("x", "mask", "blank", "role"):
        out[key] = b[key][r0:r1]
    out["seg_patch"] = b["seg_patch"][r0:r1] - s0
    out["gid"] = b["gid"][r0:r1] - s0
    for key in ("slot_valid", "slot_ctx_len", "slot_loss"):
        out[key] = b[key][s0:s1]
    out["y"] = b["y"][s0:s1]
    valid = out["slot_valid"]
    out["points"] = int(out["slot_ctx_len"][valid].sum())
    out["n_series"] = int(valid.sum())
    return out


def loss_series(b: dict, patch_len: int) -> int:
    """Number of series that carry loss in a packed batch, by the model's own rule (a patch
    with >= 1 observed input point so far whose NEXT patch of the same series has an observed
    target point; slot_valid; not a known covariate). It equals the model's loss_n without
    a forward pass, so a batch cut into micro-batches is weighted exactly like one batch."""
    R, W = b["x"].shape
    P = W // patch_len
    seg = b["seg_patch"]
    obs_in = (b["mask"] & ~b["blank"]).view(R, P, patch_len).any(-1)
    cs = obs_in.long().cumsum(1)
    pidx = torch.arange(P, device=seg.device).expand(R, P)
    first = torch.where(torch.cat([torch.ones_like(seg[:, :1], dtype=torch.bool), seg[:, 1:] != seg[:, :-1]], 1),
                        pidx, torch.zeros_like(pidx)).cummax(1).values           # series start of each patch
    before = torch.where(first > 0, cs.gather(1, (first - 1).clamp(min=0)), torch.zeros_like(cs))
    seen = (cs - before) > 0
    tgt = b["mask"].view(R, P, patch_len)[:, 1:].any(-1)
    pair = (seg[:, 1:] == seg[:, :-1]) & seen[:, :-1] & tgt
    S = b["slot_valid"].shape[0]
    has = torch.zeros(S, dtype=torch.long, device=seg.device).index_add_(
        0, seg[:, :-1][pair], torch.ones(int(pair.sum()), dtype=torch.long, device=seg.device)) > 0
    has = has & b["slot_valid"]
    if b.get("slot_loss") is not None:
        has = has & b["slot_loss"]
    return int(has.sum())


# =========================== new: data stream ===============================

def _mix(*xs: int) -> int:
    h = 0
    for x in xs:
        h = (h * 1_000_003 + int(x) + 1) % (2 ** 61 - 1)
    return h


def _fork(device):
    devs = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    return torch.random.fork_rng(devices=devs)


class DataStream:
    """Training batches of one rank: batch(step) depends only on (seed, rank, step).
    rows = this rank's rows per step (global_rows / world)."""

    def __init__(self, cfg, device, rank: int, world: int):
        rows = cfg.train.global_rows // world
        if rows * world != cfg.train.global_rows or rows % cfg.model.group_lanes:
            raise ValueError(f"global_rows {cfg.train.global_rows} does not split into {world} ranks "
                             f"of whole {cfg.model.group_lanes}-lane blocks")
        self.cfg, self.device, self.rank, self.world, self.rows = cfg, device, rank, world, rows
        self.gcfg = cfg.gen_cfg(rows)
        self.block = cfg.data.block_steps
        self._blk = None
        self._next = None
        self.gen_seconds = 0.0

    def seed_of(self, block: int) -> int:
        return _mix(self.cfg.seed, self.rank, block)

    def _start(self, blk: int):
        seed = self.seed_of(blk)
        self.batcher = PackedBatcher(self.gcfg, self.device, seed=seed % (2 ** 31))
        with _fork(self.device):
            torch.manual_seed(seed)
            self._cpu = torch.get_rng_state()
            self._cuda = torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None
        self._blk = blk

    def _gen(self) -> dict:
        t0 = time.time()
        with _fork(self.device):
            torch.set_rng_state(self._cpu)
            if self._cuda is not None:
                torch.cuda.set_rng_state(self._cuda, self.device)
            set_group_options(self.gcfg)
            b = self.batcher.next_batch()
            self._cpu = torch.get_rng_state()
            if self._cuda is not None:
                self._cuda = torch.cuda.get_rng_state(self.device)
        self.gen_seconds += time.time() - t0
        return b

    def batch(self, step: int) -> dict:
        """The batch of 0-based step `step`. Sequential calls cost one batch each; a jump
        (resume) restarts the block and regenerates the skipped batches."""
        blk = step // self.block
        if blk != self._blk or step != self._next:
            self._start(blk)
            for _ in range(blk * self.block, step):
                self._gen()
        b = self._gen()
        self._next = step + 1
        return b


# ============================== new: holdout ================================

HOLDOUT_POOL_POINTS = 2e6        # smaller generation pools than training: the holdout is small


def holdout_batch(cfg, device, i: int) -> dict:
    """Holdout batch i (depends only on the config and i)."""
    d = cfg.data
    vcfg = dict(cfg.gen_cfg(d.holdout_batch_rows), corpus=d.val_corpus, group_prob=d.val_group_prob,
                cpm_long_p=0.0, gen_pool_points=min(HOLDOUT_POOL_POINTS, d.gen_pool_points))
    seed = _mix(d.holdout_seed, i)
    with _fork(device):
        torch.manual_seed(seed)
        set_group_options(None)                  # the holdout always uses the v5 group mix
        b = PackedBatcher(vcfg, device, seed=seed % (2 ** 31)).next_batch()
    return b


def holdout_shard(cfg, device, rank: int = 0, world: int = 1) -> list[dict]:
    n = cfg.data.holdout_rows // cfg.data.holdout_batch_rows
    return [holdout_batch(cfg, device, i) for i in range(rank, n, world)]
