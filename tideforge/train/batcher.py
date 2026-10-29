"""The packed batcher of the 5M recipe: a port of ft3_core.PackedBatcher (L2682-L2903: _loguniform_int, PackedBatcher,
packed_forward, layout_stats) that draws from tideforge.gen, with tidemark's resumable pools and the T3 train-side items.

Ported verbatim except the lines marked [tf]:
  * data: tideforge.gen.generate_blocks with the profile name tideforge.gen.configure(gc) returned (the generator config, its
    switches included, is set up by the trainer before the batcher is built);
  * pools (pool_mode "seeded", the default): each generation pool is a pure function of (seed, stream, bucket, pool number), as
    tidemark/real/sampler.py SyntheticBatcher (L66-L112); state_dict / load_state_dict keep the pool numbers and pointers, so
    a resumed run regenerates the very pool it was in. pool_mode "global" is ft3_core's own _take (global RNG): only for the
    parity test against ft3_core.PackedBatcher;
  * generation in deterministic mode (owner 2026-10-27, E0.9): torch.use_deterministic_algorithms(True) is set around the
    pool generation only and restored after it. Generation runs on the MAIN thread between training steps, never overlapping
    a model kernel launch (owner 2026-10-29: the flag is process-global, and strict mode raises on ops such as index_add_ /
    scatter backward that have no deterministic implementation);
  * train-side items (all off by default; with all off and no NaN in the blocks the batch is bit-exact with the plain port):
      nan_mask     always on, a no-op without NaN: a non-finite generated value becomes value 0 with mask False, so it is
                   neither input nor target (the loss scores only mask True points; G6 switches write NaN at H8)
      crop_breaks  G4.3 at crop time: dict(p, lo, hi): after the crop, a share p of the series gets g4_breaks.breaks with the
                   break time U(lo, hi) x c, c the series' ACTUAL context (lo 0.7, hi 0.9 = the last 10-30% of it)
      lead_noise   G1.2: dict(p_noisy, sigma0_lo, sigma0_hi, step): on the known lanes of a cell whose target has a CPM span,
                   g1_lead_noise.lead_noise from the first hidden point of that span
      long_row_frac  G3.3: the cfg value (ft3's own setting)
    Their draws come from the batcher's own generator (hook_gen), seeded from (seed, stream, batch number): never torch's global
    stream, so switching one on changes nothing else.
"""
from __future__ import annotations

import hashlib
import math
import time

import torch

import tideforge.gen as G
from tideforge.gen.ext import g1_lead_noise as LN
from tideforge.gen.ext import g4_breaks as GB
from tideforge.gen.groups import ROLE_KNOWN


def _loguniform_int(shape, lo, hi, gen=None):
    """Integers log-uniform on [lo, hi] (CPU)."""
    lo, hi = max(1, int(lo)), max(1, int(hi))
    if hi <= lo:
        return torch.full(shape, lo, dtype=torch.long)
    u = torch.rand(shape, generator=gen)
    v = torch.exp(math.log(lo) + u * (math.log(hi + 1) - math.log(lo)))
    return v.floor().long().clamp_(lo, hi)


class PackedBatcher:
    """ft3_core.PackedBatcher (see its docstring: fixed-shape packed rows, blocks of V lanes sharing one layout, CPM spans)."""

    def __init__(self, cfg, device, profile: str, seed: int, stream: int = 0, weights=None, pool_mode: str = "seeded",  # [tf]
                 deterministic_gen: bool = True, crop_breaks: dict | None = None, lead_noise: dict | None = None):  # [tf]
        self.pl = cfg["patch_len"]
        self.W = cfg["max_ctx"]
        assert self.W % self.pl == 0, "max_ctx must be a multiple of patch_len"
        self.P = self.W // self.pl
        self.R = cfg["pack_rows"]
        self.M = cfg["max_segs_per_row"]
        self.S = self.R * self.M
        self.decoder = cfg.get("decoder", "pool") == "causal"
        self.group = self.decoder and bool(cfg.get("group_attn", False))
        self.V = int(cfg.get("group_lanes", 8)) if self.group else 1
        self.group_prob = float(cfg.get("group_prob", 0.5)) if self.group else 0.0
        assert self.R % self.V == 0, "pack_rows must be a multiple of group_lanes"
        self.Rb = self.R // self.V
        self.Sb = self.Rb * self.M
        self.profile = profile                                                       # [tf] from tideforge.gen.configure
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
        self.pool = {}
        self.gen_seconds = 0.0
        # [tf] resumable pools, deterministic generation, train-side items
        if pool_mode not in ("seeded", "global"):
            raise ValueError(f"pool_mode {pool_mode!r}")
        self.pool_mode, self.seed, self.stream = pool_mode, int(seed), int(stream)
        self.pool_state: dict[int, dict] = {}
        self.deterministic_gen = bool(deterministic_gen)
        self.crop_breaks, self.lead_noise = crop_breaks, lead_noise
        self.n_batches = 0

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
        """(R, W) bool: points hidden from the INPUT (still used as targets); see ft3_core.PackedBatcher._cpm_blank."""
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
    def _pool_seed(self, bi: int, n: int) -> int:                                   # [tf] tidemark sampler.py L73-L74
        return (self.seed * 1_000_003 + self.stream * 10_007 + bi * 101 + n * 13) % (2 ** 62)

    def _generate(self, nb, Lb):                                                      # [tf]
        """generate_blocks on the main thread, in deterministic mode if asked (the flag is restored after)."""
        prev = torch.are_deterministic_algorithms_enabled()
        prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        if self.deterministic_gen:
            torch.use_deterministic_algorithms(True)
        try:
            return G.generate_blocks(nb, self.V, Lb, self.device, self.weights, profile=self.profile,
                                     group_prob=self.group_prob)
        finally:
            torch.use_deterministic_algorithms(prev, warn_only=prev_warn)

    def _make_pool(self, bi: int, n: int, nb: int, Lb: int):                         # [tf] tidemark sampler.py L76-L81
        devs = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(self._pool_seed(bi, n))
            return self._generate(nb, Lb)

    def _take(self, bi, k):
        """k blocks from bucket bi: x (k, V, Lb), role (k, V), lgid (k, V)."""
        E = int(self.edges[bi])
        Lb = E + self.pred
        V = self.V
        t0 = time.time()
        if self.pool_mode == "global":                                                # ft3_core L2809-L2825
            buf, ptr = self.pool.get(bi, (None, 0))
            if buf is None or buf[0].shape[0] - ptr < k:
                nb = max(k, -(-256 // V), min(self.pool_max_rows // V, int(self.pool_pts // (Lb * V))))
                buf = self._generate(nb, Lb)
                ptr = 0
            out = tuple(t[ptr:ptr + k] for t in buf)
            self.pool[bi] = (buf, ptr + k)
        else:                                                                         # [tf] tidemark sampler.py L83-L100
            st = self.pool_state.setdefault(bi, {"n": -1, "ptr": 0, "nb": 0})
            buf = self.pool.get(bi)
            if buf is None and st["n"] >= 0 and st["nb"]:            # resumed: rebuild the pool the saved pointer refers to
                buf = self._make_pool(bi, st["n"], st["nb"], Lb)
            if buf is None or buf[0].shape[0] - st["ptr"] < k:
                nb = max(k, -(-256 // V), min(self.pool_max_rows // V, int(self.pool_pts // (Lb * V))))
                st["n"], st["nb"], st["ptr"] = st["n"] + 1, nb, 0
                buf = self._make_pool(bi, st["n"], nb, Lb)
            ptr = st["ptr"]
            out = tuple(t[ptr:ptr + k] for t in buf)
            st["ptr"] = ptr + k
            self.pool[bi] = buf
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.gen_seconds += time.time() - t0
        return out

    def hook_gen(self) -> torch.Generator:                                           # [tf]
        s = f"{self.seed}|{self.stream}|hooks|{self.n_batches}"
        g = torch.Generator(device=self.device)
        g.manual_seed(int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little") & (2 ** 63 - 1))
        return g

    # ---- full batch -------------------------------------------------------------
    @torch.no_grad()
    def next_batch(self, layout=None):
        lay = layout or self.sample_layout()
        dev, W, pred, V, pl, P = self.device, self.W, self.pred, self.V, self.pl, self.P
        hg = self.hook_gen() if (self.crop_breaks or self.lead_noise) else None       # [tf]
        self.n_batches += 1                                                           # [tf]
        X = torch.zeros(self.R * W, device=dev)
        MK = torch.zeros(self.R * W, dtype=torch.bool, device=dev)
        Y = torch.zeros(self.S, pred, device=dev)
        lane = torch.arange(V, device=dev)
        ROLE = torch.zeros(self.Sb, V, dtype=torch.long, device=dev)
        LG = lane.expand(self.Sb, V).clone()

        c, row, start, n, slot = lay["c"], lay["row"], lay["start"], lay["n"], lay["slot"]
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
            if self.crop_breaks:                                                      # [tf] G4.3 at crop time
                vals = self._crop_breaks(vals, cs, hg)
            fin = torch.isfinite(vals)                                                # [tf] G6: NaN -> value 0, mask 0
            if not bool(fin.all()):
                vals = torch.where(fin, vals, torch.zeros_like(vals))
            dst = first_pt[sel].to(dev).unsqueeze(-1) + j
            X[dst[ok]] = vals[ok]
            MK[dst[ok]] = (fin & ok)[ok]                                              # [tf] was: True
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
            if self.lead_noise:                                                       # [tf] G1.2
                X = self._lead_noise(X.view(self.R, W), MK.view(self.R, W), blank, role_p, seg_full, hg).reshape(-1)
            out = dict(blank=blank, lanes=V, gid=gid, role=role_p,
                       slot_loss=(ROLE != ROLE_KNOWN).reshape(-1))
        return dict(out, x=X.view(self.R, W), mask=MK.view(self.R, W), y=Y,
                    seg_patch=seg_full, slot_valid=lay["slot_valid"].to(dev).repeat_interleave(V),
                    slot_ctx_len=torch.zeros(self.Sb, dtype=torch.long).index_put_(
                        (slot,), c).to(dev).repeat_interleave(V),
                    points=n_ctx + n_tgt, n_series=int(c.numel()) * V)

    # ---- [tf] train-side items ----------------------------------------------------
    def _crop_breaks(self, vals, cs, g):
        """g4_breaks.breaks on every lane of the cropped blocks, the break time U(lo, hi) x c (c = the block's context)."""
        k, V, E = vals.shape
        p = self.crop_breaks
        x = vals.reshape(k * V, E)
        c = cs.view(k, 1).expand(k, V).reshape(-1)
        y, tb = GB.breaks(x, float(p.get("p", 0.2)), g, float(p.get("lo", 0.7)), float(p.get("hi", 0.9)), E=c)
        self.last_breaks = (tb, c)                                                    # for the test
        return y.reshape(k, V, E)

    def _lead_noise(self, X, MK, blank, role_p, seg_full, g):
        """g1_lead_noise's random walk on known lanes, per series, from the first hidden point of its cell's CPM span (the span of
        the block's lane 0, a target; known lanes themselves are never hidden). Vectorised over segments: the walk restarts at
        every series, its scale is sigma0 x the sd of the series' observed points before the origin, Var grows by 1 / step per
        point."""
        R, W = X.shape
        V, pl = self.V, self.pl
        dev = X.device
        p = self.lead_noise
        step = int(p.get("step", 32))
        seg = seg_full.repeat_interleave(pl, dim=1).reshape(-1)                      # series id per point
        known = (role_p == ROLE_KNOWN).repeat_interleave(pl, dim=1).reshape(-1)
        span = blank.view(R // V, V, W)[:, 0].repeat_interleave(V, dim=0).reshape(-1)  # the cell's span, on every lane
        obs = MK.reshape(-1)
        x = X.reshape(-1)
        idx = torch.arange(R * W, device=dev)
        nseg = int(seg.max()) + 1
        BIG = R * W
        hid = known & span
        first = torch.full((nseg,), BIG, device=dev, dtype=torch.long).scatter_reduce(0, seg[hid], idx[hid], "amin")
        o = first[seg]
        has = known & (o < BIG)
        after = has & (idx >= o)
        before = has & (idx < o) & obs
        cnt = torch.zeros(nseg, device=dev).index_add_(0, seg[before], torch.ones(int(before.sum()), device=dev)).clamp(min=1)
        mean = torch.zeros(nseg, device=dev).index_add_(0, seg[before], x[before]) / cnt
        var = torch.zeros(nseg, device=dev).index_add_(0, seg[before], (x[before] - mean[seg[before]]) ** 2) / cnt
        sd = var.sqrt() + 1e-6
        sig = LN.draw_sigma0(nseg, g, float(p.get("p_noisy", LN.P_NOISY)), float(p.get("sigma0_lo", LN.SIGMA0[0])),
                             float(p.get("sigma0_hi", LN.SIGMA0[1])), device=dev)
        z = torch.randn(R * W, generator=g, device=dev) * after
        cz = z.cumsum(0)
        seg_start = torch.full((nseg,), BIG, device=dev, dtype=torch.long).scatter_reduce(0, seg, idx, "amin")
        base = cz[seg_start.clamp(max=BIG - 1)] - z[seg_start.clamp(max=BIG - 1)]
        eps = (cz - base[seg]) * (sig[seg] * sd[seg] / step ** 0.5)
        return torch.where(after & obs, x + eps, x).view(R, W)

    # ---- [tf] resume ---------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"gen": self.gen.get_state(), "pool_state": {int(b): dict(s) for b, s in self.pool_state.items()},
                "n_batches": self.n_batches}

    def load_state_dict(self, sd: dict) -> None:
        if self.pool_mode != "seeded":
            raise ValueError("only seeded pools can be resumed")
        self.gen.set_state(sd["gen"])
        self.pool_state = {int(b): dict(s) for b, s in sd["pool_state"].items()}
        self.n_batches = int(sd["n_batches"])
        self.pool = {}


def packed_forward(model, b):
    """ft3_core.packed_forward (L2883-L2890), causal-decoder branch, through the loss-masked forward."""
    return model(b["x"], mask=b["mask"], seg_patch=b["seg_patch"], slot_valid=b["slot_valid"], blank=b["blank"],
                 lanes=b.get("lanes", 1), gid=b.get("gid"), role=b.get("role"), slot_loss=b.get("slot_loss"),
                 loss_mask=b.get("loss_ok"))


def layout_stats(batcher, n_layouts=20):
    """ft3_core.layout_stats (L2892-L2902): realised length distribution of the sampler (CPU only, no generation)."""
    cs, per_row = [], []
    for _ in range(n_layouts):
        lay = batcher.sample_layout()
        cs.append(lay["c"])
        per_row.append(lay["slot_valid"].view(batcher.Rb, batcher.M).sum(1).float())
    c = torch.cat(cs).float()
    pr = torch.cat(per_row)
    pts_per_step = (float(c.sum()) / n_layouts + pr.sum().item() / n_layouts * batcher.pred) * batcher.V
    return c, pr, pts_per_step
