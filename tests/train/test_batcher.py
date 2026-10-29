"""T3 batcher port: bit-exact with ft3_core.PackedBatcher (global pools) and tidemark's SyntheticBatcher (seeded pools) for the
same cfg and seed; with every train-side item off and no NaN the batch is the plain port's; NaN -> value 0 / mask 0; crop-time
breaks land in the last 10-30% of the actual context; lead noise only on known lanes from the CPM span; resume reproduces the
next batch (CPU, small layout)."""
import json
from pathlib import Path

import pytest
import torch

import tidemark.real.ft3_core as C
import tidemark.real.sampler as SM
import tideforge.gen as G
from tideforge.gen.config import GenConfig
from tideforge.gen.groups import ROLE_KNOWN
from tideforge.train import batcher as TB

REPO = Path(__file__).resolve().parents[2]
CPU = torch.device("cpu")


def small_cfg(**kw):
    cfg = dict(C.CONFIG, **json.loads((REPO / "configs" / "ft3_main_cfg.json").read_text()))
    cfg.update(pack_rows=16, max_ctx=512, len_bucket_edges=[16, 32, 64, 128, 256, 512], long_min_ctx=256,
               gen_pool_points=2e5, gen_pool_max_rows=1024, max_segs_per_row=8)
    cfg.update(kw)
    return cfg


def ours(cfg, seed=7, **kw):
    prof = G.configure(GenConfig.from_ft3_cfg(cfg))
    return TB.PackedBatcher(cfg, CPU, prof, seed=seed, deterministic_gen=False, **kw)


def same(a, b):
    for k in a:
        if k in ("points", "n_series", "lanes"):
            assert a[k] == b[k], k
        elif torch.is_tensor(a[k]):
            assert torch.equal(a[k], b[k]), k
    assert set(a) - {"loss_ok"} == set(b) - {"loss_ok"}


def test_global_pools_bit_exact_with_ft3_core():
    cfg = small_cfg()
    C.set_group_options(cfg)
    torch.manual_seed(0)
    ref = C.PackedBatcher(cfg, CPU, seed=7)
    rb = [ref.next_batch() for _ in range(3)]
    torch.manual_seed(0)
    mine = ours(cfg, pool_mode="global")
    for r in rb:
        same(mine.next_batch(), r)


def test_seeded_pools_bit_exact_with_tidemark_sampler():
    cfg = small_cfg()
    C.set_group_options(cfg)
    torch.manual_seed(0)
    ref = SM.SyntheticBatcher(cfg, CPU, seed=7, stream=0)
    rb = [ref.next_batch() for _ in range(3)]
    torch.manual_seed(0)
    mine = ours(cfg)
    for r in rb:
        same(mine.next_batch(), r)


def test_items_off_and_deterministic_flag_restored():
    cfg = small_cfg()
    torch.manual_seed(0)
    a = [ours(cfg).next_batch() for _ in range(1)][0]
    torch.manual_seed(0)
    prof = G.configure(GenConfig.from_ft3_cfg(cfg))
    bt = TB.PackedBatcher(cfg, CPU, prof, seed=7, deterministic_gen=True)
    assert not torch.are_deterministic_algorithms_enabled()
    b = bt.next_batch()
    assert not torch.are_deterministic_algorithms_enabled()                   # set around generation only
    same(a, b)


def test_nan_becomes_unobserved(monkeypatch):
    cfg = small_cfg()
    orig = G.generate_blocks

    def with_nan(*a, **k):
        x, r, l = orig(*a, **k)
        x = x.clone()
        x[:, :, 5:9] = float("nan")
        return x, r, l
    monkeypatch.setattr(G, "generate_blocks", with_nan)
    torch.manual_seed(0)
    b = ours(cfg).next_batch()
    assert bool(torch.isfinite(b["x"]).all())
    assert int((~b["mask"]).sum()) > 0
    monkeypatch.setattr(G, "generate_blocks", orig)
    torch.manual_seed(0)
    clean = ours(cfg).next_batch()
    lost = clean["mask"] & ~b["mask"]
    assert int(lost.sum()) > 0 and bool((b["x"][lost] == 0).all())
    assert torch.equal(b["x"][b["mask"]], clean["x"][b["mask"]])            # every other point is untouched


def test_crop_breaks_in_the_last_part_of_the_context():
    cfg = small_cfg(pack_rows=64)
    torch.manual_seed(0)
    bt = ours(cfg, crop_breaks=dict(p=1.0, lo=0.7, hi=0.9))
    tbs, cs = [], []
    for _ in range(3):
        bt.next_batch()
        tb, c = bt.last_breaks
        tbs.append(tb[:, 0])
        cs.append(c.float())
    tb, c = torch.cat(tbs), torch.cat(cs)
    ok = tb >= 0
    frac = tb[ok] / c[ok]
    assert int(ok.sum()) > 50 and bool((frac >= 0.7 - 1e-6).all()) and bool((frac < 0.9 + 1e-6).all())
    torch.manual_seed(0)
    plain = ours(cfg).next_batch()
    torch.manual_seed(0)
    br = ours(cfg, crop_breaks=dict(p=1.0, lo=0.7, hi=0.9)).next_batch()
    assert torch.equal(plain["mask"], br["mask"]) and not torch.equal(plain["x"], br["x"])


def test_lead_noise_only_on_known_lanes_from_the_span():
    cfg = small_cfg(pack_rows=64)
    torch.manual_seed(0)
    plain = ours(cfg).next_batch()
    torch.manual_seed(0)
    nz = ours(cfg, lead_noise=dict(p_noisy=1.0, sigma0_lo=0.5, sigma0_hi=0.5, step=32)).next_batch()
    diff = plain["x"] != nz["x"]
    known = (plain["role"] == ROLE_KNOWN).repeat_interleave(32, dim=1)
    assert int(diff.sum()) > 0 and bool((diff <= known).all())
    R, W = plain["x"].shape
    span = plain["blank"].view(R // 8, 8, W)[:, 0].repeat_interleave(8, dim=0)
    seg = plain["seg_patch"].repeat_interleave(32, dim=1)
    for r in diff.any(1).nonzero().squeeze(1).tolist()[:20]:                 # changes start at the span's first point
        for s in seg[r][diff[r]].unique().tolist():
            m = seg[r] == s
            first_change = int((diff[r] & m).nonzero()[0])
            first_hidden = int((span[r] & m).nonzero()[0])
            assert first_change >= first_hidden
    for k in ("mask", "blank", "role", "seg_patch"):
        assert torch.equal(plain[k], nz[k])


def test_resume_reproduces_the_next_batch():
    cfg = small_cfg()
    torch.manual_seed(0)
    a = ours(cfg)
    for _ in range(4):
        a.next_batch()
    sd, rng = a.state_dict(), torch.get_rng_state()
    want = a.next_batch()
    b = ours(cfg, seed=7)
    b.load_state_dict(sd)
    torch.set_rng_state(rng)
    same(b.next_batch(), want)
