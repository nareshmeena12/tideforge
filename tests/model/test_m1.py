"""M1 (G1.13): known covariates see the whole window, behind a firewall. CPU, a tiny random group model (3 layers, so paths
through two layers exist). Every leak test runs on the real mask (must hold) AND on a deliberately broken one (must fail), so
a test that can never fail is caught:

  (a) a target's later values never change any earlier output                   broken: target time attention not causal
  (b) a known lane's far future DOES change earlier target outputs (mask)       broken: known_attn off (1-patch look only)
  (c) firewall: no target value ever changes a covariate hidden state           broken: today's symmetric group mask
  (d) a past covariate's future never changes anything earlier                  broken: known -> past allowed
  (e) series packed in one row stay isolated (both directions)                  broken: segment / group-id checks dropped
  (f) forecast_groups == the training forward pass on the same window           (an equality, both modes)
  (g) two targets: B's values after t never change A's outputs up to t          broken: target time full; symmetric group
plus the known_lookahead bound (t + K + 1, exact), off == the port (weights, outputs, loss), the same parameter count (5M:
5,447,968), the copy of _core against the port, the checkpoint loader and the scorer adapter (an M1 checkpoint loaded with
known_attn off scores exactly like the plain model)."""
import numpy as np
import pytest
import torch

import tidemark.real.ft3_core as C
from tideforge import model as TM
from tideforge.model import ft3_model as FM
from tideforge.model import m1 as M1
from tideforge.model.ft3_model import ROLE_KNOWN, ROLE_PAST, ROLE_TARGET

PL = 8
CFG = dict(C.CONFIG, decoder="causal", patch_len=PL, d_model=32, d_kv=8, n_heads=4, d_ff=64, n_layers=3, dropout=0.0,
           group_attn=True, group_lanes=8, max_pass_patches=32)
ROLES = (ROLE_TARGET, ROLE_TARGET, ROLE_PAST, ROLE_KNOWN)        # rows: A, B, past, known
A, B, PAST, KNOWN = range(4)
V, P0, P = 4, 8, 14                                               # group 0: patches 0..7, group 1: 8..13 (packed in one row)
T = 3                                                             # "now": outputs at patches <= T are checked
EPS, BIG = 1e-6, 1e-4


def make(known_attn="mask", K=None, breaks=(), cfg=CFG):
    torch.manual_seed(0)                                          # every variant has the same weights
    m = M1.build_m1(cfg, known_attn, K).eval()
    m.breaks = frozenset(breaks)
    return m


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    W = P * PL
    x = torch.randn(V, W, generator=g).cumsum(1) * 0.3 + torch.randn(V, 1, generator=g)
    seg = torch.empty(V, P, dtype=torch.long)
    seg[:, :P0], seg[:, P0:] = torch.arange(V)[:, None], V + torch.arange(V)[:, None]
    gid = torch.zeros(V, P, dtype=torch.long)
    gid[:, P0:] = 1
    role = torch.tensor(ROLES)[:, None].expand(V, P).contiguous()
    return dict(x=x, mask=torch.ones(V, W, dtype=torch.bool), seg_patch=seg, slot_valid=torch.ones(2 * V, dtype=torch.bool),
                lanes=V, gid=gid, role=role)


@torch.no_grad()
def trace(m, b) -> list[torch.Tensor]:
    """Every hidden state (R, P, D) after each time / group / feed-forward sublayer of every layer, then q (R, P, pl * nq)
    and the frozen stats mu, sd (R, P)."""
    hs, hooks = [], []
    for blk in m.blocks:
        for mod in (blk.time, blk.group, blk.ffn):
            hooks.append(mod.register_forward_hook(lambda mod, i, o: hs.append(o.float())))
    try:
        out = m(b["x"], mask=b["mask"], seg_patch=b["seg_patch"], slot_valid=b["slot_valid"], lanes=b["lanes"],
                gid=b["gid"], role=b["role"])
    finally:
        for h in hooks:
            h.remove()
    R, Pn = b["seg_patch"].shape
    return hs + [out["_q"].reshape(R, Pn, -1), out["_mu_e"][..., None], out["_sd_e"][..., None]]


def changed(m, b, edit, rows, patches) -> float:
    """Largest change, over every traced tensor, at (rows, patches) when `edit` changes b["x"] in place on a copy."""
    b2 = dict(b, x=b["x"].clone())
    edit(b2["x"])
    t1, t2 = trace(m, b), trace(m, b2)
    r, p = torch.as_tensor(rows), torch.as_tensor(patches)
    return max(float((a[r][:, p] - c[r][:, p]).abs().max()) for a, c in zip(t1, t2))


def at(rows, p_from, p_to=P0):
    """An edit: new values on `rows`, patches p_from .. p_to - 1."""
    def f(x):
        x[rows, p_from * PL:p_to * PL] += 5.0 + torch.arange((p_to - p_from) * PL, dtype=x.dtype)
    return f


EARLY = list(range(T + 1))                                       # patches <= T of group 0
ALL_ROWS = list(range(V))


# ------------------------------------------------------------------------------------------------- leak tests (a)-(g)

@pytest.mark.parametrize("breaks,leaks", [((), False), (("target_time_full",), True)])
def test_a_target_future_never_changes_earlier_outputs(breaks, leaks):
    m, b = make(breaks=breaks), batch()
    for row in (A, B):
        d = changed(m, b, at([row], T + 1), ALL_ROWS, EARLY)
        assert (d > BIG) if leaks else (d <= EPS), (row, d)


@pytest.mark.parametrize("known_attn,reaches", [("mask", True), ("off", False)])
def test_b_known_future_reaches_earlier_targets(known_attn, reaches):
    m, b = make(known_attn), batch()
    d = changed(m, b, at([KNOWN], T + 2), [A, B], EARLY)         # beyond the 1-patch look embedding
    assert (d > BIG) if reaches else (d <= EPS), d


@pytest.mark.parametrize("breaks,leaks", [((), False), (("symmetric_group",), True)])
def test_c_firewall_covariates_never_read_targets(breaks, leaks):
    m, b = make(breaks=breaks), batch()
    d = changed(m, b, at([A, B], 0, P), [PAST, KNOWN], list(range(P)))   # every target value; every covariate state
    assert (d > BIG) if leaks else (d <= EPS), d


@pytest.mark.parametrize("breaks,leaks", [((), False), (("known_reads_past",), True)])
def test_d_past_future_never_changes_anything_earlier(breaks, leaks):
    m, b = make(breaks=breaks), batch()
    d = changed(m, b, at([PAST], T + 1), ALL_ROWS, EARLY)
    assert (d > BIG) if leaks else (d <= EPS), d
    assert changed(m, b, at([PAST], T + 1), [A], [T + 1]) > BIG   # it does reach the target at its own patch


@pytest.mark.parametrize("breaks,leaks", [((), False), (("no_isolation",), True)])
def test_e_packed_series_stay_isolated(breaks, leaks):
    m, b = make(breaks=breaks), batch()
    g0, g1 = list(range(P0)), list(range(P0, P))
    d01 = changed(m, b, at(ALL_ROWS, 0, P0), ALL_ROWS, g1)         # group 0 -> group 1 (later in the row)
    d10 = changed(m, b, at(ALL_ROWS, P0, P), ALL_ROWS, g0)         # group 1 -> group 0 (known lanes read ahead)
    if leaks:
        assert d01 > BIG and d10 > BIG, (d01, d10)
    else:
        assert d01 <= EPS and d10 <= EPS, (d01, d10)


@pytest.mark.parametrize("breaks,leaks", [((), False), (("target_time_full",), True), (("symmetric_group",), True)])
def test_g_second_target_never_opens_a_path_to_the_future(breaks, leaks):
    m, b = make(breaks=breaks), batch()
    d = changed(m, b, at([B], T + 1), [A], EARLY)
    assert (d > BIG) if leaks else (d <= EPS), d


def _group(seed=1, Lc=5 * PL, H=2 * PL + 3):
    rng = np.random.default_rng(seed)
    w = lambda n: (rng.normal(size=n).cumsum() * 0.3).astype(np.float32)   # noqa: E731
    kc, kf = w(Lc + H)[:Lc], w(H)
    return dict(targets=[w(Lc), w(Lc)], past=[w(Lc)], known=[(kc, kf)]), H


@pytest.mark.parametrize("known_attn", ["mask", "off"])
def test_f_forecast_groups_equals_the_training_forward(known_attn):
    m = make(known_attn)
    gr, H = _group()
    res = C.forecast_groups(m, [gr], H, torch.device("cpu"))
    q_inf = np.asarray(res[0][1])                                  # (2 targets, nq, H)
    # the same window as one packed training row block: lanes targets, known, past (forecast_groups' order)
    Lc, hp = len(gr["targets"][0]), -(-H // PL)
    nc = Lc // PL
    Pn, R = nc + hp, 4
    x = np.zeros((R, Pn * PL), np.float32)
    obs = np.zeros((R, Pn * PL), bool)
    lanes = [gr["targets"][0], gr["targets"][1], gr["known"][0][0], gr["past"][0]]
    for r, v in enumerate(lanes):
        x[r, :Lc], obs[r, :Lc] = v, True
    x[2, Lc:Lc + H], obs[2, Lc:Lc + H] = gr["known"][0][1], True
    role = torch.tensor([ROLE_TARGET, ROLE_TARGET, ROLE_KNOWN, ROLE_PAST])[:, None].expand(R, Pn).contiguous()
    seg = torch.arange(R)[:, None].expand(R, Pn).contiguous()
    with torch.no_grad():
        out = m(torch.from_numpy(x), mask=torch.from_numpy(obs), seg_patch=seg, slot_valid=torch.ones(2 * R, dtype=torch.bool),
                lanes=R, gid=torch.zeros(R, Pn, dtype=torch.long), role=role)
        allq = m._inv_tf(out["_q"]) * out["_sd_e"][..., None, None] + out["_mu_e"][..., None, None]
        # and the unpacked path (forward without seg_patch: one series per row)
        un = m(torch.from_numpy(x), mask=torch.from_numpy(obs), lanes=R, gid=torch.zeros(R, Pn, dtype=torch.long),
               role=role)["quantile_preds_all"]
    for qa in (allq, un):
        q_tr = qa[:2, nc - 1:nc - 1 + hp].reshape(2, hp * PL, -1)[:, :H].permute(0, 2, 1).numpy()
        np.testing.assert_allclose(q_inf, q_tr, rtol=0, atol=1e-5)


def test_f_known_future_reaches_forecast_groups_only_with_mask():
    gr, H = _group()
    gr2 = dict(gr, known=[(gr["known"][0][0], gr["known"][0][1] + 50.0)])
    for known_attn, moves in (("mask", True), ("off", True)):     # off still sees one patch ahead (the look)
        m = make(known_attn)
        a, b = (np.asarray(C.forecast_groups(m, [g], H, torch.device("cpu"))[0][1]) for g in (gr, gr2))
        assert (np.abs(a - b).max() > BIG) == moves
    # the first horizon step (read at the last context patch): off sees horizon patch 1 only through the look, mask sees all
    late = dict(gr, known=[(gr["known"][0][0], np.concatenate([gr["known"][0][1][:PL], gr["known"][0][1][PL:] + 50.0]))])
    for known_attn, moves in (("mask", True), ("off", False)):
        m = make(known_attn)
        a, b = (np.asarray(C.forecast_groups(m, [g], H, torch.device("cpu"))[0][1])[:, :, :PL] for g in (gr, late))
        assert (np.abs(a - b).max() > BIG) == moves, known_attn


# ------------------------------------------------------------------------------------------------- known_lookahead

@pytest.mark.parametrize("K", [1, 2, 4])
def test_lookahead_bound_is_exact(K):
    """A target at t never sees a known value beyond patch t + K + 1, and does see patch t + K + 1 (t + K + 2 inside group 0)."""
    m, b = make(K=K), batch()
    assert sum(m.bands()) == K
    t = P0 - K - 3
    far = t + K + 2
    assert changed(m, b, at([KNOWN], far, far + 1), [A, B], [t]) <= EPS
    assert changed(m, b, at([KNOWN], far - 1, far), [A, B], [t]) > 10 * EPS        # a 3-hop path at random init: weak, not 0


def test_bands_split_over_layers():
    assert make(K=32).bands() == [11, 11, 10]
    assert make(K=2).bands() == [1, 1, 0]
    assert make().bands() == [None] * 3
    with pytest.raises(ValueError):
        make("off", K=4)


# ------------------------------------------------------------------------------------------------- off == the port

def test_off_is_the_port_and_mask_has_the_same_parameters():
    cfg = dict(CFG)
    torch.manual_seed(0)
    port = FM.build_masked_model(cfg).eval()
    plain = TM.build(dict(cfg, known_attn="off", known_lookahead=None)).eval()
    assert type(plain) is FM.MaskedLossDecoder
    off, mask = make("off"), make("mask")
    for m in (off, mask):
        assert TM.n_params(m) == TM.n_params(port)
        assert list(m.state_dict()) == list(port.state_dict())
        assert all(torch.equal(a, c) for a, c in zip(m.state_dict().values(), port.state_dict().values()))
    plain.load_state_dict(port.state_dict())
    b = batch()
    tgt = dict(blank=torch.zeros_like(b["mask"]))
    tgt["blank"][[A, B], 5 * PL:7 * PL] = True                       # a CPM span on the targets
    with torch.no_grad():
        outs = [m(b["x"], mask=b["mask"], seg_patch=b["seg_patch"], slot_valid=b["slot_valid"], lanes=V, gid=b["gid"],
                  role=b["role"], **tgt) for m in (port, plain, off, mask)]
    for o in outs[1:3]:
        assert torch.equal(o["loss"], outs[0]["loss"]) and torch.equal(o["_q"], outs[0]["_q"])
    assert not torch.equal(outs[3]["_q"], outs[0]["_q"])


def test_mask_copy_of_core_matches_the_port_on_targets_only():
    """With no covariate lanes the M1 masks equal the port's (causal; target <-> target), so the copied _core must give the
    port's numbers: the copy itself is faithful."""
    port, mask = make("off"), make("mask")
    b = batch()
    b["role"] = torch.full_like(b["role"], ROLE_TARGET)
    with torch.no_grad():
        o1, o2 = (m(b["x"], mask=b["mask"], seg_patch=b["seg_patch"], slot_valid=b["slot_valid"], lanes=V, gid=b["gid"],
                    role=b["role"]) for m in (port, mask))
    torch.testing.assert_close(o1["_q"], o2["_q"], rtol=0, atol=1e-6)
    assert torch.equal(o1["loss"], o2["loss"]) or abs(float(o1["loss"] - o2["loss"])) < 1e-6


def test_5m_parameter_count_with_m1():
    from tideforge.train import recipe as RC
    for exp in ("E0.7", "G1.13a", "G1.13c"):
        rc = RC.load(exp)
        m = TM.build(rc.ft3_cfg())
        assert TM.n_params(m) == rc.sections["model"]["params"] == 5_447_968, exp
    assert type(TM.build(RC.load("G1.13a").ft3_cfg())) is M1.M1Decoder
    assert TM.build(RC.load("G1.13c").ft3_cfg()).known_lookahead == 32


def test_options():
    with pytest.raises(NotImplementedError):
        TM.build(CFG, {"known_attn": "whole_window"})
    with pytest.raises(KeyError):
        TM.build(CFG, {"lanes": 16})
    with pytest.raises(ValueError):
        TM.build(CFG, {"known_attn": "off", "known_lookahead": 32})
    with pytest.raises(ValueError):
        TM.build(dict(CFG, group_attn=False), {"known_attn": "mask"})
    from tideforge.expcfg import MODEL_KEYS
    assert set(MODEL_KEYS) == set(TM.OPTIONS) and MODEL_KEYS["known_attn"] == M1.KNOWN_ATTN


# ------------------------------------------------------------------------------------------------- loading and scoring

def _save(tmp_path, name, cfg, sd):
    p = tmp_path / f"{name}.pt"
    torch.save({"cfg": cfg, "model": sd}, p)
    return p


def test_load_builds_from_the_cfg(tmp_path):
    sd = make().state_dict()
    m, cfg, _ = TM.load(_save(tmp_path, "m1", dict(CFG, known_attn="mask", known_lookahead=None), sd), "cpu")
    assert type(m) is M1.M1Decoder and m.known_attn == "mask" and not m.training
    m, _, _ = TM.load(_save(tmp_path, "old", dict(CFG), sd), "cpu")                 # ft3_main.pt: no option keys
    assert type(m) is FM.MaskedLossDecoder
    from tidemark.real import score_ft3
    saved = score_ft3.load_ft3
    with TM.use_loader():
        from tidemark.real.score_ft3 import load_ft3                                 # as score_groups imports it
        assert load_ft3 is TM.load
    assert score_ft3.load_ft3 is saved


def test_m1_checkpoint_with_off_scores_like_the_plain_model(tmp_path):
    """The scorer adapter (probes Ft3 / M5, used by devscore and finish) on groups ON: an M1 checkpoint read with known_attn
    off == the same weights through third_party's plain loader (C.build_model); with mask it differs."""
    import tideforge.probes.p01_covariates as P1
    from tideforge.probes.models import Ft3
    from tidemark.real.score_ft3 import load_ft3
    cfg = dict(C.CONFIG, d_model=32, d_kv=8, n_heads=4, d_ff=64, n_layers=2, max_ctx=512, group_lanes=8, dropout=0.0)
    torch.manual_seed(0)
    sd = M1.build_m1(cfg).state_dict()
    p_mask = _save(tmp_path, "mask", dict(cfg, known_attn="mask", known_lookahead=None), sd)
    p_off = _save(tmp_path, "off", dict(cfg, known_attn="off", known_lookahead=None), sd)
    p_plain = _save(tmp_path, "plain", cfg, sd)
    cell = P1.build(2).cell("lagged_linear@1024")
    q = {k: Ft3(p, "cpu").predict_cell(cell, "on") for k, p in (("mask", p_mask), ("off", p_off))}
    plain = Ft3(p_plain, "cpu")
    plain.model, _, _ = load_ft3(str(p_plain), torch.device("cpu"))                 # third_party's loader: C.build_model
    assert type(plain.model) is C.CausalPatchDecoder
    q_plain = plain.predict_cell(cell, "on")
    assert np.array_equal(q["off"], q_plain)
    assert not np.allclose(q["mask"], q_plain)
    assert np.array_equal(Ft3(p_mask, "cpu").predict_cell(cell, "hidden"), plain.predict_cell(cell, "hidden"))
