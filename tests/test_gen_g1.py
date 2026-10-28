"""G1 covariate scenarios (G1.3-G1.7), causal graph (G1.9), covariate count (G1.11): nothing drawn when off, the mechanism of
each scenario is present, and the oracle check (a forecast with the true covariates beats one without; the measured ratios
are printed, run with -s; owner 2026-10-28: 0.8 is a check, a gain under 10% is reported, nothing is tuned to pass)."""
import numpy as np
import pytest
import torch

import tideforge.gen as G
from tideforge.diagnose import g1_oracle as O
from tideforge.gen import _hooks, switches
from tideforge.gen import groups as _groups
from tideforge.gen.ext import g1_common as C
from tideforge.gen.ext import g1_events, g1_graph, g1_load, g1_price, g1_pv, g1_retail
from tideforge.gen.groups import ROLE_KNOWN, ROLE_PAST, ROLE_TARGET

B0 = G.b0()
KINDS = ["sc_load", "sc_price", "sc_retail", "sc_pv", "sc_events", "causal_graph"]
BUILD = {"sc_load": (g1_load, 1.0), "sc_price": (g1_price, 1.0), "sc_pv": (g1_pv, 1.0), "sc_retail": (g1_retail, 24.0),
         "sc_events": (g1_events, 24.0)}
L = 2048


@pytest.fixture(autouse=True)
def strict():
    _hooks.STRICT = True
    yield
    _hooks.STRICT = False
    G.configure(B0)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def blocks(gc, N=64, V=8, L_=512, seed=0):
    prof = G.configure(gc)
    torch.manual_seed(seed)
    return G.generate_blocks(N, V, L_, "cpu", profile=prof, group_prob=gc.group_prob), torch.get_rng_state()


def built(name, n=64, dt=None, seed=1):
    m, d = BUILD[name]
    return m.build(gen(seed), n, L, dt or d, "cpu")


# ---------------------------------------------------------------------------------------------------- off / on
@pytest.mark.parametrize("name", KINDS + ["cov_count"])
def test_off_draws_nothing(name):
    assert name in switches.REGISTRY
    (x0, _, _), st0 = blocks(B0)
    assert not _hooks.installed() and not _hooks._gens and _groups.GROUP_KINDS == G.GROUP_KINDS[:5]
    (x1, _, _), st1 = blocks(B0)
    assert torch.equal(x0, x1) and torch.equal(st0, st1)


@pytest.mark.parametrize("name", KINDS)
def test_on_adds_one_kind(name):
    (x, role, lgid), _ = blocks(B0.with_changes(switches={name: {"share": 0.15}}), N=128)
    assert _groups.GROUP_KINDS[:5] == ("mix", "leadlag", "known_cov", "past_cov", "unrelated")
    assert _groups.GROUP_KINDS[5:] == (name,) and _groups.GROUP_KIND_PROBS[5] == 0.15
    assert bool(torch.isfinite(x).all()) and set(role.unique().tolist()) <= {0, 1, 2}


def test_g18_kinds_keep_their_numbers():
    G.configure(B0.with_changes(switches={k: {"share": 0.05} for k in KINDS[:5]}))
    assert _groups.GROUP_KINDS[:5] == ("mix", "leadlag", "known_cov", "past_cov", "unrelated")
    assert sorted(_groups.GROUP_KINDS[5:]) == sorted(KINDS[:5])
    p = np.array(_groups.GROUP_KIND_PROBS)
    assert abs(0.7 * p[5:].sum() / p.sum() - 0.14) < 1e-6           # G1.8: 14.0% of blocks


# ---------------------------------------------------------------------------------------------------- shared pieces
def test_ar1_and_windows():
    x = C.ar1(gen(), (64, 4096), torch.full((64, 1), 0.9), "cpu")
    assert abs(float(x.std()) - 1) < 0.05
    ac = float((x[:, 1:] * x[:, :-1]).mean() / (x * x).mean())
    assert abs(ac - 0.9) < 0.02
    w = C.windows(gen(), 32, 2000, torch.full((32, 1), 0.02), 3, 5, "cpu")
    assert 0.04 < float(w["inside"].float().mean()) < 0.1


def test_assemble_takes_the_first_covariates_in_random_lanes():
    n, V = 256, 8
    z = torch.randn(n, V, 50)
    act = torch.arange(V).view(1, V) < torch.randint(2, V + 1, (n, 1))
    covs = torch.arange(1, 7).float().view(1, 6, 1).expand(n, 6, 50) * 100
    roles = torch.tensor([ROLE_KNOWN, ROLE_PAST, ROLE_KNOWN, ROLE_KNOWN, C.DECOY, ROLE_PAST]).expand(n, -1)
    y, r = C.assemble(gen(), z, act, torch.zeros(n, V, 1), torch.ones(n, V, 1), torch.zeros(n, 50), covs, roles)
    assert bool((r[:, 0] == ROLE_TARGET).all())
    k = act.sum(1) - 1
    first = []
    for i in range(n):
        got = sorted(int(v) // 100 for v in y[i, 1:1 + k[i], 0])
        assert got == list(range(1, min(int(k[i]), 6) + 1)) + [0] * max(0, int(k[i]) - 6) or int(k[i]) > 6
        first.append(int(y[i, 1, 0]) // 100)
    assert len(set(first)) > 3                                   # the lane order is shuffled
    assert set(r[:, 1:].unique().tolist()) <= {ROLE_KNOWN, ROLE_PAST}


def test_weekly_rows_are_sums_of_days_and_counts_stay_integer():
    t, c, r, pick, infos = C.by_resolution(gen(), 32, 100, ((24.0, 7, 1.0),), g1_retail.build, "cpu")
    low = infos[0]["low"].view(-1)
    assert bool((t[low] == torch.round(t[low])).all())
    assert float(c[:, 1].max()) <= 7 and float(c[:, 1].min()) >= 0    # open days per week
    assert float(c[:, 0].max()) <= 1                                   # share of promo days


def test_holiday_source_is_pluggable():
    C.HOLIDAYS["none"] = lambda g, n, gr, lo, hi, dev: torch.zeros_like(gr["day"], dtype=torch.bool)
    try:
        _, c, _, _ = g1_retail.build(gen(), 8, 400, 24.0, "cpu", holidays="none")
        assert float(c[:, 3].abs().sum()) == 0
        _, c, _, _ = g1_retail.build(gen(), 8, 400, 24.0, "cpu")
        assert float(c[:, 3].sum()) > 0
    finally:
        del C.HOLIDAYS["none"]


# ---------------------------------------------------------------------------------------------------- mechanisms
def bins(x, y, k=10):
    q = np.quantile(x, np.linspace(0, 1, k + 1))
    i = np.clip(np.searchsorted(q, x, side="right") - 1, 0, k - 1)
    return np.array([np.median(y[i == b]) if (i == b).any() else np.nan for b in range(k)])


def test_load_is_u_shaped_in_temperature():
    _, _, _, info = built("sc_load", n=96)
    both = ((info["bh"] > 0.01) & (info["bc"] > 0.01)).view(-1).numpy()
    ok = []
    for i in np.nonzero(both)[0]:
        te, ratio = info["Te"][i].numpy(), (info["load"][i] / info["B"][i]).numpy()
        band = float(info["Tc"][i]) - float(info["w"][i]) / 2, float(info["Tc"][i]) + float(info["w"][i]) / 2
        if (te < band[0]).mean() < 0.1 or (te > band[1]).mean() < 0.1:
            continue                    # each arm must hold a full decile of the temperatures seen (else the end bin mixes)
        b = bins(te, ratio)
        ok.append(0 < int(np.nanargmin(b)) < 9 and b[0] > np.nanmin(b) * 1.01 and b[-1] > np.nanmin(b) * 1.01)
    print(f"\nload U-shape: {np.mean(ok):.2f} of {len(ok)} two-sided groups")
    assert len(ok) >= 10 and np.mean(ok) >= 0.8


def test_price_is_convex_with_spikes_high_and_negatives_low():
    t, _, _, info = built("sc_price", n=64)
    conv, spike_ratio, neg_in_low = [], [], []
    for i in range(64):
        r = info["r"][i].numpy()
        keep = ~info["lo"][i].numpy() & (info["spikes"][i].numpy() < 1e-6)
        p = (t[i].numpy() / float(info["S"][i]) / info["F"][i].numpy())[keep]
        b = bins(r[keep], p)
        conv.append(np.nanmean(np.diff(b, 2)) > 0)
        sp = info["spikes"][i].numpy() > 1e-6
        top = info["hi"][i].numpy()
        if sp.any():
            spike_ratio.append((sp & top).sum() / max(1, sp.sum()))
        neg = t[i].numpy() < 0
        if neg.any():
            neg_in_low.append(float(info["lo"][i].numpy()[neg].mean()))
    print(f"\nprice: convex {np.mean(conv):.2f}; spikes starting in the top residual decile (median share) "
          f"(incl. decay tails) {np.median(spike_ratio):.2f}; negative prices inside the low tail {np.mean(neg_in_low):.3f} "
          f"({len(neg_in_low)} groups with negatives)")
    assert np.mean(conv) >= 0.8 and len(neg_in_low) > 5 and np.mean(neg_in_low) > 0.95


def test_price_spikes_start_only_in_the_top_decile():
    _, _, _, info = built("sc_price", n=32)
    imp = torch.diff(info["spikes"], dim=1, prepend=info["spikes"][:, :1]) > 1e-3 * info["spikes"].amax(1, keepdim=True)
    out = float((imp & ~info["hi"]).float().sum()) / max(1.0, float(imp.float().sum()))
    print(f"\nprice: spike starts outside the top residual decile {out:.3f}")
    assert out < 0.05


def test_pv_zero_at_night_and_clipped():
    t, _, _, info = built("sc_pv", n=64)
    night = info["G"] == 0
    assert bool((t[night] == 0).all()) and 0.3 < float(night.float().mean()) < 0.7
    shares = []
    for i in np.nonzero(info["clips"].view(-1).numpy())[0]:
        day = ~night[i]
        cap = float(info["S"][i] * info["clip"][i])
        shares.append(float(torch.isclose(t[i][day], torch.tensor(cap), rtol=1e-5).float().mean()))
    print(f"\nPV: night share {float(night.float().mean()):.2f}; clipping groups at the limit (median share of daytime) "
          f"{np.median(shares):.3f}, min {np.min(shares):.3f}")
    assert len(shares) > 10 and np.min(shares) >= 0.01
    assert not bool(torch.isfinite(info["clip"][~info["clips"]]).any())


def test_retail_promo_uplift_then_dip_and_closed_zeros():
    t, _, _, info = built("sc_retail", n=96)
    assert bool((t[info["opn"] == 0] == 0).all())
    low = info["low"].view(-1)
    assert bool((t[low] == torch.round(t[low])).all())
    up, dip = [], []
    for i in np.nonzero(~low.numpy())[0]:
        ratio = (t[i] / info["base"][i]).numpy()
        o = info["opn"][i].numpy() > 0
        ins, se = info["promo"]["inside"][i].numpy(), info["promo"]["since_end"][i].numpy()
        hol = info["hol"][i].numpy()
        calm = o & ~ins & (se > 30) & ~hol
        after = o & ~ins & (se < 2) & ~hol
        if calm.sum() > 20 and (o & ins).sum() > 5 and after.sum() > 3:
            up.append(ratio[o & ins & ~hol].mean() / ratio[calm].mean())
            dip.append(ratio[after].mean() / ratio[calm].mean())
    print(f"\nretail: promo uplift median {np.median(up):.2f}, post-promo dip median {np.median(dip):.2f} ({len(up)} groups)")
    assert np.median(up) > 1.1 and np.median(dip) < 0.97


def test_events_build_up_uplift_and_decay():
    t, _, _, info = built("sc_events", n=96)
    hi = (info["S"] >= 20).view(-1).numpy()
    ev = info["ev"]
    r_in, r_pre, r_post = [], [], []
    for i in np.nonzero(hi)[0]:
        ratio = (t[i] / info["base"][i]).numpy()
        far = (ev["until"][i].numpy() > 6) & (ev["since_end"][i].numpy() > 6) & ~ev["inside"][i].numpy()
        pos = info["eff"][i].numpy() > 0
        if far.sum() < 20:
            continue
        base = np.median(ratio[far])
        r_in.append(np.median(ratio[ev["inside"][i].numpy() & pos]) / base)
        pre = (ev["until"][i].numpy() == 1) & pos
        post = (ev["since_end"][i].numpy() == 0) & pos
        if pre.any():
            r_pre.append(np.median(ratio[pre]) / base)
        if post.any():
            r_post.append(np.median(ratio[post]) / base)
    print(f"\nevents (positive): inside {np.median(r_in):.2f}, day before {np.median(r_pre):.2f}, day after "
          f"{np.median(r_post):.2f} x the far-from-event level")
    assert np.median(r_in) > 1.15 and np.median(r_pre) > 1.02 and np.median(r_post) > 1.02


def graph_nodes(n=192, seed=3):
    p = G.configure(B0)
    torch.manual_seed(0)
    x = G.generate_batch(n * 8, L, "cpu", profile=p).view(n, 8, L)
    z = (x - x.mean(2, keepdim=True)) / (x.std(2, keepdim=True) + 1e-6)
    return g1_graph.graph(gen(seed), z)


def test_graph_target_follows_its_parents_at_the_drawn_lag():
    X, info = graph_nodes()
    A, lag, form, weak, tg = (info[k] for k in ("A", "lag", "form", "weak", "target"))
    hit = []
    for i in range(X.shape[0]):
        t = int(tg[i])
        for p in A[i, :, t].nonzero().view(-1).tolist():
            ell = int(lag[i, p, t])
            if weak[i, p, t] or form[i, p, t] != 0 or ell < 5:
                continue
            y = X[i, t].numpy()
            xp = X[i, p].numpy()
            right = abs(np.corrcoef(y[ell:], xp[:L - ell])[0, 1])
            w = ell + 150
            wrong = abs(np.corrcoef(y[w:], xp[:L - w])[0, 1])
            hit.append(right > wrong)
    print(f"\ngraph: corr at the drawn lag > corr 150 steps off for {np.mean(hit):.2f} of {len(hit)} strong linear edges")
    assert len(hit) >= 20 and np.mean(hit) >= 0.8


def test_graph_roles_and_descendants_past():
    from tideforge.gen._hooks import Ctx
    p = G.configure(B0.with_changes(switches={"causal_graph": True}))
    torch.manual_seed(0)
    x = G.generate_batch(64 * 8, 256, "cpu", profile=p).view(64, 8, 256)
    mu, sd = x.mean(2, keepdim=True), x.std(2, keepdim=True) + 1e-6
    act = torch.ones(64, 8, dtype=torch.bool)
    y, role = g1_graph.kind((x - mu) / sd, act, mu, sd, Ctx("blocks", torch.device("cpu"), 256, switch="causal_graph"))
    assert bool((role[:, 0] == ROLE_TARGET).all()) and set(role[:, 1:].unique().tolist()) == {ROLE_KNOWN, ROLE_PAST}
    X, info = graph_nodes(64, seed=5)
    assert bool((info["parent"] & info["desc"]).sum() == 0)                 # a DAG: no parent is also a descendant
    assert bool(info["parent"].any(1).all())


# ---------------------------------------------------------------------------------------------------- oracle
@pytest.mark.parametrize("name,dt", [("sc_load", 1.0), ("sc_load", 0.25), ("sc_price", 1.0), ("sc_price", 0.25),
                                     ("sc_pv", 1.0), ("sc_retail", 24.0), ("sc_events", 24.0)])
def test_oracle_true_covariates_help(name, dt):
    t, c, r, info = built(name, n=48, dt=dt)
    rr = O.scenario_ratios(t, c, r, info["gr"])
    med = float(np.nanmedian(rr))
    print(f"\noracle {name} dt {dt}: MAE with / without covariates median {med:.3f} (q25 {np.nanpercentile(rr, 25):.3f}, "
          f"q75 {np.nanpercentile(rr, 75):.3f}); 0.8 check {'met' if med <= 0.8 else 'NOT met'}")
    assert med < 0.9                                                         # gain of at least 10%


def test_oracle_graph_by_role_and_lag():
    X, info = graph_nodes()
    rk = (torch.rand(X.shape[0], X.shape[1], generator=gen(9)) < 0.5).numpy()
    res = O.graph_parent_ratios(X, info, rk)
    print("\ngraph oracle (strong parents, one at a time; median MAE ratio, count):")
    for k in sorted(res):
        print(f"   {k[0]:5s} {k[1]:11s} {np.median(res[k]):.3f}  ({len(res[k])})")
    assert np.median(res[("known", "lag 0")]) < 0.9
    assert np.median(res[("past", "lag 1-127")]) > 0.95                     # past + short lag: nearly useless
    d = O.graph_decoy_gain(X, info)
    print(f"graph decoys: MAE(parents + decoys) / MAE(parents) median {np.median(d):.3f}, q10 {np.percentile(d, 10):.3f}")
    assert np.median(d) > 0.98


# ---------------------------------------------------------------------------------------------------- G1.11
def test_cov_count_distribution_and_other_kinds_unchanged():
    G.configure(B0.with_changes(switches={"cov_count": True, "sc_load": True}))
    N, V = 20000, 8
    g = torch.randint(2, V + 1, (N,))
    kinds = torch.tensor([0, 2, 3, 5]).repeat(N // 4)
    g2 = _hooks.group_size(g, kinds, torch.ones(N, dtype=torch.bool), V, "cpu")
    assert torch.equal(g2[kinds == 0], g[kinds == 0])
    k = (g2[kinds != 0] - 1).numpy()
    share = np.bincount(k, minlength=8)[1:] / len(k)
    print("\ncov_count shares k = 1..7:", np.round(share, 3))
    assert np.abs(share - np.array(g1_graph.COUNT_P)).max() < 0.015
