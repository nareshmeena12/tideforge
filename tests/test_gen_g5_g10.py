"""G5.2-G5.4, G7.1-G7.3, G8.1, G9.1-G9.2, G10.1: nothing drawn when off, group-size rules, replace from several families, and
one mechanism check per family (run with -s for the measured numbers). The ODE pool tests build a tiny pool in tmp_path."""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.signal import hilbert

import tideforge.gen as G
from tideforge.gen import _hooks, profiles, switches
from tideforge.gen.ext import _pool
from tideforge.gen.ext import g5_cycles as G5
from tideforge.gen.ext import g7_demand as D
from tideforge.gen.ext import g7_stockout as S
from tideforge.gen.ext import g8_hierarchy as H
from tideforge.gen.ext import g9_tar as TAR
from tideforge.gen.ext import g10_finance as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
import build_ode_pool  # noqa: E402

B0 = G.b0()
NEW = ["amfm", "beats", "irregular_cycles", "demand", "stockout", "hierarchy", "ode", "tar", "garch_jump"]


@pytest.fixture(autouse=True)
def strict():
    _hooks.STRICT = True
    yield
    _hooks.STRICT = False
    G.configure(B0)


@pytest.fixture(scope="module")
def tiny_pool(tmp_path_factory):
    d = tmp_path_factory.mktemp("pool")
    build_ode_pool.main(["--out", str(d), "--manifest-dir", str(d), "--name", "ode_t", "--per-system", "3", "--T", "2048"])
    return dict(pool="ode_t", pool_dir=str(d), manifest_dir=str(d))


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def blocks(gc, N=32, L=256, seed=0):
    prof = G.configure(gc)
    torch.manual_seed(seed)
    return G.generate_blocks(N, 8, L, "cpu", profile=prof, group_prob=gc.group_prob)


# ---------------------------------------------------------------------------------------------------- off / on / hooks
@pytest.mark.parametrize("name", NEW)
def test_off_draws_nothing(name):
    assert name in switches.REGISTRY
    x0 = blocks(B0)[0]
    st0 = torch.get_rng_state()
    assert not _hooks.installed() and not _hooks._gens
    x1 = blocks(B0)[0]
    assert torch.equal(x0, x1) and torch.equal(st0, torch.get_rng_state())


@pytest.mark.parametrize("sw", [{"amfm": True}, {"beats": True}, {"irregular_cycles": True}, {"demand": True},
                                {"demand": {"lifecycle_p": 0.5}}, {"stockout": True}, {"stockout": {"unobserved": True}},
                                {"hierarchy": True}, {"tar": True}, {"garch_jump": True},
                                {"amfm": {"mode": "replace:tsi"}}, {"demand": {"mode": "replace:count+intermittent"}},
                                {"tar": {"mode": "replace:regime_switch+nonlinear_ar"}},
                                {"garch_jump": {"mode": "replace:stochastic_vol"}}])
def test_on_runs(sw):
    x = blocks(B0.with_changes(switches=sw), N=32, L=384)[0]
    nan_ok = "stockout" in sw and isinstance(sw["stockout"], dict)
    assert bool(torch.isfinite(x).all()) or (nan_ok and not bool(torch.isinf(x).any()))


def test_no_rules_is_b0_after_hierarchy():
    x0 = blocks(B0)[0]
    blocks(B0.with_changes(switches={"hierarchy": True, "cov_count": True}))
    assert len(_hooks.GROUP_SIZE) == 2
    x1 = blocks(B0)[0]
    assert not _hooks.GROUP_SIZE and torch.equal(x1, x0)


def test_two_rules_on_one_kind_fail():
    from tideforge.gen.switches import Switch, register, unregister
    register(Switch("t_hsize", "TEST", "", lambda p, inst: inst.set_group_size(lambda g, k, i, V, ctx: g, ("hierarchy",))))
    try:
        with pytest.raises(ValueError, match="already have a size rule"):
            G.configure(B0.with_changes(switches={"hierarchy": True, "t_hsize": True}))
    finally:
        unregister("t_hsize")


def test_rules_touch_only_their_kinds():
    seen = {}
    orig = _hooks.group_size

    def spy(g, kind, is_group, V, device):
        out = orig(g, kind, is_group, V, device)
        seen.update(kind=kind.clone(), g=out.clone(), is_group=is_group.clone())
        return out
    _hooks_gs, _hooks.group_size = _hooks.group_size, spy
    try:
        blocks(B0.with_changes(switches={"hierarchy": {"share": 1.0}, "cov_count": True}), N=256)
    finally:
        _hooks.group_size = _hooks_gs
    names = G.groups.GROUP_KINDS
    k, g = seen["kind"], seen["g"]
    hier = k == names.index("hierarchy")
    assert bool((g[hier] >= 3).all()) and int(hier.sum()) > 20
    cov = torch.isin(k, torch.tensor([names.index("known_cov"), names.index("past_cov")]))
    assert float((g[cov] == 2).float().mean()) > 0.25                       # cov_count: k = 1 in 35%


def test_replace_takes_from_listed_families_in_order():
    prof = G.configure(B0.with_changes(switches={"demand": {"mode": "replace:count+intermittent"}}))
    w = profiles.GEN_PROFILES[prof]["weights"]
    assert w["count"] == 0 and w["intermittent"] == 0 and abs(w["demand"] - 0.05) < 1e-12      # 3.16% < 5%: takes all
    tot = 1.58
    prof = G.configure(B0.with_changes(switches={"amfm": {"mode": "replace:tsi"}}))
    w = profiles.GEN_PROFILES[prof]["weights"]
    assert abs(w["amfm"] - 0.04 * tot) < 1e-9 and abs(w["tsi"] - (0.10 - 0.04 * tot)) < 1e-9
    assert abs(sum(w.values()) - tot) < 1e-9
    assert list(w).index("amfm") == list(w).index("tsi") + 1
    prof = G.configure(B0.with_changes(switches={"amfm": True}))
    w = profiles.GEN_PROFILES[prof]["weights"]
    assert abs(w["amfm"] / sum(w.values()) - 0.04) < 1e-9


def test_add_share_rows():
    prof = G.configure(B0.with_changes(switches={"garch_jump": True}))
    n = dict(G.batch._allocate(10000, profiles.GEN_PROFILES[prof]["weights"]))
    assert abs(n["garch_jump"] - 400) <= 16               # the last family takes the other families' rounding


# ---------------------------------------------------------------------------------------------------- G5
def test_amfm_period_and_amplitude_wander():
    g = gen()
    n, L = 256, 2048
    x, info = G5.amfm_parts(g, n, L, torch.device("cpu"))
    ph = np.unwrap(np.angle(hilbert(np.cos(info["phi"].numpy()), axis=1)), axis=1)
    f = np.diff(ph, axis=1) / (2 * np.pi)
    P0, d, w = info["P0"].numpy().ravel(), info["delta"].numpy().ravel(), info["w"].numpy()
    meas, want = [], []
    for i in np.nonzero((P0 >= 16) & (P0 <= 128))[0]:
        k, cut = int(P0[i]), int(2 * P0[i])
        sm = np.convolve(f[i] * P0[i], np.ones(k) / k, "valid")[cut:-cut]          # one-cycle mean: removes Hilbert ripple
        ws = np.convolve(1 + d[i] * w[i, 1:], np.ones(k) / k, "valid")[cut:-cut]
        meas.append(sm.std())
        want.append(ws.std())
    r = np.array(meas) / np.array(want)
    print(f"\namfm: instantaneous period (Hilbert) relative sd median {np.median(meas):.4f}, from the drawn w {np.median(want):.4f}; "
          f"rows {len(r)}")
    assert len(r) > 30 and np.median(np.abs(r - 1)) < 0.2
    A = info["A"].numpy()
    cv = A.std(1) / A.mean(1)
    print(f"amfm: amplitude CV median {np.median(cv):.3f}")
    assert np.median(cv) > 0.05


def _envelope_peak(x, lo_p, hi_p):
    """Band-pass x (n, L) to periods [lo_p, hi_p] (FFT), Hilbert envelope, the envelope's dominant period."""
    n, L = x.shape
    F_ = np.fft.rfft(x, axis=1)
    fr = np.fft.rfftfreq(L)
    keep = (fr >= 1 / hi_p) & (fr <= 1 / lo_p)
    xb = np.fft.irfft(F_ * keep, n=L, axis=1)
    env = np.abs(hilbert(xb, axis=1))
    env = env - env.mean(1, keepdims=True)
    P = np.abs(np.fft.rfft(env, axis=1)) ** 2
    k = 1 + np.argmax(P[:, 1:], axis=1)
    return L / k


def test_beats_envelope_at_the_beat_period():
    g = gen(1)
    n, L = 512, 4096
    x, info = G5.beats_parts(g, n, L, torch.device("cpu"))
    x = x.numpy()
    tide = info["tide"].numpy().ravel() & (info["h"].numpy().ravel() == 1)
    pk = _envelope_peak(x[tide], 11.0, 13.5)
    print(f"\nbeats: tide rows (1 step / h) envelope period median {np.median(pk):.1f} h (spring-neap {G5.SPRING_NEAP_H:.1f})")
    assert abs(np.median(pk) / G5.SPRING_NEAP_H - 1) < 0.1
    gen_ = ~info["tide"].numpy().ravel()
    P1, P2 = info["P1"].numpy().ravel()[gen_], info["P2"].numpy().ravel()[gen_]
    beat = P1 * P2 / (P2 - P1)
    pk = np.array([_envelope_peak(x[gen_][i:i + 1], P1[i] / 1.2, P2[i] * 1.2)[0] for i in range(gen_.sum())])
    ok = beat < L / 2
    r = pk[ok] / beat[ok]
    print(f"beats: other rows, envelope / beat period median {np.median(r):.3f}, within 15%: {np.mean(abs(r - 1) < 0.15):.2f}")
    assert np.mean(abs(r - 1) < 0.15) > 0.7


def test_irregular_cycle_lengths_and_asymmetry():
    g = gen(2)
    n, L = 512, 2048
    x, info = G5.irregular_parts(g, n, L, torch.device("cpu"))
    sun = info["sun"].numpy().ravel()
    cv = (info["lens"].std(1) / info["lens"].mean(1)).numpy()
    print(f"\nirregular: cycle-length CV sunspot {np.median(cv[sun]):.3f} (uniform 0.78-1.22: 0.127), "
          f"ENSO {np.median(cv[~sun]):.3f} (0.44-1.56: 0.323)")
    assert abs(np.median(cv[sun]) - 0.127) < 0.02 and abs(np.median(cv[~sun]) - 0.323) < 0.04
    up = (np.diff(x.numpy(), axis=1) > 0).mean(1)
    rho = info["rho"].numpy().ravel()
    print(f"irregular: share of rising steps {np.median(up):.3f} (rise fraction median {np.median(rho):.3f})")
    assert np.median(up) < 0.45 and abs(np.median(up - rho)) < 0.03


# ---------------------------------------------------------------------------------------------------- G7
def test_demand_counts_dispersion_and_croston_gaps():
    g = gen(3)
    n, L = 1024, 512
    y, info = D.demand_parts(g, n, L, torch.device("cpu"))
    assert bool((y == torch.round(y)).all() and (y >= 0).all())
    mode = info["mode"].view(-1)
    lam = info["lam"]
    disp = (((y - lam) ** 2).mean(1) / lam.mean(1))
    dp, dn = disp[mode == 1].median().item(), disp[mode == 2].median().item()
    print(f"\ndemand: dispersion index median Poisson {dp:.2f}, NB {dn:.2f}")
    assert abs(dp - 1) < 0.15 and dn > 1.3
    cro = (mode == 0).nonzero().squeeze(1)
    occ = info["occur"][cro].float()
    q = (info["p"][cro] * 1.0).view(-1)
    zero = (y[cro] == 0).float().mean(1)
    print(f"demand: Croston rows zero share median {zero.median():.2f}, 1 - mean occurrence prob {1 - occ.mean():.2f}")
    assert abs(float(zero.mean()) - float(1 - occ.mean())) < 0.02
    gaps = []
    for i in range(min(200, len(cro))):
        t = torch.nonzero(y[cro[i]] > 0).squeeze(1).numpy()
        if len(t) > 5:
            gaps.append(np.diff(t).mean() * float(q[i]))
    print(f"demand: Croston mean gap x p median {np.median(gaps):.2f} (geometric: about 1 / (season x promo) ~ 1)")
    assert 0.6 < np.median(gaps) < 1.4


def test_lifecycle_exact_zeros_and_ramp():
    g = gen(4)
    n, L = 1024, 512
    y, info = D.demand_parts(g, n, L, torch.device("cpu"), lifecycle_p=1.0)
    t = torch.arange(L).view(1, L)
    before = t < info["tau0"]
    after = t >= info["tend"]
    assert bool((y[before] == 0).all()) and bool((y[after] == 0).all())
    inside = (info["tau0"].view(-1) > 0.05 * L)
    print(f"\nlifecycle: launch inside the window {float(inside.float().mean()):.2f}, "
          f"death inside {float(torch.isfinite(info['tend']).float().mean()):.2f}")
    assert 0.4 < float(inside.float().mean()) < 0.6
    env, alive, li = D.life(gen(4), n, L, torch.device("cpu"))
    grow = (t >= li["tau0"]) & (t < torch.minimum(li["td"], li["tend"]))
    d = torch.diff(env, dim=1)
    print(f"lifecycle: envelope non-decreasing on the launch ramp: {float((d[grow[:, 1:]] >= -1e-6).float().mean()):.3f}")
    assert float((d[grow[:, 1:]] >= -1e-6).float().mean()) > 0.999


def test_stockout_runs_and_rebound():
    g = gen(5)
    n, L = 512, 1024
    x = torch.poisson(torch.full((n, L), 6.0), generator=gen(6))
    p = dict(switches.REGISTRY["stockout"].params)
    y, out, reb = S.apply(g, x, p, torch.device("cpu"))
    assert bool((y[out] == 0).all()) and bool((y == torch.round(y)).all())
    ratio = float(y[reb].mean() / x[reb].mean())
    print(f"\nstockout: share of steps out of stock {float(out.float().mean()):.3f}, rebound mean / base {ratio:.2f}")
    assert 1.3 < ratio < 1.8
    yn, outn, _ = S.apply(gen(5), x, dict(p, unobserved=True), torch.device("cpu"))
    assert bool(torch.isnan(yn[outn]).all()) and bool(torch.isfinite(yn[~outn]).all())
    gc = B0.with_changes(switches={"stockout": {"share": 1.0}})
    x0, r0, _ = blocks(B0, N=64)
    x1, r1, _ = blocks(gc, N=64)
    cl = S.count_lanes(x0)
    assert int(cl.sum()) > 0
    changed = ~(x0 == x1).all(-1)
    print(f"stockout: count lanes {int(cl.sum())}, changed {int(changed.sum())}, all changed lanes are count lanes: "
          f"{bool((changed <= cl).all())}")
    x1c = x1[changed]
    assert bool((changed <= cl).all()) and bool((x1c == torch.round(x1c)).all() and (x1c >= 0).all())


# ---------------------------------------------------------------------------------------------------- G8
def test_hierarchy_parent_is_the_exact_sum():
    g = gen(7)
    n, V, L = 512, 8, 256
    z = torch.randn(n, V, L)
    z = (z.cumsum(-1) - z.cumsum(-1).mean(-1, keepdim=True)) / (z.cumsum(-1).std(-1, keepdim=True))
    gsz = torch.randint(3, V + 1, (n,), generator=g)
    p = dict(switches.REGISTRY["hierarchy"].params)
    y, info = H.build(g, z, gsz, torch.device("cpu"), p)
    leaf = info["leaf"]
    s = (y.double() * leaf.unsqueeze(-1)).sum(1)
    rel = ((y[:, 0].double() - s).abs() / s.abs().clamp(min=1e-9)).max().item()
    two = info["two"].view(-1)
    m = (y[two, 1].double() - (y[two].double() * info["to1"][two].unsqueeze(-1)).sum(1)).abs().max().item()
    print(f"\nhierarchy: parent vs sum of leaves, max relative error {rel:.2e}; two-level groups {int(two.sum())}")
    assert rel < 1e-6 and m <= 1e-3 * y[two, 1].abs().max().item()
    allint = (leaf.unsqueeze(-1) <= (y == torch.round(y))).all(-1).all(-1) & info["cnt"].eq(leaf).all(1)
    if bool(allint.any()):
        assert torch.equal(y[allint, 0], (y[allint] * leaf[allint].unsqueeze(-1)).sum(1))

    def rough(v):                                         # step changes relative to the level (positive series)
        return v.diff(dim=-1).std(-1) / (v.mean(-1).abs() + 1e-9)
    rp = rough(y[:, 0])
    # the level-weighted mean of the children's roughness (sum sd(diff) / sum mean): the parent is below it unless every
    # child moves in step (the parts' own changes partly cancel)
    lv = (y * leaf.unsqueeze(-1)).mean(-1)
    rc = (y.diff(dim=-1).std(-1) * leaf).sum(1) / lv.sum(1)
    r = rp / rc
    print(f"hierarchy: parent roughness / children's level-weighted roughness median {r.median():.3f}")
    assert bool((r <= 1 + 1e-4).all()) and float(r.median()) < 0.9


# ---------------------------------------------------------------------------------------------------- G9
def test_ode_pool_sha_refusal(tiny_pool, tmp_path):
    pool = _pool.load(**{"name": tiny_pool["pool"], "pool_dir": tiny_pool["pool_dir"], "manifest_dir": tiny_pool["manifest_dir"]})
    assert pool["traj"].shape == (15, 2048, 3)
    bad = tmp_path / "ode_t"
    bad.mkdir()
    for f in ("traj.npy", "meta.npy"):
        (bad / f).write_bytes((Path(tiny_pool["pool_dir"]) / "ode_t" / f).read_bytes())
    with open(bad / "meta.npy", "r+b") as fh:
        fh.seek(-1, 2)
        fh.write(b"\x01")
    with pytest.raises(RuntimeError, match="refusing"):
        _pool.load("ode_t", str(tmp_path), tiny_pool["manifest_dir"])
    with pytest.raises(RuntimeError, match="no committed manifest"):
        _pool.load("ode_none", str(tmp_path), str(tmp_path))


def test_ode_pool_checks_content_not_file_bytes(tiny_pool, tmp_path):
    """The same arrays written with another .npy header version (what another numpy may do) still load."""
    src = Path(tiny_pool["pool_dir"]) / "ode_t"
    d = tmp_path / "ode_t"
    d.mkdir()
    for f in ("traj.npy", "meta.npy"):
        x = np.load(src / f)
        with open(d / f, "wb") as fh:
            np.lib.format.write_array(fh, x, version=(2, 0))
        assert (d / f).read_bytes() != (src / f).read_bytes()
    pool = _pool.load("ode_t", str(tmp_path), tiny_pool["manifest_dir"])
    assert pool["traj"].shape == (15, 2048, 3)
    man = json.loads((Path(tiny_pool["manifest_dir"]) / "ode_t.json").read_text())
    assert set(man["content"]) == set(man["files"]) == {"traj.npy", "meta.npy"}


def test_ode_pool_deterministic(tiny_pool, tmp_path):
    build_ode_pool.main(["--out", str(tmp_path), "--manifest-dir", str(tmp_path), "--name", "ode_t", "--per-system", "3",
                         "--T", "2048", "--workers", "2"])
    a = (Path(tiny_pool["manifest_dir"]) / "ode_t.json").read_text()
    b = (tmp_path / "ode_t.json").read_text()
    assert a == b


def test_ode_family_runs_and_systems(tiny_pool):
    gc = B0.with_changes(switches={"ode": tiny_pool})
    x = blocks(gc, N=32, L=512)[0]
    assert bool(torch.isfinite(x).all())
    pool = _pool.load(tiny_pool["pool"], tiny_pool["pool_dir"], tiny_pool["manifest_dir"])
    from tideforge.gen.ext import g9_ode
    y, info = g9_ode.ode_rows(gen(8), pool, 256, 512, torch.device("cpu"))
    assert bool(torch.isfinite(y).all())
    pos = torch.isin(info["sid"], torch.tensor([0, 1])) & ~info["is_mix"].view(-1)
    assert int(pos.sum()) > 20 and bool((y[pos] > 0).all())                    # populations stay positive
    meta = pool["meta"]
    lor = meta[meta[:, 0] == 4]
    zmean = lor[:, 5]                                                              # mean of z (channel 2)
    zsd = lor[:, 8]
    print(f"\node: Lorenz z mean {zmean.round(1)}, sd {zsd.round(1)} for rho {lor[:, 10].round(1)} "
          f"(on the attractor: mean about rho - 4.5, not the fixed point rho - 1 with sd 0)")
    assert np.all(np.abs(zmean - (lor[:, 10] - 4.5)) < 2.0) and np.all(zsd > 5)


def test_ode_divergence_lorenz_vs_van_der_pol():
    from scipy.integrate import solve_ivp
    rng = np.random.default_rng(0)
    res = {}
    for sid in (2, 4):
        while True:
            f, y0, dt, burn, par, C, _ = build_ode_pool._draw(sid, np.random.default_rng(rng.integers(1 << 30)))
            if sid != 2 or par[1] == 0.0:
                break
        a = solve_ivp(f, (0, burn), np.asarray(y0, float), method="LSODA", rtol=1e-10, atol=1e-12).y[:, -1]
        T = 2000 * dt
        r1 = solve_ivp(f, (burn, burn + T), a, method="LSODA", rtol=1e-10, atol=1e-12, t_eval=[burn + T]).y[:, -1]
        r2 = solve_ivp(f, (burn, burn + T), a + 1e-8, method="LSODA", rtol=1e-10, atol=1e-12, t_eval=[burn + T]).y[:, -1]
        res[sid] = np.abs(r1 - r2).max() / 1e-8
    print(f"\node: separation growth over 2000 steps: Lorenz {res[4]:.1e}, unforced van der Pol {res[2]:.1e}")
    assert res[4] > 1e4 and res[2] < 1e3


def test_tar_regimes_differ():
    g = gen(9)
    n, T = 256, 4096
    y, info = TAR.tar_raw(g, n, T, torch.device("cpu"))
    y = y.double().numpy()
    hard2 = ((info["form"] == 0) & (info["R"] == 2)).numpy()
    d, c = info["d"].numpy(), info["c"][:, 0].numpy()
    p1 = info["phi1"].numpy()
    errs = []
    for i in np.nonzero(hard2 & (np.abs(p1[:, 0] - p1[:, 1]) > 0.5))[0]:
        z = y[i, 4 - d[i]:T - d[i]]
        Y, X1, X2 = y[i, 4:], y[i, 3:T - 1], y[i, 2:T - 2]
        for r, m in ((0, z <= c[i]), (1, z > c[i])):
            if m.sum() < 200:
                continue
            A = np.stack([X1[m], X2[m], np.ones(m.sum())], 1)
            est = np.linalg.lstsq(A, Y[m], rcond=None)[0][0]
            errs.append(abs(est - p1[i, r]))
    print(f"\ntar: per-regime phi1 estimate error median {np.median(errs):.3f} over {len(errs)} regimes")
    assert len(errs) > 20 and np.median(errs) < 0.1
    assert np.abs(y).max() < 1e4


def test_garch_tails_clustering_leverage():
    g = gen(10)
    n, T = 256, 4096
    r, info = F.garch_raw(g, n, T, torch.device("cpu"))
    r = r.double().numpy()
    rc = r - r.mean(1, keepdims=True)
    kurt = (rc ** 4).mean(1) / (rc ** 2).mean(1) ** 2
    r2 = rc ** 2
    ac = np.array([np.corrcoef(r2[i, 1:], r2[i, :-1])[0, 1] for i in range(n)])
    lev = np.array([np.corrcoef(rc[i, :-1], r2[i, 1:])[0, 1] for i in range(n)])
    print(f"\ngarch_jump: kurtosis median {np.median(kurt):.1f}, acf(r^2, 1) median {np.median(ac):.3f}, "
          f"corr(r_t-1, r_t^2) median {np.median(lev):.3f}, jump steps {info['jumps'].gt(0).float().mean():.4f}")
    assert np.median(kurt) > 3 and np.median(ac) > 0 and np.median(lev) < 0
    out, price = F.to_output(gen(11), torch.as_tensor(r[:, :512], dtype=torch.float32), torch.device("cpu"))
    assert bool((out[price.view(-1)] > 0).all())
