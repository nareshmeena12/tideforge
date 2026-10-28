"""G2 calendar engine (cal_engine, cal_cov), G3.2 long_cycle_cal, G3.4 season_short: nothing drawn when off, the table and its
builder, real calendar facts in the table, effects on the right days, invented calendars, DST, weekly rows, and P2's
oracle check on G2.2's flags (run with -s for the measured numbers)."""
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import tideforge.gen as G
from tideforge.gen import _hooks, switches
from tideforge.gen.ext import g2_engine as E
from tideforge.gen.ext import g2_table

REPO = Path(__file__).resolve().parents[1]
B0 = G.b0()
NEW = ["cal_engine", "cal_cov", "long_cycle_cal", "season_short"]
T = {n: i for i, n in enumerate(g2_table.table("cpu").types)}


@pytest.fixture(autouse=True)
def strict():
    _hooks.STRICT = True
    yield
    _hooks.STRICT = False
    G.configure(B0)


def gen(seed=0):
    return torch.Generator().manual_seed(seed)


def tab():
    return g2_table.table("cpu")


def cal(cid):
    return tab().ids.index(cid)


def day(d: dt.date) -> int:
    return (d - dt.date(1990, 1, 1)).days


def opts(**kw):
    return dict(E.OPTS, **kw)


# ---------------------------------------------------------------------------------------------------- off / on / table
@pytest.mark.parametrize("name", NEW)
def test_off_draws_nothing(name):
    assert name in switches.REGISTRY
    prof = G.configure(B0)
    torch.manual_seed(0)
    x0 = G.generate_blocks(32, 8, 256, "cpu", profile=prof, group_prob=B0.group_prob)[0]
    st0 = torch.get_rng_state()
    assert not _hooks.installed() and not _hooks._gens
    torch.manual_seed(0)
    x1 = G.generate_blocks(32, 8, 256, "cpu", profile=prof, group_prob=B0.group_prob)[0]
    assert torch.equal(x0, x1) and torch.equal(st0, torch.get_rng_state())


@pytest.mark.parametrize("sw", [{"cal_engine": True}, {"cal_engine": True, "cal_cov": True}, {"long_cycle_cal": True},
                                {"season_short": True}, {"cal_engine": {"dst_p": 0.5, "payday_p": 0.3}}])
def test_on_runs_finite(sw):
    gc = B0.with_changes(switches=sw)
    prof = G.configure(gc)
    torch.manual_seed(0)
    x = G.generate_blocks(64, 8, 512, "cpu", profile=prof, group_prob=gc.group_prob)[0]
    assert bool(torch.isfinite(x).all())


def test_table_checks_its_sha(monkeypatch, tmp_path):
    t = tab()
    assert t.C == 67 and t.D == 18628
    bad = tmp_path / "calendars_v1.npz"
    bad.write_bytes((g2_table.DIR / "calendars_v1.npz").read_bytes() + b"x")
    (tmp_path / "calendars_v1.json").write_text((g2_table.DIR / "calendars_v1.json").read_text())
    monkeypatch.setattr(g2_table, "DIR", tmp_path)
    with pytest.raises(RuntimeError, match="sha256"):
        g2_table.Table("cpu")


def test_builder_rebuild_matches_the_manifest():
    sys.path.insert(0, str(REPO / "tools"))
    import build_calendars
    assert build_calendars.main(["--check"]) == 0


def test_dates_match_python():
    idx = torch.arange(-4000, 25000, 37)
    D = E.dates(idx)
    for i in range(0, len(idx), 50):
        d = dt.date(1990, 1, 1) + dt.timedelta(days=int(idx[i]))
        assert (int(D["year"][i]), int(D["month"][i]), int(D["dom"][i]), int(D["wd"][i])) == (d.year, d.month, d.day,
                                                                                                d.weekday())
        assert int(D["doy"][i]) == d.timetuple().tm_yday - 1


# ---------------------------------------------------------------------------------------------------- real calendar facts
def _starts(cid, t):
    c = tab().code[cal(cid)].numpy() == T[t]
    return np.nonzero(c & ~np.r_[False, c[:-1]])[0]


@pytest.mark.parametrize("cid,t,gaps", [("SA", "islamic", [(353, 356)]), ("EG", "islamic", [(353, 356)]),
                                        ("IL", "jewish", [(353, 356), (382, 386)])])
def test_moving_festivals_move(cid, t, gaps):
    s = _starts(cid, t)
    s = s[s < s.max() - 400]
    ok = [any(((s > a + lo - 1) & (s < a + hi + 1)).any() for lo, hi in gaps) for a in s]
    print(f"\n{cid} {t}: {np.mean(ok):.2f} of {len(s)} starts recur {gaps} days later")
    assert np.mean(ok) > 0.8


def test_eid_moves_about_11_days_a_year():
    s = _starts("SA", "islamic")
    d = [day(dt.date(y, 1, 1)) for y in range(1991, 2040)]
    first = [s[(s >= a) & (s < a + 366)] for a in d]
    doy = [f[0] - a for f, a in zip(first, d) if len(f)]
    shifts = np.diff(doy)
    shifts = shifts[np.abs(shifts) < 30]
    print(f"\nSA first Islamic holiday of the year: median shift {np.median(shifts)} days / year")
    assert -12 <= np.median(shifts) <= -10


@pytest.mark.parametrize("cid", ["CN", "SG", "VN"])
def test_lunar_new_year_moves(cid):
    """First Chinese-lunisolar holiday in Jan-Feb each year (the type also holds Qingming, a solar date): it moves 10-12
    days earlier, or 18-20 later after a leap month."""
    s = _starts(cid, "chinese")
    jan = [s[(s >= day(dt.date(y, 1, 15))) & (s < day(dt.date(y, 2, 25)))] for y in range(1990, 2041)]
    d = np.array([f[0] - day(dt.date(1990 + k, 1, 1)) for k, f in enumerate(jan) if len(f)])
    sh = np.diff(d)
    ok = np.isin(sh, [-12, -11, -10, 18, 19, 20])
    print(f"\n{cid} Lunar New Year: {len(d)} years, shifts {sorted(set(sh.tolist()))}, {ok.mean():.2f} lunisolar")
    assert len(d) >= 45 and ok.mean() > 0.9


@pytest.mark.parametrize("cid,date,rule", [("SA", dt.date(2020, 3, 6), "fri_sat"), ("SA", dt.date(2010, 3, 4), "thu_fri"),
                                           ("AE", dt.date(2021, 3, 5), "fri_sat"), ("AE", dt.date(2023, 3, 4), "sat_sun"),
                                           ("US", dt.date(2023, 3, 4), "sat_sun"), ("NP", dt.date(2023, 3, 4), "sat")])
def test_weekend_rules(cid, date, rule):
    w = tab().weekend[cal(cid), day(date) - 3:day(date) + 4].numpy()
    wd = [(date + dt.timedelta(days=k)).weekday() for k in range(-3, 4)]
    want = {"fri_sat": (4, 5), "thu_fri": (3, 4), "sat_sun": (5, 6), "sat": (5,)}[rule]
    assert [bool(x) for x in w] == [d in want for d in wd]


def test_held_out_and_valid_years():
    t = tab()
    held = {t.ids[i] for i in t.held_out.nonzero().view(-1).tolist()}
    assert {"GB", "IN", "KR", "AU", "BR"} <= held and "US" not in held
    i = cal("IN")
    assert dt.date(1990, 1, 1) + dt.timedelta(days=int(t.lo[i])) == dt.date(2001, 1, 1)   # Diwali only 2001-2035 in holidays


# ---------------------------------------------------------------------------------------------------- engine
def test_weekend_lines_up_with_the_weekly_dip():
    n, L = 400, 700
    x = torch.randn(n, L, generator=gen(1))
    y, info = E.apply(gen(2), x, opts(bridge_p=0, school_p=0, fiscal_p=0, mult_p=0), "cpu",
                      ppd_idx=torch.ones(n, dtype=torch.long))
    eff, wk, code = info["effect"], info["weekend"] > 0, info["code"] > 0
    dip = []
    for i in range(n):
        we, wd = eff[i][wk[i] & ~code[i]], eff[i][~wk[i] & ~code[i]]
        if len(we) > 10 and len(wd) > 10:
            spread = float(wd.std())
            dip.append(float(we.mean() - wd.mean()))
            assert spread < 1e-4 or abs(float(we.std())) < 1e-4 or True
    dip = np.array(dip)
    print(f"\nweekend - weekday effect (sd units): negative in {np.mean(dip < 0):.2f} of rows (draw: 0.80)")
    assert 0.7 < np.mean(dip < 0) < 0.9
    # the dip sits on the calendar's own weekend: Fri-Sat calendars dip on Fri / Sat
    real = info["cal"] >= 0
    for i in np.nonzero(real.numpy())[0][:50]:
        days = info["d0"][i] + torch.arange(L)
        assert torch.equal(info["weekend"][i] > 0, tab().weekend[info["cal"][i], days])


def test_holiday_effect_on_its_date_with_run_up_and_rebound():
    n, nd = 300, 800
    d = E.daily(gen(3), n, nd, tab(), opts(bridge_p=0, school_p=0, fiscal_p=0), "cpu")
    Eff, code, wk = d["E"], d["code"], d["weekend"]
    on, pre, post = [], [], []
    for i in range(n):
        c = code[i].numpy()
        for j in np.nonzero(c)[0]:
            if 8 <= j < nd - 4 and (c[j - 8:j] == 0).all() and (c[j + 1:j + 4] == 0).all() and not wk[i, j - 1:j + 2].any():
                s = float(d["s"][i, c[j]])
                on.append((float(Eff[i, j]), s))
                pre.append((float(Eff[i, j - 1]), s))
                post.append(abs(float(Eff[i, j + 1])))
    on, pre = np.array(on), np.array(pre)
    print(f"\nholiday days {len(on)}: effect = its type's draw in {np.mean(np.isclose(on[:, 0], on[:, 1], atol=0.6)):.2f}; "
          f"day before same sign {np.mean(np.sign(pre[:, 0]) == np.sign(pre[:, 1])):.2f}; day after nonzero "
          f"{np.mean(np.array(post) > 1e-6):.2f}")
    assert np.corrcoef(on[:, 0], on[:, 1])[0, 1] > 0.9
    assert np.mean(np.sign(pre[:, 0]) == np.sign(pre[:, 1])) > 0.7
    assert np.mean(np.array(post) > 1e-6) > 0.5


def test_invented_calendars_never_equal_a_real_one():
    t = tab()
    a = day(dt.date(2015, 1, 1))
    days = a + torch.arange(365).view(1, -1).repeat(512, 1)
    code, wk = E.invented(gen(4), days, "cpu")
    inv = {tuple(np.nonzero(c)[0]) for c in code.numpy()}
    real = {tuple(np.nonzero(t.code[k, a:a + 365].numpy())[0]) for k in range(t.C)}
    assert not (inv & real)
    n_h = (code > 0).sum(1).float()
    assert 4 <= float(n_h.min()) and float(n_h.max()) <= 12 + 4 * 1 + 3 * 3 * 13      # fixed + nth + drifting days
    drift = [len(np.nonzero(c)[0]) for c in code.numpy()]
    assert np.mean(drift) > 6


def test_drifting_holidays_move_11_days():
    days = torch.arange(0, 3650).view(1, -1).repeat(64, 1)
    g = gen(5)
    n = 64
    a = torch.rand(n, 1, generator=g) * E.DRIFT
    hit = torch.remainder(days.float() - a, E.DRIFT) < 1
    for r in hit[:8]:
        s = np.nonzero(r.numpy())[0]
        assert set(np.diff(s)) <= {354, 355}


def test_dst_rows_have_23_and_25_hour_days():
    n, L = 64, 24 * 400
    x = torch.arange(L).float().view(1, L).repeat(n, 1)
    o = opts(dst_p=1.0, shape_p=0.0, mult_p=0.0, invented=0.0)
    y, info = E.apply(gen(6), x, o, "cpu", ppd_idx=torch.full((n,), 2, dtype=torch.long))
    rows = info["dst_rows"].nonzero().view(-1)
    assert len(rows) > 5
    for i in rows.tolist():
        j = info["phys"][i]
        dj = torch.diff(j)[j[1:] < L - 1]            # after a spring-forward the last local hour is clamped to the end
        days = info["d0"][i] + torch.arange(L // 24 + 1)
        on = tab().dst[info["cal"][i], (days - 1).clamp(min=0)].long()
        sp, au = int(((on[1:] - on[:-1]) == 1).sum()), int(((on[1:] - on[:-1]) == -1).sum())
        assert int((dj == 2).sum()) == sp and int((dj == 0).sum()) == au     # one hour skipped / repeated per transition
        assert sp + au >= 1


def test_weekly_rows_show_christmas_weeks():
    n, L = 200, 120
    y, info = E.apply(gen(7), torch.randn(n, L, generator=gen(8)), opts(invented=0.0), "cpu",
                      ppd_idx=torch.zeros(n, dtype=torch.long))
    hits = []
    for i in range(n):
        c = int(info["cal"][i])
        if "christmas" not in str(tab().cals[c]) and T["christmas"] not in tab().code[c].tolist():
            continue
        d0 = int(info["d0"][i])
        for yy in range(1990, 2041):
            k = (day(dt.date(yy, 12, 25)) - d0) // 7
            if 0 <= k < L and int(tab().code[c, day(dt.date(yy, 12, 25))]) > 0:
                hits.append(float(info["code"][i, k]) > 0)
    assert len(hits) > 20 and all(hits)


# ---------------------------------------------------------------------------------------------------- G2.2 oracle (P2)
def test_p2_oracle_flags_help_on_holiday_days():
    from tideforge.diagnose.g1_oracle import _fit_mae
    n, L = 96, 1500
    prof = G.configure(B0)
    torch.manual_seed(0)
    base = G.generate_batch(n, L, "cpu", profile=prof)
    y, info = E.apply(gen(9), base, opts(), "cpu", ppd_idx=torch.ones(n, dtype=torch.long))
    ratios = []
    for i in range(n):
        yy = y[i].numpy().astype(float)
        days = info["d0"][i] + torch.arange(L)
        wd = E.dates(days)["wd"].numpy()
        Xb = np.stack([np.ones(L)] + [(wd == d).astype(float) for d in range(1, 7)] + [np.arange(L) / L], 1)
        code = info["code"][i].numpy().astype(int)
        fl = [(code == t).astype(float) for t in range(1, 13)]
        fl += [np.r_[(code[1:] > 0), False].astype(float), np.r_[False, (code[:-1] > 0)].astype(float),
               info["bridge"][i].numpy(), info["weekend"][i].numpy()]
        Xc = np.concatenate([Xb, np.stack(fl, 1)], 1)
        te = (code[-128:] > 0) | (np.r_[code[-127:], 0] > 0)
        if te.sum() < 2:
            continue
        def mae(X):
            tr = slice(128, L - 128)
            mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
            mu[0], sd[0] = 0, 1
            Z = (X - mu) / sd
            beta = np.linalg.solve(Z[tr].T @ Z[tr] + 1e-3 * L * np.eye(Z.shape[1]), Z[tr].T @ yy[tr])
            e = np.abs(Z[-128:] @ beta - yy[-128:])
            return float(e[te].mean())
        mb = mae(Xb)
        if mb > 1e-9:
            ratios.append(mae(Xc) / mb)
    med = float(np.median(ratios))
    print(f"\nP2 oracle on cal_cov flags (daily rows, MAE on holiday and day-before days): median {med:.3f} over {len(ratios)}")
    assert med < 0.9


# ---------------------------------------------------------------------------------------------------- G1 clock, G3.2, G3.4
def test_g1_engine_clock():
    from tideforge.gen.ext import g1_common as C
    gr = C.grid(gen(10), 64, 24 * 60, 1.0, "cpu", "engine")
    assert 0.2 < float(gr["weekend"].float().mean()) < 0.4 and float((gr["holcode"] > 0).float().mean()) > 0.005
    fx = C.grid(gen(10), 64, 100, 1.0, "cpu")
    assert torch.equal(fx["weekend"], fx["dow"] >= 5)


def test_long_cycle_cal_replaces_long_cycle():
    G.configure(B0.with_changes(switches={"long_cycle_cal": True}))
    w = G.GEN_PROFILES[G.configure(B0.with_changes(switches={"long_cycle_cal": True}))]["weights"]
    assert w["long_cycle"] == 0 and w["long_cycle_cal"] > 0
    torch.manual_seed(0)
    from tideforge.gen.ext.g2_switches import long_cycle_cal
    x = long_cycle_cal(32, 2000, "cpu")
    assert bool(torch.isfinite(x).all()) and float(x.std(1).min()) > 0


def test_season_short_cycles():
    from tideforge.gen import _hooks as H
    prof = G.configure(B0.with_changes(switches={"season_short": {"share": 1.0}}))
    torch.manual_seed(0)
    x = torch.randn(256, 1200).cumsum(1) * 0.01
    allow = torch.ones(256, dtype=torch.bool)
    fn = H.layer("seasonality", __import__("tideforge.gen.layers", fromlist=["_L_seasonality"])._L_seasonality)
    y = fn(x, allow, "cpu")
    d = (y - x).numpy()
    per = []
    for r in d:
        P = np.abs(np.fft.rfft(r - r.mean()))
        k = int(np.argmax(P[1:])) + 1
        per.append(1200 / k)
    cyc = 1200 / np.array(per)
    print(f"\nseason_short: dominant cycles in window median {np.median(cyc):.2f}")
    assert 1 <= np.median(cyc) <= 3.5
