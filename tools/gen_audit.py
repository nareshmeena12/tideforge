"""Measure what a generator config actually produces (CPU, ~minutes): family shares, per-length shape statistics, detected
periods, covariate-group shares. Works for B0 and for any switch set.

Usage (from the repo root):
    python tools/gen_audit.py --base b0                                   # B0 (needs configs/ft3_main_cfg.json)
    python tools/gen_audit.py --exp G4.1                                  # an experiment config
    python tools/gen_audit.py --base b0 --switch NAME --param k=v --set gen_additions.breaks=true
    python tools/gen_audit.py --base defaults --set gen_additions.yearly=true --set cov_r3=true
        # = the earlier measurement in docs/notes/GENERATOR_GAPS.md (CONFIG defaults + yearly, cov_r3 for the group part)
    python tools/gen_audit.py --exp G4.1 --row runs/audit/rows.csv        # also append one CSV row (the B0 comparison table)
Everything goes through tideforge.gen (bit-exact with ft3_core when no switch is on); torch seed 0, switch seed 0.
"""
import argparse, math, collections, json
import numpy as np, torch
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO), str(REPO / "third_party" / "tidemark_e21cbf0")]
import tideforge.gen as G
from tideforge.expcfg import gc_from_cli
from tideforge.gen.primitives import PERIOD_BANK_EXT

ap = argparse.ArgumentParser()
ap.add_argument("--exp")
ap.add_argument("--base", default="b0", choices=["b0", "defaults"])
ap.add_argument("--switch", action="append", default=[])
ap.add_argument("--param", action="append", default=[])
ap.add_argument("--set", dest="set_", action="append", default=[])
ap.add_argument("--n", type=int, default=4096)
ap.add_argument("--row", default=None, help="append one summary row to this CSV")
a = ap.parse_args()
base, gc, label = gc_from_cli(a.base, a.exp, a.switch, a.param, a.set_)
prof = G.configure(gc)
torch.manual_seed(0)

W = G.GEN_PROFILES[prof]["weights"]
tot = sum(W.values())
print("config", label, "| profile", prof, "| additions", {k: v for k, v in gc.gen_additions if v},
      "| switches", {k: dict(v) for k, v in gc.switches})
print("family shares (% of series):")
for k, v in sorted(W.items(), key=lambda x: -x[1]):
    print(f"   {k:18s} {100*v/tot:5.1f}")

def feats(x):
    x = x.astype(np.float64); L = len(x); t = np.arange(L)
    f = {}
    sd = x.std()
    f["flat"] = sd < 1e-9 * (abs(x).mean() + 1)
    if f["flat"]:
        return f
    d = np.diff(x)
    f["zero_frac"] = float(np.mean(x == 0))
    f["integer"] = bool(np.all(np.abs(x - np.round(x)) < 1e-6))
    # trend: R^2 of linear fit
    A = np.vstack([t, np.ones(L)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    r = x - A @ coef
    f["trend_r2"] = 1 - r.var() / x.var()
    # seasonality: ACF of linearly detrended series, best lag in [2, L/3]
    rr = r - r.mean(); den = (rr * rr).sum() + 1e-12
    F = np.fft.rfft(rr, n=2 * L); ac = np.fft.irfft(F * np.conj(F))[:L] / den
    hi = max(3, L // 3)
    lag = int(np.argmax(ac[2:hi])) + 2
    f["acf_peak"] = float(ac[lag]); f["period"] = lag
    # local maximum check (a real cycle, not a slow decay)
    f["seasonal"] = bool(ac[lag] > 0.5 and ac[lag] > ac[max(lag - 1, 1)] and ac[lag] >= ac[min(lag + 1, L - 1)])
    f["cycles"] = L / lag
    # spectral slope of the periodogram (log-log), persistence
    P = np.abs(np.fft.rfft(rr)) ** 2
    fr = np.fft.rfftfreq(L)[1:]; P = P[1:]
    ok = P > 0
    f["spec_slope"] = float(np.polyfit(np.log(fr[ok]), np.log(P[ok]), 1)[0]) if ok.sum() > 5 else np.nan
    # roughness, jumps, tails, heteroskedasticity
    f["rough"] = float(d.std() / sd)
    mad = np.median(np.abs(d - np.median(d))) + 1e-12
    f["n_jumps"] = int(np.sum(np.abs(d) > 8 * 1.4826 * mad)) if mad > 1e-9 else 0
    k = ((d - d.mean()) ** 4).mean() / (d.var() ** 2 + 1e-30)
    f["kurt_diff"] = float(k)
    h = L // 2
    f["var_ratio"] = float(np.log((x[h:].std() + 1e-12) / (x[:h].std() + 1e-12)))
    # SHORT seasons (G3.4): the dominant frequency of the detrended series (periodogram zero-padded 4x, steps of 0.25
    # cycles per window) lies in [1.5, 3) cycles and holds >= 25% of the power; the ACF detector above needs >= 3 cycles
    Pz = np.abs(np.fft.rfft(rr, n=4 * L)) ** 2
    cyc = np.arange(len(Pz)) / 4.0
    band = cyc >= 1.0
    kpk = int(np.argmax(np.where(band, Pz, 0)))
    f["short_season"] = bool(1.5 <= cyc[kpk] < 3.0 and Pz[kpk] >= 0.25 * Pz[band].sum() / 4.0)
    # exact repetition at the detected lag
    f["exact_repeat"] = bool(np.mean(np.abs(x[lag:] - x[:-lag])) < 1e-3 * sd)
    return f

rows = []
for L in (128, 512, 2048, 8192):
    n = a.n if L <= 2048 else a.n // 4
    x = G.generate_batch(n, L, "cpu", profile=prof).numpy()
    for s in x:
        f = feats(s); f["L"] = L; rows.append(f)

def share(sel, key, fn=lambda v: v):
    v = [fn(r[key]) for r in sel if key in r]
    return 100 * np.mean(v) if v else float("nan")

print("\nPER LENGTH (all numbers % of series unless noted)")
hdr = ["L", "flat", "seasonal", "per<=1/8L", "exact_rep", "trend_r2>.5", "jumps>=1", "zero>20%", "integer", "kurt>10", "slope<-2", "var_change>2x"]
print(" | ".join(hdr))
for L in (128, 512, 2048, 8192):
    s = [r for r in rows if r["L"] == L]
    ns = [r for r in s if not r["flat"]]
    seas = [r for r in ns if r["seasonal"]]
    out = [L, share(s, "flat"), share(ns, "seasonal"),
           share(seas, "cycles", lambda c: c >= 8), share(ns, "exact_repeat"),
           share(ns, "trend_r2", lambda v: v > 0.5), share(ns, "n_jumps", lambda v: v >= 1),
           share(ns, "zero_frac", lambda v: v > 0.2), share(ns, "integer"),
           share(ns, "kurt_diff", lambda v: v > 10), share(ns, "spec_slope", lambda v: v < -2),
           share(ns, "var_ratio", lambda v: abs(v) > math.log(2))]
    print(" | ".join(f"{v:.0f}" if isinstance(v, float) else str(v) for v in out))

print("\nDETECTED PERIODS among seasonal series (top 15, L=2048):")
c = collections.Counter(r["period"] for r in rows if r["L"] == 2048 and not r["flat"] and r["seasonal"])
tot_s = sum(c.values())
bank = set(PERIOD_BANK_EXT)
print("  ", [(p, round(100 * k / tot_s, 1)) for p, k in c.most_common(15)])
print(f"   share at a bank period (+-1): {100*sum(k for p,k in c.items() if any(abs(p-b)<=1 for b in bank))/tot_s:.0f}%")
print(f"   share with period > 1/3 of what fits (<4 cycles): {100*sum(k for p,k in c.items() if 2048/p<4)/tot_s:.0f}%")

ROW = dict(config=label, run_id=gc.run_id_fragment(base))
# long_cycle and its G3.2 replacement long_cycle_cal count together
ROW["share_long_cycle"] = round(100 * (W.get("long_cycle", 0.0) + W.get("long_cycle_cal", 0.0)) / tot, 2)
ns_all = [r for r in rows if not r["flat"]]
ROW.update(integer=round(share(ns_all, "integer"), 2), jumps=round(share(ns_all, "n_jumps", lambda v: v >= 1), 2),
           var_change=round(share(ns_all, "var_ratio", lambda v: abs(v) > math.log(2)), 2),
           seasonal=round(share(ns_all, "seasonal"), 2), short_season=round(share(ns_all, "short_season"), 2),
           flat=round(share(rows, "flat"), 2))


def write_row():
    if not a.row:
        return
    import csv
    out = Path(a.row)
    out.parent.mkdir(parents=True, exist_ok=True)
    have = out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ROW), lineterminator="\n")
        if not have:
            w.writeheader()
        w.writerow(ROW)
    print(f"row appended to {out}")


# groups / covariates (the config's own group settings)
V, gp = gc.group_lanes, gc.group_prob
if V < 2 or gp <= 0:
    print(f"\nGROUPS: none in this config (group_lanes {V}, group_prob {gp})")
    write_row()
    sys.exit(0)
x, role, lgid = G.generate_blocks(512, V, 512, "cpu", profile=prof, group_prob=gp)
nan_share = float(torch.isnan(x).float().mean())
lane_nan = torch.isnan(x).any(-1)
is_tgt = role == G.ROLE_TARGET
ROW.update(nan_points=round(100 * nan_share, 2),
           nan_target_lanes=round(100 * float(lane_nan[is_tgt].float().mean()), 2),
           nan_cov_lanes=round(100 * float(lane_nan[~is_tgt].float().mean()), 2) if bool((~is_tgt).any()) else 0.0)
role = role.numpy(); lgid = lgid.numpy()
n_lanes = role.size
in_group = np.zeros_like(role, bool)
for i in range(role.shape[0]):
    ids, cnt = np.unique(lgid[i], return_counts=True)
    big = ids[cnt > 1]
    in_group[i] = np.isin(lgid[i], big)
tgt_with_known = 0; tgt_with_past = 0; tgt_with_cov = 0
for i in range(role.shape[0]):
    for gid in np.unique(lgid[i][in_group[i]]):
        m = lgid[i] == gid
        r = role[i][m]
        if (r == G.ROLE_KNOWN).any(): tgt_with_known += (r == G.ROLE_TARGET).sum()
        if (r == G.ROLE_PAST).any(): tgt_with_past += (r == G.ROLE_TARGET).sum()
        if (r != G.ROLE_TARGET).any(): tgt_with_cov += (r == G.ROLE_TARGET).sum()
print(f"\nGROUPS (V={V}, group_prob {gp}, cov_r3 {gc.cov_r3}): lanes in a related group {100*in_group.mean():.0f}%,"
      f" targets with KNOWN covariates {100*tgt_with_known/n_lanes:.1f}% of lanes,"
      f" with PAST covariates {100*tgt_with_past/n_lanes:.1f}%; NaN share {100*nan_share:.2f}%")
ROW.update(target_known=round(100 * tgt_with_known / n_lanes, 2), target_past=round(100 * tgt_with_past / n_lanes, 2),
           target_any_cov=round(100 * tgt_with_cov / n_lanes, 2),
           in_group=round(100 * float(in_group.mean()), 2))
from tideforge.gen import groups as _groups  # noqa: E402
kp = np.array(_groups.GROUP_KIND_PROBS)
# added kinds (G1): share of blocks = group_prob x p / sum(p), one column "name:share;..."
ROW["added_kinds"] = ";".join(f"{nm}:{100 * gp * pk / kp.sum():.2f}" for nm, pk in zip(_groups.GROUP_KINDS[5:], kp[5:]))
xt = x.numpy()[(role == G.ROLE_TARGET) & in_group]
xt = xt[np.isfinite(xt).all(1)]
ROW.update(group_tgt_zero20=round(100 * float(np.mean((xt == 0).mean(1) > 0.2)), 2),
           group_tgt_integer=round(100 * float(np.mean(np.all(np.abs(xt - np.round(xt)) < 1e-6, 1))), 2))
print(f"   added kinds (% of blocks): {ROW['added_kinds'] or 'none'}; group targets with > 20% zeros "
      f"{ROW['group_tgt_zero20']}%, integer {ROW['group_tgt_integer']}%")
write_row()
