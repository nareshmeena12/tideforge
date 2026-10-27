"""Figures for the landscape report. Run from the report folder: python3 make_figs.py
Reads: data/fev_rel_errors.csv (fev per-task relative errors, from tools/fev_cov_split.py), ../../catalog/dataset_catalog_v2.csv,
and the generator in third_party/tidemark_e21cbf0."""
import csv, math, sys, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figs"); os.makedirs(OUT, exist_ok=True)
for f in glob.glob("/usr/share/fonts/opentype/inter/Inter-*.otf"):
    fm.fontManager.addfont(f)

# ---------------- design tokens (reference palette, light) ----------------
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100",
                                                          "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID, GRAY = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#b9b8b2"
plt.rcParams.update({
    "font.family": "Inter", "font.size": 8.5, "axes.titlesize": 9.5, "axes.titleweight": "semibold",
    "axes.titlelocation": "left", "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.size": 0, "ytick.major.size": 0,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "pdf.fonttype": 42,
    "text.color": INK, "axes.titlecolor": INK,
})

def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf")); plt.close(fig)

def rbar(ax, y, w, color, h=0.62, **kw):
    """Horizontal bar with rounded data-end, square at the baseline."""
    from matplotlib.patches import FancyBboxPatch
    ax.barh(y, w, height=h, color=color, linewidth=0, **kw)

# ---------------- fev: skill per model, covariate split, concentration ----------------
rows = list(csv.reader(open(os.path.join(HERE, "data", "fev_rel_errors.csv"))))
models = rows[0][1:]
tasks = [r[0] for r in rows[1:]]
R = np.array([[float(v) if v not in ("", "nan") else np.nan for v in r[1:]] for r in rows[1:]])
L = np.log(np.clip(R, 1e-2, 100))
def skill(col, mask=None):
    v = L[:, col] if mask is None else L[mask, col]
    return 100 * (1 - math.exp(np.nanmean(v)))
def has_cov(ix):
    f = ix.split("|"); lists = [x for x in f if x.startswith("[")]
    known = lists[-3] if len(lists) >= 3 else "[]"; past = lists[-2] if len(lists) >= 2 else "[]"
    return known != "[]" or past != "[]"
cov = np.array([has_cov(t) for t in tasks])
name = lambda t: t.split("|")[1]
mi = {m: i for i, m in enumerate(models)}
sk = sorted(((skill(i), m) for m, i in mi.items() if m not in ("Seasonal Naive",)), reverse=True)

# F1 leaderboard (selected) + ours
keep = ["TimeRSI-1.5", "timeh2o-v1", "Timestral-209M", "Timestral-123M", "t0-gamma", "timesfm-3", "Timestral-28M",
        "ChakraTS", "Chronos-2", "Lingjiang2.0", "TimesFM-2.5", "t0-beta", "Timestral-5M", "TiRex-2",
        "Toto-2.0-2.5B", "TabPFN-TS-3", "Toto-2.0-22m", "TiRex", "FlowState", "Moirai-2.0", "Chronos-Bolt",
        "Sundial-Base", "AutoTheta"]
pts = [(m, skill(mi[m])) for m in keep if m in mi]
pts.append(("Fluxion-Tide 30M (ours)", 43.78))
pts.sort(key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(6.3, 4.6))
for i, (m, v) in enumerate(pts):
    c = BLUE if "ours" in m else (ORANGE if m == "TimeRSI-1.5" else GRAY)
    ax.barh(i, v, height=0.62, color=c, linewidth=0)
    ax.text(v + 0.4, i, f"{v:.1f}", va="center", fontsize=7.2, color=INK if c != GRAY else INK2,
            fontweight="semibold" if c != GRAY else "normal")
ax.set_yticks(range(len(pts))); ax.set_yticklabels([m for m, _ in pts], fontsize=7.4)
for t in ax.get_yticklabels():
    if "ours" in t.get_text(): t.set_fontweight("bold"); t.set_color(INK)
ax.set_xlim(0, 58); ax.grid(axis="y", visible=False)
ax.set_xlabel("fev-bench skill score vs Seasonal Naive (higher is better)")
iy = [k for k, (m_, _) in enumerate(pts) if "ours" in m_][0]
ax.text(48.2, iy, "8.0 points below the leader", va="center", fontsize=7.2, color=BLUE, fontweight="semibold")
ax.set_title("Where we stand on fev-bench (100 tasks)")
save(fig, "f1_fev_board")

# F2 covariate split dumbbell
sel = ["TimeRSI-1.5", "Timestral-28M", "Chronos-2", "TimesFM-2.5", "Timestral-5M", "TiRex", "Toto-2.0-22m"]
fig, ax = plt.subplots(figsize=(6.3, 2.9))
for i, m in enumerate(sel[::-1]):
    a, b = skill(mi[m], ~cov), skill(mi[m], cov)
    ax.plot([a, b], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
    ax.scatter([a], [i], s=46, color=BLUE, edgecolor="white", linewidth=1.5, zorder=3, label="Tasks without covariates" if i == 0 else None)
    ax.scatter([b], [i], s=46, color=ORANGE, edgecolor="white", linewidth=1.5, zorder=3, label="Tasks with covariates" if i == 0 else None)
    ax.text(max(a, b) + 0.6, i, f"{b - a:+.1f}", va="center", fontsize=7.2, color=INK2)
ax.set_yticks(range(len(sel))); ax.set_yticklabels(sel[::-1], fontsize=7.6)
ax.set_xlabel("fev-bench skill score on each half of the benchmark"); ax.grid(axis="y", visible=False)
ax.set_xlim(34, 56)
ax.legend(loc="lower right", fontsize=7.4, handletextpad=0.3)
ax.set_title("Small models lose most on tasks that come with covariates")
save(fig, "f2_cov_split")

# F3 concentration of the gap to the leader
top = mi["TimeRSI-1.5"]
fig, ax = plt.subplots(figsize=(6.3, 2.7))
for m, c in (("Toto-2.0-22m", ORANGE), ("Timestral-5M", BLUE)):
    g = np.clip(L[:, mi[m]] - L[:, top], 0, None)
    g = np.sort(g[~np.isnan(g)])[::-1]
    cs = np.cumsum(g) / g.sum()
    x = np.arange(1, len(cs) + 1)
    ax.plot(x, 100 * cs, color=c, lw=2, label=m, solid_capstyle="round")
ax.axvline(10, color=GRID, lw=1); ax.axvline(25, color=GRID, lw=1)
ax.text(10.8, 8, "10 tasks: 42%", fontsize=7, color=MUTED); ax.text(25.8, 8, "25 tasks: 70%", fontsize=7, color=MUTED)
ax.set_xlim(0, 100); ax.set_ylim(0, 104)
ax.set_xlabel("Tasks, sorted from largest gap to smallest"); ax.set_ylabel("% of total gap")
ax.legend(loc="lower right", fontsize=7.4)
ax.set_title("The gap to the leader is concentrated in a few dozen tasks")
save(fig, "f3_gap_conc")

# F4 worst tasks for Toto-22m vs leader, colored by covariates
m = mi["Toto-2.0-22m"]
g = L[:, m] - L[:, top]
order = np.argsort(-np.nan_to_num(g, nan=-9))[:14]
fig, ax = plt.subplots(figsize=(6.3, 3.4))
for j, k in enumerate(order[::-1]):
    c = ORANGE if cov[k] else GRAY
    ax.barh(j, (math.exp(g[k]) - 1) * 100, height=0.62, color=c, linewidth=0)
    ax.text((math.exp(g[k]) - 1) * 100 + 3, j, f"{(math.exp(g[k]) - 1) * 100:.0f}%", va="center", fontsize=7, color=INK2)
ax.set_yticks(range(len(order))); ax.set_yticklabels([name(tasks[k]) for k in order[::-1]], fontsize=7.4)
ax.grid(axis="y", visible=False); ax.set_xlabel("How much larger Toto-2.0-22m's error is than TimeRSI-1.5's")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=ORANGE, label="Task has covariates"), Patch(color=GRAY, label="No covariates")],
          loc="lower right", fontsize=7.4)
ax.set_title("The 14 worst tasks of a 22M model: mostly covariate tasks")
save(fig, "f4_worst_tasks")

# ---------------- TIME board ----------------
time_rows = [("Timestral-209M", .626), ("Timestral-123M", .628), ("TimeRSI", .630), ("t0_gamma", .637),
             ("Timestral-28M", .638), ("TimesFM-3", .640), ("Timestral-5M", .652), ("Chronos-2", .662),
             ("Hart-36M", .666), ("Toto-2.0-22m", .670), ("Fluxion-Tide 30M (ours)", .680), ("TiRex", .683),
             ("Moirai2", .703)]
fig, ax = plt.subplots(figsize=(6.3, 3.3))
for i, (mname, v) in enumerate(time_rows[::-1]):
    c = BLUE if "ours" in mname else GRAY
    ax.barh(i, v, height=0.62, color=c, linewidth=0)
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7.2, color=INK if c == BLUE else INK2)
ax.set_yticks(range(len(time_rows))); ax.set_yticklabels([r[0] for r in time_rows[::-1]], fontsize=7.4)
for t in ax.get_yticklabels():
    if "ours" in t.get_text(): t.set_fontweight("bold")
ax.set_xlim(0.6, 0.72); ax.grid(axis="y", visible=False)
ax.set_xlabel("TIME normalised MASE (lower is better; Seasonal Naive = 1.0)")
ax.set_title("TIME: 23rd of 40, behind the 2026 small models")
save(fig, "f5_time_board")

# ---------------- generator: family mix and measured properties ----------------
mix = [("Textbook stationary", 36.6, "ets, gp_kernel, ar_p, tsi, ar, OU ..."),
       ("Exact repetition and day types", 28.5, "repeat, long_cycle, repeat_week, daily_weekly"),
       ("Trend, level, regime, price", 13.3, "random_walk, piecewise_trend, stochastic_vol ..."),
       ("Sparse, counts, events", 12.2, "count, intermittent, spikes, epidemic, flat ..."),
       ("Low frequency", 8.3, "aggregated, short_macro_trend")]
fig, ax = plt.subplots(figsize=(6.3, 2.2))
cols = [BLUE, ORANGE, AQUA, VIOLET, MAGENTA]
for i, (lab, v, fam) in enumerate(mix[::-1]):
    ax.barh(i, v, height=0.6, color=cols[::-1][i], linewidth=0)
    ax.text(v + 0.6, i, f"{v:.1f}%   {fam}", va="center", fontsize=7, color=INK2)
ax.set_yticks(range(len(mix))); ax.set_yticklabels([m[0] for m in mix[::-1]], fontsize=7.6)
ax.set_xlim(0, 75); ax.grid(axis="y", visible=False); ax.set_xticks([0, 10, 20, 30, 40])
ax.set_xlabel("Share of generated series")
ax.set_title("What ft3_main's generator makes (measured, clean2 profile)")
save(fig, "f6_gen_mix")

Ls = [128, 512, 2048, 8192]
props = {"Clear seasonality": [48, 47, 45, 47], "At least one large jump": [22, 34, 44, 52],
         "Heavy-tailed steps": [13, 25, 34, 39], "Strong linear trend": [20, 21, 23, 22],
         "More than 20% zeros": [12, 12, 12, 12], "Integer-valued": [3, 3, 3, 3]}
fig, ax = plt.subplots(figsize=(6.3, 2.7))
pc = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET]
for (k, v), c in zip(props.items(), pc):
    ax.plot(range(4), v, color=c, lw=2, marker="o", ms=4.5, mec="white", mew=1.2, label=k)
ax.set_xticks(range(4)); ax.set_xticklabels([f"L = {l}" for l in Ls]); ax.set_xlim(-0.15, 3.15)
ax.set_ylabel("% of non-flat series"); ax.set_ylim(0, 58)
ax.legend(fontsize=7, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), handlelength=1.6, columnspacing=1.2)
ax.set_title("Measured properties of generated series by window length", pad=30)
save(fig, "f7_gen_props")

# generator gallery
sys.path.insert(0, os.path.join(HERE, "..", "..", "third_party", "tidemark_e21cbf0"))
import torch
import tidemark.real.ft3_core as G
cfg = dict(G.CONFIG); cfg["gen_additions"] = dict(cfg["gen_additions"], yearly=True)
prof = G.resolve_profile(cfg)
fams = ["ets", "gp_kernel", "ar_p", "tsi", "repeat", "repeat_week", "long_cycle", "daily_weekly",
        "aggregated", "epidemic", "intermittent", "stochastic_vol"]
fig, axs = plt.subplots(4, 3, figsize=(6.3, 5.2))
for k, (fam, ax) in enumerate(zip(fams, axs.flat)):
    torch.manual_seed(11 + k)
    x = G.generate_batch(4, 512, "cpu", weights={fam: 1.0}, profile=prof)[0].numpy()
    ax.plot(x, color=BLUE, lw=0.9)
    ax.set_title(fam, fontsize=8, fontweight="semibold", pad=3)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ("left", "bottom"): ax.spines[s].set_visible(False)
fig.suptitle("One random draw per family (512 steps each, all layers applied)", x=0.01, ha="left",
             fontsize=9.5, fontweight="semibold", y=1.0)
fig.tight_layout(h_pad=0.8, w_pad=0.6)
save(fig, "f8_gallery")

# ---------------- dev catalog ----------------
cat = list(csv.DictReader(open(os.path.join(HERE, "..", "..", "catalog", "dataset_catalog_v2.csv"), encoding="utf-8")))
from collections import Counter
dom = Counter(r["domain"] for r in cat)
merge = {"Utilities-water": "Utilities (water, waste)", "Utilities-waste": "Utilities (water, waste)",
         "Physics-reference": "Space-geo", "Government-stats": "Macro-finance", "Telecom-network": "Web-attention"}
dom2 = Counter()
for k, v in dom.items(): dom2[merge.get(k, k)] += v
items = sorted(dom2.items(), key=lambda x: x[1])
fig, ax = plt.subplots(figsize=(6.3, 4.4))
for i, (k, v) in enumerate(items):
    ax.barh(i, v, height=0.62, color=BLUE, linewidth=0)
    ax.text(v + 0.3, i, str(v), va="center", fontsize=7, color=INK2)
ax.set_yticks(range(len(items))); ax.set_yticklabels([k.replace("-", " / ") for k, _ in items], fontsize=7.2)
ax.grid(axis="y", visible=False); ax.set_xlabel("Datasets in the benchmark-free dev catalog")
ax.set_title(f"Dev catalog: {len(cat)} datasets across {len(items)} domains")
save(fig, "f9_catalog_domains")

S, P, lab = [], [], []
for r in cat:
    if r["counted"] == "yes" and r["series_avail"] and r["overlap"] not in ("pool", "n/a"):
        s, p = float(r["series_avail"]), float(r["points_avail"])
        if s > 0 and p > 0: S.append(s); P.append(p); lab.append(r["name"])
S, P = np.array(S), np.array(P)
fig, ax = plt.subplots(figsize=(6.3, 3.4))
ax.scatter(S, P / S, s=14, color=BLUE, alpha=0.55, edgecolor="white", linewidth=0.5)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Number of series (log)"); ax.set_ylabel("Average points per series (log)")
for nm, short in (("NREL WIND Toolkit", "WIND Toolkit"), ("NHS English Prescribing Data (practice x drug)", "NHS prescribing"),
                  ("Grupo Bimbo inventory demand", "Grupo Bimbo"), ("Kepler/TESS light curves (MAST)", "Kepler / TESS"),
                  ("CHB-MIT scalp EEG (seizures)", "CHB-MIT EEG"), ("Kyoto cherry blossom dates (812 AD-)", "Kyoto blossom"),
                  ("Backblaze Drive Stats (daily SMART)", "Backblaze")):
    if nm in lab:
        k = lab.index(nm); ax.annotate(short, (S[k], P[k] / S[k]), xytext=(4, 3), textcoords="offset points", fontsize=6.8, color=INK2)
ax.axhline(10000, color=MUTED, lw=0.8, ls=(0, (3, 2)))
ax.text(3e6, 13000, "dev slice cap: 10k points", fontsize=6.8, color=MUTED, ha="right", va="bottom")
ax.set_title("Size of each catalog dataset (estimates)")
save(fig, "f10_catalog_sizes")

# ---------------- diagnosis: repair attribution on a worked example ----------------
t = np.arange(96); P_ = 24
truth = 10 + 3 * np.sin(2 * np.pi * t / P_) + 1.2 * np.sin(4 * np.pi * t / P_ + 0.6)
rng = np.random.default_rng(3); truth = truth + rng.normal(0, 0.35, len(t))
fc = 8.9 + 0.55 * (3 * np.sin(2 * np.pi * (t - 2) / P_) + 1.2 * np.sin(4 * np.pi * (t - 2) / P_ + 0.6))
mae = lambda f: np.mean(np.abs(truth - f))
e0 = mae(fc)
f1 = fc - fc.mean() + truth.mean(); e1 = mae(f1)                    # fix level
best = min(range(-6, 7), key=lambda s: mae(np.roll(f1 - f1.mean(), -s) + f1.mean()))
f2 = np.roll(f1 - f1.mean(), -best) + f1.mean(); e2 = mae(f2)       # fix phase
amp = np.std(truth - np.convolve(truth, np.ones(5) / 5, "same")) and (np.std(truth) / np.std(f2))
f3 = (f2 - f2.mean()) * (np.std(truth) / np.std(f2)) + f2.mean(); e3 = mae(f3)   # fix amplitude
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.3, 2.4), gridspec_kw=dict(width_ratios=[1.6, 1]))
a1.plot(t, truth, color=INK2, lw=1.4, label="What happened")
a1.plot(t, fc, color=ORANGE, lw=2, label="Our forecast")
a1.plot(t, f3, color=BLUE, lw=1.2, alpha=0.9, label="After the three repairs")
a1.set_xlabel("Forecast step"); a1.legend(fontsize=6.6, loc="upper left", ncol=3, handlelength=1.2, columnspacing=0.8); a1.set_ylim(4, 19)
a1.set_title("A forecast that is low, late and too flat")
parts = [("Level", e0 - e1), ("Phase", e1 - e2), ("Amplitude", e2 - e3), ("Left over", e3)]
tot = e0
for i, (k, v) in enumerate(parts[::-1]):
    a2.barh(i, 100 * v / tot, height=0.6, color=[GRAY, BLUE, AQUA, ORANGE][i], linewidth=0)
    a2.text(100 * v / tot + 2, i, f"{100 * v / tot:.0f}%", va="center", fontsize=7, color=INK2)
a2.set_yticks(range(4)); a2.set_yticklabels([p[0] for p in parts[::-1]], fontsize=7.4); a2.set_xlim(0, 75)
a2.grid(axis="y", visible=False); a2.set_xlabel("Share of the error")
a2.set_title("Why the error happened")
fig.tight_layout(w_pad=1.2)
save(fig, "f11_repair")

# ---------------- priorities: impact vs effort ----------------
gaps = [("Covariate scenarios\n+ causal graph", 8.6, 7.5), ("Calendar engine", 7.2, 6.4), ("Few-cycle history", 6.3, 4.2),
        ("Breaks switch", 3.2, 0.9), ("tv_season / AM-FM", 3.6, 2.6), ("Masked gaps + NaN fix", 5.2, 1.5),
        ("Demand / life cycle", 4.4, 4.6), ("Hierarchies", 2.6, 4.4), ("ODE dynamics", 2.0, 5.6), ("Financial tails", 1.6, 3.0),
        ("Known covariates see\nthe whole window (model)", 7.9, 3.3)]
fig, ax = plt.subplots(figsize=(6.3, 4.0))
ax.axhline(5, color=GRID, lw=1); ax.axvline(5, color=GRID, lw=1)
for lab_, imp, eff in gaps:
    c = VIOLET if "(model)" in lab_ else BLUE
    ax.scatter(eff, imp, s=60, color=c, edgecolor="white", linewidth=1.5, zorder=3)
    ax.annotate(lab_.replace(" (model)", ""), (eff, imp), xytext=(6, -2), textcoords="offset points", fontsize=6.9, color=INK2, va="top")
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_xticks([]); ax.set_yticks([])
ax.set_xlabel("Effort  →"); ax.set_ylabel("Expected impact on fev / TIME  →")
for x, y, s in ((0.2, 9.7, "Do first"), (5.2, 9.7, "Main bets"), (0.2, 0.35, "Quick wins, small"), (5.2, 0.35, "Later")):
    ax.text(x, y, s, fontsize=7.4, color=MUTED, fontweight="semibold", va="top" if y > 5 else "bottom")
ax.scatter([], [], s=40, color=BLUE, label="Generator"); ax.scatter([], [], s=40, color=VIOLET, label="Model")
ax.legend(loc="lower right", fontsize=7.2)
ax.set_title("Where to spend effort (judgement, not measurement)")
ax.grid(False)
save(fig, "f12_priorities")
print("figures written to", OUT, sorted(os.listdir(OUT)))
print("repair shares", [(k, round(100 * v / tot)) for k, v in parts])
