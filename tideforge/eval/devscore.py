"""The dev scorer (T1b): score a model on a frozen dev version, report, compare two runs, rank-correlate with fev / TIME.

    python -m tideforge.eval.devscore score  --dev $TF_DATA/dev --version dev-v1 --set fast|full --model ft3main --ckpt PATH
                                             [--save-forecasts]   (forecasts.npz next to windows.parquet, for D3; off by default)
                                             [--cov] [--role locked --locked --experiment E --why "..."] [--out runs/dev]
    python -m tideforge.eval.devscore report --run <run dir> [--cov-run <cov run dir>]
    python -m tideforge.eval.devscore compare --a <run> --b <run> [--noise runs/noise/e07.json] [--slice domain=Energy] [--metric MASE]
    python -m tideforge.eval.devscore ranks  --runs <run> [<run> ...] --ref ranks_ref.csv

Sets (all frozen by tideforge.devdata.build, sha256 in the manifest, checked before scoring):
- main: <role>_fast.parquet (~15k windows) or <role>_score.parquet (~75k, `--set full`); models see targets only.
- dev-covariate (`--cov`): covtasks_windows.parquet of that role; each model is scored twice, groups ON (past and known
  covariates passed like fev) and hidden (targets only). Every covariate task weighs the same.
Headline numbers (owner 2026-10-27): the main dev score, the dev-covariate score (ON) with its ON-vs-hidden gain, and a
combined score (`report --cov-run`): every main dataset and every covariate task one unit, equal weight per unit (the main
datasets share their part in proportion to their manifest weights, so the domain cap still holds inside it).
Metrics and aggregation: tideforge.eval.devmetrics (TIME's definitions). The baseline is SeasonalNaiveTIME (tideforge.eval
.naive_time: TIME's statsforecast band without its unseeded sampling step), always scored on the same windows.
The locked role needs --locked and --why; every locked run adds a LOOKS.md row. Runs carry the version and the set sha256;
compare refuses runs from different versions or sets.
`ranks` is a sanity check (EXPERIMENTS.md E0.5), never a gate: Spearman and Kendall between the dev ranking and fev / TIME,
with bootstrap CIs over models.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tideforge.devdata.freqs import seasonality
from tideforge.devdata.ingest import sha256_file
from tideforge.eval import devmetrics as M
from tideforge.eval import looks
from tideforge.eval import noise as NZ
from tideforge.eval.naive_time import SeasonalNaiveTIME
from tideforge.probes.common import Cell

BATCH = 256
NOISE_UNKNOWN = "noise floor unknown (E0.7 has not written its seed-noise file)"
SET_FILES = {"fast": "fast", "full": "score"}


# ---------------------------------------------------------------------------------------------------------- models

def make_model(spec: str, ckpt: str | None = None, device: str = "cuda"):
    """naive_time | naive | m5 | ft3main (or any ft3 name; needs --ckpt) | a public model (tideforge.eval.public_models)."""
    from tideforge.eval import public_models
    from tideforge.probes import models as PM
    if spec == "naive_time":
        return SeasonalNaiveTIME()
    if spec in public_models.REGISTRY:
        return public_models.make(spec, device=device)
    return PM.make(spec, ckpt, device)


def max_context(model) -> int | None:
    return getattr(model, "max_context", getattr(model, "max_ctx", None))


# ---------------------------------------------------------------------------------------------------------- windows

class Version:
    def __init__(self, dev: str | Path, version: str):
        self.dir = Path(dev) / version
        self.manifest = json.loads((self.dir / "manifest.json").read_text(encoding="utf-8"))
        self.version = version
        self._series: dict = {}

    def set_path(self, role: str, set_: str) -> Path:
        return self.dir / f"{role}_{SET_FILES[set_]}.parquet"

    def check_sha(self, role: str, set_: str) -> str:
        p = self.set_path(role, set_)
        want = self.manifest["sets"][f"{role}_{SET_FILES[set_]}"]["sha256"]
        got = sha256_file(p)
        if got != want:
            raise RuntimeError(f"{p} sha256 {got[:12]} differs from the manifest ({want[:12]}): the version is not intact")
        return got

    def series(self, path: Path, target: str, covs: tuple = ()) -> dict:
        """{series_id: (target values, {cov: values})} of one data file, sorted by time (cached)."""
        key = (path, target, covs)
        if key not in self._series:
            df = pd.read_parquet(path).sort_values(["series_id", "timestamp"])
            self._series[key] = {sid: (g[target].to_numpy(float), {c: g[c].to_numpy(float) for c in covs})
                                 for sid, g in df.groupby("series_id", sort=False)}
        return self._series[key]


def main_items(v: Version, win: pd.DataFrame):
    """Per window: dict(meta, ctx, y, S) for the main set."""
    doms = {d["dataset_id"]: d["domain"] for d in v.manifest["datasets"]}
    for (slug, freq), g in win.groupby(["slug", "freq"], sort=True):
        ser = v.series(v.dir / "data" / slug / f"{freq}.parquet", "value")
        S = seasonality(freq)
        for r in g.itertuples():
            y = ser[r.series_id][0]
            c = int(r.cutoff)
            yield dict(dataset_id=r.dataset_id, domain=doms.get(r.dataset_id, r.domain), freq=freq, term=r.term,
                       series_id=r.series_id, window=int(r.window), ctx=y[:c], y=y[c:c + int(r.H)], S=S, cov=False,
                       past={}, known={})


def cov_meta(v: Version, task: str) -> dict:
    return json.loads((v.dir / "covtasks" / task / "meta.json").read_text(encoding="utf-8"))


def cov_items(v: Version, win: pd.DataFrame):
    """Per covariate-task window: dict(meta, ctx, y, S, past {name: (L,)}, known {name: ((L,), (H,))})."""
    slug_dom = {d["slug"]: d["domain"] for d in v.manifest["datasets"]}
    for (task, freq), g in win.groupby(["task", "freq"], sort=True):
        meta = cov_meta(v, task)
        roles = {c: x.get("role") for c, x in meta["covariates"].items()}
        ser = v.series(v.dir / "covtasks" / task / f"{freq}.parquet", "target", tuple(sorted(roles)))
        S = seasonality(freq)
        for r in g.itertuples():
            y, cv = ser[r.series_id]
            c, H = int(r.cutoff), int(r.H)
            yield dict(dataset_id=task, domain=slug_dom.get(meta.get("target_slug"), "unknown"), freq=freq, term=r.term,
                       series_id=r.series_id, window=int(r.window), ctx=y[:c], y=y[c:c + H], S=S, cov=True,
                       past={k: cv[k][:c] for k, ro in roles.items() if ro == "past"},
                       known={k: (cv[k][:c], cv[k][c:c + H]) for k, ro in roles.items() if ro == "known"})


def _cells(items: list, cap: int | None, mode: str):
    """Batches of items with the same (truncated) context length, horizon and season, as probe-interface Cells."""
    def L(it):
        return len(it["ctx"]) if cap is None else min(cap, len(it["ctx"]))
    groups: dict = {}
    for it in items:
        groups.setdefault((L(it), len(it["y"]), it["S"], tuple(sorted(it["past"])), tuple(sorted(it["known"]))), []).append(it)
    for (Lc, H, S, pk, kk), its in groups.items():
        for i in range(0, len(its), BATCH):
            b = its[i:i + BATCH]
            a = {"targets": np.stack([it["ctx"][len(it["ctx"]) - Lc:] for it in b]).astype(np.float32)[:, None, :],
                 "future": np.zeros((len(b), 1, H)), "ctx_len": np.full(len(b), Lc)}
            if mode == "on" and pk:
                a["past"] = np.stack([[it["past"][k][len(it["ctx"]) - Lc:] for k in pk] for it in b]).astype(np.float32)
            if mode == "on" and kk:
                a["known_ctx"] = np.stack([[it["known"][k][0][len(it["ctx"]) - Lc:] for k in kk] for it in b]).astype(np.float32)
                a["known_fut"] = np.stack([[it["known"][k][1] for k in kk] for it in b]).astype(np.float32)
            yield b, Cell("dev", a, S, modes=(mode,))


def predict(model, items: list, mode: str) -> list[np.ndarray]:
    """(9, H) quantiles per item, in the items' order."""
    out = {}
    for b, cell in _cells(items, max_context(model), mode):
        q = np.asarray(model.predict_cell(cell, mode if mode != "main" else "main"))
        for it, qi in zip(b, q):
            out[id(it)] = qi[0]
    return [out[id(it)] for it in items]


def score_items(model, items: list, mode: str, naive_q: list | None = None, keep: list | None = None) -> pd.DataFrame:
    """keep: if a list, (item, mode, q) is appended per window (for --save-forecasts)."""
    q = predict(model, items, mode)
    if keep is not None:
        keep.extend((it, mode, qi) for it, qi in zip(items, q))
    nq = q if (naive_q is None and isinstance(model, SeasonalNaiveTIME)) else (naive_q or predict(SeasonalNaiveTIME(), items, "main"))
    rows = []
    for it, qi, ni in zip(items, q, nq):
        m = M.window_metrics(qi, it["y"], it["ctx"], it["S"])
        n = M.window_metrics(ni, it["y"], it["ctx"], it["S"])
        rows.append(dict(dataset_id=it["dataset_id"], domain=it["domain"], freq=it["freq"], term=it["term"],
                         series_id=it["series_id"], window=it["window"], ctx_len=len(it["ctx"]),
                         hist=M.hist_band(len(it["ctx"])), cov="yes" if it["cov"] else "no", mode=mode,
                         MASE=m[0], CRPS=m[1], MASE_naive=n[0], CRPS_naive=n[1]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------------------- runs

def save_forecasts(path: Path, keep: list) -> None:
    """forecasts.npz: per window (same key as windows.parquet: dataset_id, series_id, window, mode) the 9 quantiles and the
    truth, concatenated over windows: q (sum H, 9), y (sum H,), start / H per window, and the season S (for D3)."""
    H = np.array([len(it["y"]) for it, _, _ in keep], np.int64)
    np.savez_compressed(
        path, dataset_id=np.array([it["dataset_id"] for it, _, _ in keep]), series_id=np.array([str(it["series_id"]) for it, _, _ in keep]),
        window=np.array([it["window"] for it, _, _ in keep], np.int64), mode=np.array([m for _, m, _ in keep]),
        S=np.array([it["S"] for it, _, _ in keep], np.int64), H=H, start=np.concatenate([[0], np.cumsum(H)[:-1]]).astype(np.int64),
        q=np.concatenate([np.asarray(q, np.float32).T for _, _, q in keep]), y=np.concatenate([np.asarray(it["y"], np.float32) for it, _, _ in keep]))


def load_forecasts(path: str | Path) -> dict:
    """forecasts.npz -> dict of key arrays plus "q" / "y" lists of (9, H) / (H,) per window."""
    z = np.load(path)
    out = {k: z[k] for k in ("dataset_id", "series_id", "window", "mode", "S", "H")}
    out["q"] = [z["q"][s:s + h].T for s, h in zip(z["start"], z["H"])]
    out["y"] = [z["y"][s:s + h] for s, h in zip(z["start"], z["H"])]
    return out


def run_dir(out: str | Path, version: str, role: str, set_: str, cov: bool, model: str) -> Path:
    return Path(out) / version / f"{role}_{'cov' if cov else set_}" / model


def weights_of(run: dict, win: pd.DataFrame) -> dict[str, float]:
    if run["cov"]:
        ids = sorted(win["dataset_id"].unique())
        return {d: 1.0 / len(ids) for d in ids}
    return run["weights"]


def summarise(win: pd.DataFrame, weights: dict) -> dict:
    s = M.score(win, weights)
    return dict(overall=s["overall"], datasets=s["datasets"].to_dict("records"), slices=s["slices"].to_dict("records"))


def cmd_score(a) -> int:
    if a.role == "locked" and not a.locked:
        raise SystemExit("the locked set is scored only with --locked (and --why): it is looked at once per decision")
    if a.locked and not (a.why and a.experiment):
        raise SystemExit("--locked needs --experiment and --why (every locked use is a LOOKS.md row)")
    v = Version(a.dev, a.version)
    if a.cov:
        p = v.dir / "covtasks_windows.parquet"
        win = pd.read_parquet(p)
        win = win[win["role"] == a.role]
        sha, modes = sha256_file(p), ("on", "hidden")
        items = list(cov_items(v, win))
    else:
        sha, modes = v.check_sha(a.role, a.set), ("main",)
        win = pd.read_parquet(v.set_path(a.role, a.set))
        items = list(main_items(v, win))
    if not items:
        raise SystemExit("no windows in that set")
    model = make_model(a.model, a.ckpt, a.device)
    covariates = getattr(model, "covariates", "past+known")
    if a.cov and covariates == "none":                   # public models without covariate input: hidden mode only, recorded
        modes = ("hidden",)
        print(f"{a.model}: no covariate support, scored in hidden mode only")
    naive_q = predict(SeasonalNaiveTIME(), items, "main")
    keep = [] if a.save_forecasts else None
    tab = pd.concat([score_items(model, items, m, naive_q, keep) for m in modes], ignore_index=True)
    d = run_dir(a.out, a.version, a.role, a.set, a.cov, a.model)
    d.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(d / "windows.parquet", index=False)
    if keep is not None:
        save_forecasts(d / "forecasts.npz", keep)
    run = dict(version=a.version, role=a.role, set="cov" if a.cov else a.set, cov=bool(a.cov), model=a.model, set_sha256=sha,
               ckpt_sha256=getattr(model, "ckpt_sha", None),
               weights={} if a.cov else v.manifest["weights"][a.role]["datasets"],
               covariates=covariates if covariates != "none" else "none (hidden only)")
    w = weights_of(run, tab)
    run["scores"] = {m: summarise(tab[tab["mode"] == m], w) for m in modes}
    if a.cov and "on" in modes:
        g = M.paired(M.dataset_scores(M.pair_ratios(tab[tab["mode"] == "on"])),
                     M.dataset_scores(M.pair_ratios(tab[tab["mode"] == "hidden"])), w)
        run["on_vs_hidden"] = {k: g[k] for k in ("ratio", "lo", "hi", "datasets")}
    (d / "run.json").write_text(json.dumps(run, indent=1, default=float), encoding="utf-8")
    print(headline(run))
    print(f"written {d}")
    if a.locked:
        rows = [looks.format_row(experiment=a.experiment, checkpoint=f"{a.model} ({m})", benchmark=f"dev locked {a.version} {run['set']}",
                                 why=a.why, result=_fmt(run["scores"][m]["overall"])) for m in modes]
        added = looks.append_looks(rows, a.looks_file)
        print(f"{len(added)} LOOKS row(s) added to {a.looks_file}:", *added, sep="\n")
    return 0


def _fmt(o: dict) -> str:
    return ", ".join(f"{m} {o[m][0]:.4f} [{o[m][1]:.4f}, {o[m][2]:.4f}]" for m in M.METRICS) + f" ({o['datasets']} datasets)"


def headline(run: dict) -> str:
    lines = [f"{run['model']} on {run['version']} {run['role']} {run['set']} (vs SeasonalNaiveTIME, < 1 = better)"]
    for m, s in run["scores"].items():
        lines.append(f"  {m:6s} {_fmt(s['overall'])}")
    if "on_vs_hidden" in run:
        g = run["on_vs_hidden"]
        lines.append(f"  ON / hidden (MASE) {g['ratio']:.4f} [{g['lo']:.4f}, {g['hi']:.4f}] over {g['datasets']} tasks")
    return "\n".join(lines)


def load_run(d: str | Path) -> tuple[dict, pd.DataFrame]:
    d = Path(d)
    return json.loads((d / "run.json").read_text(encoding="utf-8")), pd.read_parquet(d / "windows.parquet")


def combined(main_run: dict, main_win: pd.DataFrame, cov_run: dict, cov_win: pd.DataFrame) -> dict:
    """Every main dataset and every covariate task one unit; main datasets share their part by manifest weight."""
    md = M.dataset_scores(M.pair_ratios(main_win))
    cd = M.dataset_scores(M.pair_ratios(cov_win[cov_win["mode"] == "on"]))
    n_m, n_c = len(md), len(cd)
    mw = {d: main_run["weights"][d] for d in md["dataset_id"]}
    tot = sum(mw.values())
    w = {d: x / tot * n_m / (n_m + n_c) for d, x in mw.items()} | {d: 1 / (n_m + n_c) for d in cd["dataset_id"]}
    return M.overall(pd.concat([md, cd], ignore_index=True), w)


def cmd_report(a) -> int:
    run, win = load_run(a.run)
    print(headline(run))
    for m, s in run["scores"].items():
        print(f"\nslices ({m}):")
        print(pd.DataFrame(s["slices"]).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    if a.cov_run:
        cr, cw = load_run(a.cov_run)
        if cr["version"] != run["version"] or not cr["cov"] or run["cov"]:
            raise SystemExit("--cov-run must be the dev-covariate run of the same version")
        print("\ncombined (equal weight per main dataset / covariate task):", _fmt(combined(run, win, cr, cw)))
    return 0


def rule5(a_run, a_win, b_run, b_win, metric: str, noise: dict | None, own: str | None) -> list[str]:
    """EXPERIMENTS.md rule 5: beats B by more than the seed noise, its own slice improves, no domain slice worse by > 1%."""
    w = weights_of(a_run, a_win)
    mode = "on" if a_run["cov"] else "main"
    aw, bw = a_win[a_win["mode"] == mode], b_win[b_win["mode"] == mode]
    out = []
    p = M.paired(M.dataset_scores(M.pair_ratios(aw)), M.dataset_scores(M.pair_ratios(bw)), w, metric)
    gain = 1 - p["ratio"]
    fl = NZ.floor(noise, metric)                     # the 3-seed RANGE (pre-registered in E0.7 Part A); the sd is not used
    if fl is None:
        out.append(f"UNKNOWN  (1) beats B by more than the seed noise: gain {gain:+.4f}; {NOISE_UNKNOWN}")
    else:
        out.append(f"{'PASS' if gain > fl else 'FAIL':8s} (1) gain {gain:+.4f} vs seed noise {fl:.4f} (range over seeds)")
    if own:
        k, _, val = own.partition("=")
        sa, sb = aw[aw[k].astype(str) == val], bw[bw[k].astype(str) == val]
        if not len(sa):
            out.append(f"FAIL     (2) own slice {own}: no windows")
        else:
            q = M.paired(M.dataset_scores(M.pair_ratios(sa)), M.dataset_scores(M.pair_ratios(sb)), w, metric)
            out.append(f"{'PASS' if q['ratio'] < 1 else 'FAIL':8s} (2) own slice {own}: A / B {q['ratio']:.4f} [{q['lo']:.4f}, {q['hi']:.4f}]")
    else:
        out.append("UNKNOWN  (2) own slice: not given (--slice key=value)")
    worse = []
    for dom in sorted(set(aw["domain"]) & set(bw["domain"])):
        q = M.paired(M.dataset_scores(M.pair_ratios(aw[aw["domain"] == dom])),
                     M.dataset_scores(M.pair_ratios(bw[bw["domain"] == dom])), w, metric, n_boot=1)
        if q["ratio"] > 1.01:
            worse.append(f"{dom} {q['ratio']:.4f}")
    out.append(f"{'FAIL' if worse else 'PASS':8s} (3) no domain slice worse by > 1%" + (f": {', '.join(worse)}" if worse else ""))
    return out


def cmd_compare(a) -> int:
    ar, aw = load_run(a.a)
    br, bw = load_run(a.b)
    if (ar["version"], ar["role"], ar["set"], ar["set_sha256"]) != (br["version"], br["role"], br["set"], br["set_sha256"]):
        raise SystemExit("A and B were scored on different dev versions or sets: they cannot be compared")
    mode = "on" if ar["cov"] else "main"
    w = weights_of(ar, aw)
    for metric in M.METRICS:
        p = M.paired(M.dataset_scores(M.pair_ratios(aw[aw["mode"] == mode])),
                     M.dataset_scores(M.pair_ratios(bw[bw["mode"] == mode])), w, metric)
        print(f"{ar['model']} / {br['model']} {metric}: {p['ratio']:.4f} [{p['lo']:.4f}, {p['hi']:.4f}] over {p['datasets']} datasets"
              " (< 1 = A better)")
    noise = json.loads(Path(a.noise).read_text(encoding="utf-8")) if a.noise and Path(a.noise).exists() else None
    print(f"\nrule 5 ({a.metric}):", *rule5(ar, aw, br, bw, a.metric, noise, a.slice), sep="\n  ")
    return 0


# ---------------------------------------------------------------------------------------------------------- ranks

def rank_corr(dev: dict[str, float], ref: dict[str, float], n_boot: int = M.N_BOOT, seed: int = M.BOOT_SEED) -> dict:
    """Spearman and Kendall between two {model: score} maps (both lower = better), with bootstrap CIs over models."""
    from scipy import stats
    ms = sorted(set(dev) & set(ref))
    x, y = np.array([dev[m] for m in ms]), np.array([ref[m] for m in ms])
    out = dict(models=len(ms))
    rng = np.random.default_rng(seed)
    for name, f in (("spearman", stats.spearmanr), ("kendall", stats.kendalltau)):
        r = float(f(x, y)[0]) if len(ms) > 2 else float("nan")
        bs = []
        for _ in range(n_boot):
            i = rng.integers(0, len(ms), len(ms))
            if len(set(i)) > 2:
                bs.append(float(f(x[i], y[i])[0]))
        bs = np.array([b for b in bs if np.isfinite(b)])
        out[name] = (r, *(np.percentile(bs, [2.5, 97.5]) if len(bs) else (float("nan"),) * 2))
    return out


def cmd_ranks(a) -> int:
    """--ref CSV: model, benchmark (fev / TIME), score, higher_is_better (fev skill: 1; TIME MASE: 0)."""
    dev = {}
    for d in a.runs:
        r, _ = load_run(d)
        dev[r["model"]] = r["scores"]["on" if r["cov"] else "main"]["overall"][a.metric][0]
    ref = pd.read_csv(a.ref)
    print("rank correlation, dev vs public boards: a SANITY CHECK, not a gate (EXPERIMENTS.md E0.5)")
    for b, g in ref.groupby("benchmark"):
        sc = {r.model: (-r.score if r.higher_is_better else r.score) for r in g.itertuples()}
        c = rank_corr(dev, sc)
        print(f"  dev {a.metric} vs {b}: {c['models']} models; Spearman {c['spearman'][0]:.3f} [{c['spearman'][1]:.3f}, "
              f"{c['spearman'][2]:.3f}], Kendall {c['kendall'][0]:.3f} [{c['kendall'][1]:.3f}, {c['kendall'][2]:.3f}]")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("score", "report", "compare", "ranks"))
    ap.add_argument("--dev")
    ap.add_argument("--version")
    ap.add_argument("--set", choices=tuple(SET_FILES), default="fast")
    ap.add_argument("--cov", action="store_true", help="score the dev-covariate set (groups ON and hidden)")
    ap.add_argument("--save-forecasts", action="store_true", help="also write forecasts.npz (quantiles + truth per window, for D3)")
    ap.add_argument("--role", choices=("tuning", "locked"), default="tuning")
    ap.add_argument("--locked", action="store_true", help="required to score the locked role")
    ap.add_argument("--experiment")
    ap.add_argument("--why")
    ap.add_argument("--looks-file", default=str(looks.LOOKS))
    ap.add_argument("--model")
    ap.add_argument("--ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="runs/dev")
    ap.add_argument("--run")
    ap.add_argument("--cov-run")
    ap.add_argument("--a")
    ap.add_argument("--b")
    ap.add_argument("--noise", default="runs/noise/e07.json")
    ap.add_argument("--slice")
    ap.add_argument("--metric", choices=M.METRICS, default="MASE")
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--ref")
    a = ap.parse_args(argv)
    if a.cmd == "score" and not (a.dev and a.version and a.model):
        raise SystemExit("score needs --dev, --version and --model")
    return {"score": cmd_score, "report": cmd_report, "compare": cmd_compare, "ranks": cmd_ranks}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
