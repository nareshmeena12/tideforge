"""X3 mixture search, RegMix-style (owner OK 2026-10-29, replaces the plan's DoReMi-style search; EXPERIMENTS.md section 6).

    python -m tideforge.campaign.mix plan --from X1.5 --anchors E0.7 X1.5 G4.3 G1.8 [--n 32] [--points 1e10] [--seed 0]
    python -m tideforge.campaign.mix fit  --board-runs runs/board/runs.csv [--plan configs/campaign/x3_mix.json]
    python -m tideforge.campaign.mix rank --board-runs runs/board/runs.csv --milestones runs/board/milestones.csv

plan: n short runs (default 32 = 28 random mixes + 4 anchors) of --points each (10B; the LR schedule follows target_points,
    so each is a real 10B run). A mix = one multiplier per role group present in the --from config (configs/campaign/roles
    .yaml), drawn log-uniform in [0.25, 4] from --seed, applied to that group's amount parameters (roles.yaml amount: share,
    p, target_lanes; capped at 0.95); groups with no amount parameter stay as they are. Anchors: configs that also have
    100B runs (B0, the X1 config, 2 Phase 2 winners; owner 2026-10-29), the same config at 10B, so 10B-run and 100B-run ranks
    compare on equal terms. Writes configs/exp/X3.rNN.yaml, X3.aN.yaml, configs/campaign/x3.yaml (the campaign) and
    configs/campaign/x3_mix.json (every run's multipliers).
fit: ridge regression of log(dev MASE) on the log multipliers with squares and pairwise products, alpha by leave-one-out CV;
    the predicted best mix in the box (random candidates + bounded L-BFGS from the best ones); the result is ONE config to
    confirm with a full 100B run (then 3 seeds), never a result in itself.
rank: the validity check. Spearman between 10B and 100B dev ranks on (1) the 4 anchors (10B run vs 100B run, equal terms)
    and (2) milestone checkpoints (ckpt_10B.pt inside 100B runs: still at a high LR, so a CHEAP EXTRA SIGNAL only, labelled
    so); judged together, rho < 0.7 -> rerun the short runs at 20B. Both are reported.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from tideforge.campaign import stack as ST
from tideforge.gen import switches as SW

LO, HI, CAP = 0.25, 4.0, 0.95
RHO_MIN = 0.7
CAMPAIGNS = ST.REPO / "configs" / "campaign"


# ------------------------------------------------------------------------------------------------- sampling and configs

def sample(groups: list[str], n: int, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(math.log(LO), math.log(HI), size=(n, len(groups)))
    return [{g: float(round(math.exp(v), 6)) for g, v in zip(groups, row)} for row in x]


def amount_params(r: dict) -> dict:
    """switch -> [amount parameter names]."""
    out: dict = {}
    for param, names in r.get("amount", {}).items():
        for n in names:
            out.setdefault(n, []).append(param)
    return out


def scalable_groups(d: dict, r: dict) -> list[str]:
    amt = amount_params(r)
    return [g for g in ST.groups_present(d, r) if any(s in amt for s in r["groups"][g].get("switches", []) if s in
                                                      (d.get("switches") or {}))]


def apply(d: dict, mults: dict, r: dict) -> dict:
    amt = amount_params(r)
    group_of = {s: g for g, v in r["groups"].items() for s in v.get("switches", [])}
    sw = {}
    for name, params in (d.get("switches") or {}).items():
        g = group_of.get(name)
        if g in mults and name in amt:
            p = {} if params is True else dict(params)
            for k in amt[name]:
                v = float(p.get(k, SW.REGISTRY[name].params[k]))
                p[k] = float(round(min(CAP, v * mults[g]), 6))
            sw[name] = p
        else:
            sw[name] = params
    return dict(d, switches=sw)


def plan(from_id: str, anchors: list[str], n: int = 32, points: float = 1e10, seed: int = 0, out_dir: Path | None = None,
         campaigns: Path = CAMPAIGNS, force: bool = False) -> dict:
    out_dir = out_dir or ST.expcfg.EXP_DIR
    r = ST.roles()
    base = {k: v for k, v in ST.raw(from_id).items() if k in ("base", "gen", "switches", "train", "model")}
    groups = scalable_groups(base, r)
    n_rand = n - len(anchors)
    if n_rand < 2 * len(groups) + 2:
        raise ValueError(f"{n_rand} random runs for {len(groups)} groups: too few to fit (need >= {2 * len(groups) + 2})")
    mixes = sample(groups, n_rand, seed)
    runs, record = [], {"from": from_id, "groups": groups, "points": points, "seed": seed, "lo": LO, "hi": HI, "runs": {}}
    for k, m in enumerate(mixes, 1):
        cid = f"X3.r{k:02d}"
        d = apply(base, m, r)
        d["train"] = dict(d.get("train") or {}, target_points=float(points))
        ST.write(cid, d, f"X3 mix {k}/{n_rand} of {from_id}: " + ", ".join(f"{g} x{v:.3f}" for g, v in m.items()), out_dir, force)
        runs.append(cid)
        record["runs"][cid] = m
    for k, a in enumerate(anchors, 1):
        cid = f"X3.a{k}"
        d = {kk: v for kk, v in ST.raw(a).items() if kk in ("base", "gen", "switches", "train", "model")}
        d["train"] = dict(d.get("train") or {}, target_points=float(points))
        ST.write(cid, d, f"X3 anchor {k}: {a} at {points / 1e9:.0f}B (10B run vs its 100B run)", out_dir, force)
        runs.append(cid)
        record["runs"][cid] = {"anchor": a}
    c = campaigns / "x3.yaml"
    if c.exists() and not force:
        raise FileExistsError(f"{c} exists (--force)")
    c.write_text(yaml.safe_dump(dict(name="x3", seeds=[0], runs=runs), sort_keys=False), encoding="utf-8", newline="\n")
    (campaigns / "x3_mix.json").write_text(json.dumps(record, indent=1), encoding="utf-8", newline="\n")
    return record


# ------------------------------------------------------------------------------------------------- regression

def features(X: np.ndarray) -> np.ndarray:
    n, d = X.shape
    cols = [np.ones(n)] + [X[:, i] for i in range(d)] + [X[:, i] ** 2 for i in range(d)]
    cols += [X[:, i] * X[:, j] for i, j in itertools.combinations(range(d), 2)]
    return np.stack(cols, 1)


def _ridge(F, y, alpha):
    pen = np.eye(F.shape[1]) * alpha
    pen[0, 0] = 0.0                                          # the intercept is not shrunk
    return np.linalg.solve(F.T @ F + pen, F.T @ y)


def fit(X: np.ndarray, y: np.ndarray, alphas=(1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0)) -> dict:
    """Ridge on quadratic features of X (log multipliers) -> y (log score); alpha by leave-one-out CV."""
    F = features(X)
    best = None
    for a in alphas:
        err = []
        for i in range(len(y)):
            m = np.arange(len(y)) != i
            w = _ridge(F[m], y[m], a)
            err.append(float(F[i] @ w - y[i]))
        mse = float(np.mean(np.square(err)))
        if best is None or mse < best[1]:
            best = (a, mse)
    w = _ridge(F, y, best[0])
    r2 = 1 - best[1] / float(np.var(y)) if np.var(y) > 0 else float("nan")
    return dict(w=w, alpha=best[0], loo_mse=best[1], loo_r2=r2, d=X.shape[1])


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    return features(np.atleast_2d(X)) @ model["w"]


def best_mix(model: dict, n_cand: int = 20000, seed: int = 0) -> tuple[np.ndarray, float]:
    """The box point (log multipliers in [log LO, log HI]^d) with the lowest predicted score."""
    from scipy.optimize import minimize
    d, lo, hi = model["d"], math.log(LO), math.log(HI)
    rng = np.random.default_rng(seed)
    cand = rng.uniform(lo, hi, size=(n_cand, d))
    p = predict(model, cand)
    best_x, best_v = None, float("inf")
    for x0 in cand[np.argsort(p)[:8]]:
        res = minimize(lambda x: float(predict(model, x)[0]), x0, method="L-BFGS-B", bounds=[(lo, hi)] * d)
        if res.fun < best_v:
            best_x, best_v = res.x, float(res.fun)
    return best_x, best_v


def cmd_fit(board_runs: str, plan_path: Path) -> dict:
    rec = json.loads(plan_path.read_text(encoding="utf-8"))
    runs = pd.read_csv(board_runs)
    dev = runs.groupby("exp")["dev_MASE"].apply(lambda v: float(np.exp(np.mean(np.log(pd.to_numeric(v)))))).to_dict()
    groups = rec["groups"]
    ids = [k for k, v in rec["runs"].items() if "anchor" not in v and k in dev]
    missing = [k for k, v in rec["runs"].items() if "anchor" not in v and k not in dev]
    X = np.log([[rec["runs"][k][g] for g in groups] for k in ids])
    y = np.log([dev[k] for k in ids])
    m = fit(X, y)
    x, v = best_mix(m)
    mult = {g: float(math.exp(xi)) for g, xi in zip(groups, x)}
    return dict(runs=len(ids), missing=missing, alpha=m["alpha"], loo_r2=m["loo_r2"], best=mult,
                predicted_dev_MASE=float(math.exp(v)), best_observed=min(((dev[k], k) for k in ids), default=None))


# ------------------------------------------------------------------------------------------------- the rank check

def spearman(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic) if len(a) >= 3 else float("nan")


def rank_check(anchor_pairs: list[tuple[float, float]], milestone_pairs: list[tuple[float, float]]) -> dict:
    """pairs: (10B score, 100B score). rho on each set and on both together; rho < RHO_MIN -> 20B."""
    ra = spearman(*zip(*anchor_pairs)) if anchor_pairs else float("nan")
    rm = spearman(*zip(*milestone_pairs)) if milestone_pairs else float("nan")
    both = anchor_pairs + milestone_pairs
    rb = spearman(*zip(*both)) if both else float("nan")
    if rb != rb:
        decision = "unknown (fewer than 3 pairs)"
    else:
        decision = "keep 10B" if rb >= RHO_MIN else "rerun the short runs at 20B"
    return dict(anchors=dict(n=len(anchor_pairs), rho=ra, note="10B run vs its 100B run (equal terms)"),
                milestones=dict(n=len(milestone_pairs), rho=rm,
                                note="CHEAP EXTRA SIGNAL: ckpt_10B inside a 100B run is still at a high LR, not a 10B run"),
                together=dict(n=len(both), rho=rb), rule=f"rho < {RHO_MIN} -> 20B", decision=decision)


def cmd_rank(board_runs: str, milestones: str, plan_path: Path) -> dict:
    rec = json.loads(plan_path.read_text(encoding="utf-8"))
    runs = pd.read_csv(board_runs)
    dev = runs.groupby("exp")["dev_MASE"].apply(lambda v: float(np.exp(np.mean(np.log(pd.to_numeric(v)))))).to_dict()
    anchors = [(dev[k], dev[v["anchor"]]) for k, v in rec["runs"].items() if "anchor" in v and k in dev and v["anchor"] in dev]
    ms = pd.read_csv(milestones) if milestones and Path(milestones).exists() else pd.DataFrame()
    pairs = []
    if len(ms):
        ms = ms[ms["points_B"] == int(round(rec["points"] / 1e9))]
        final = runs.set_index("run_id")["dev_MASE"].to_dict()
        pairs = [(float(r.dev_MASE), float(final[r.run_id])) for r in ms.itertuples() if r.run_id in final
                 and str(r.dev_MASE) not in ("", "nan")]
    return rank_check(anchors, pairs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("plan", "fit", "rank"))
    ap.add_argument("--from", dest="from_")
    ap.add_argument("--anchors", nargs="+", default=[])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--points", type=float, default=1e10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--board-runs")
    ap.add_argument("--milestones")
    ap.add_argument("--plan", default=str(CAMPAIGNS / "x3_mix.json"))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "plan":
        if len(a.anchors) != 4:
            raise SystemExit("plan needs 4 --anchors: B0, the X1 config and 2 Phase 2 winners (owner 2026-10-29)")
        rec = plan(a.from_, a.anchors, a.n, a.points, a.seed, force=a.force)
        print(f"X3: {len(rec['runs'])} configs over groups {rec['groups']}; campaign configs/campaign/x3.yaml")
    elif a.cmd == "fit":
        print(json.dumps(cmd_fit(a.board_runs, Path(a.plan)), indent=1, default=str))
    else:
        print(json.dumps(cmd_rank(a.board_runs, a.milestones, Path(a.plan)), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
