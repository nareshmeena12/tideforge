"""Sizing plan for the real half of the post-training mix: how many points each dataset contributes, its share and its epochs.

  python -m tidemark.real.plan [--manifest $SCRATCH/tidemark/real/manifest.json] [--real-points 10e9] [--max-epochs 4]
                               [--weather-max-share 0.4] [--out plan.json]

Rules (owner, 2026-10-24):
  * only data already downloaded: a dataset's unique points are the points in the manifest (hourly weatherbench: the shards taken);
  * every non-weather dataset is capped at `max_epochs` passes over its unique points;
  * all weatherbench_* datasets are ONE group: one cap (max_epochs x the group's unique points), one share, split inside the group
    in proportion to the members' unique points;
  * within the caps the real half is split in proportion to unique points (every unit sees the same number of passes), which is the
    water-filling solution of "as even in epochs as the caps allow". `unit_weights` overrides the proportions; `weather_max_share`
    (default None = off) caps the weather group's share of the real half and hands the rest to the other datasets.
The plan is also what the loader aims at: `targets` = planned points / real budget per dataset.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

WEATHER_PREFIX = "weatherbench_"


def unit_of(name: str) -> str:
    return "weather" if name.startswith(WEATHER_PREFIX) else name


WEATHER_SPLIT = {"hourly": 0.5, "daily": 0.4, "weekly": 0.1}      # owner, 2026-10-24: shares of the weather group


def weather_part(name: str) -> str:
    return "hourly" if name.startswith("weatherbench_hourly_") else ("daily" if name == "weatherbench_daily" else "weekly")


def water_fill(total: float, weights: dict[str, float], caps: dict[str, float]) -> tuple[dict[str, float], float]:
    """Split `total` in proportion to weights, no unit above its cap; the excess of a capped unit goes to the others.
    Returns ({unit: points}, points that could not be placed because every unit is capped)."""
    alloc = {k: 0.0 for k in weights}
    free = {k for k in weights if weights[k] > 0 and caps[k] > 0}
    left = total
    while free and left > 1e-9:
        w = sum(weights[k] for k in free)
        over = {k for k in free if alloc[k] + left * weights[k] / w > caps[k] + 1e-9}
        if not over:
            for k in free:
                alloc[k] += left * weights[k] / w
            left = 0.0
            break
        for k in over:                                   # fill the capped units to their cap, redistribute the rest
            left -= caps[k] - alloc[k]
            alloc[k] = caps[k]
        free -= over
    return alloc, max(left, 0.0)


def build_plan(unique: dict[str, float], real_points: float, max_epochs: float = 4.0, unit_weights: dict[str, float] | None = None,
               weather_max_share: float | None = None, weather_split: dict[str, float] | None = None) -> dict:
    units: dict[str, float] = {}
    for d, u in unique.items():
        units[unit_of(d)] = units.get(unit_of(d), 0.0) + float(u)
    weights = {k: (unit_weights or {}).get(k, v) for k, v in units.items()}
    caps = {k: max_epochs * v for k, v in units.items()}
    if weather_max_share is not None and "weather" in caps:
        caps["weather"] = min(caps["weather"], weather_max_share * real_points)
    alloc, unplaced = water_fill(real_points, weights, caps)
    parts: dict[str, float] = {}
    if weather_split:
        if abs(sum(weather_split.values()) - 1.0) > 1e-6:
            raise ValueError("weather_split must sum to 1")
        for d, u in unique.items():
            if unit_of(d) == "weather":
                parts[weather_part(d)] = parts.get(weather_part(d), 0.0) + float(u)
    rows = {}
    for d, u in unique.items():
        k = unit_of(d)
        if k == "weather" and weather_split:
            g = weather_part(d)
            planned = alloc[k] * weather_split.get(g, 0.0) * (float(u) / parts[g]) if parts.get(g) else 0.0
        else:
            planned = alloc[k] * (float(u) / units[k]) if units[k] else 0.0
        rows[d] = {"unique_points": float(u), "planned_points": planned, "share": planned / real_points if real_points else 0.0,
                   "epochs": planned / float(u) if u else 0.0, "unit": k}
    return {"real_points": real_points, "max_epochs": max_epochs, "weather_max_share": weather_max_share,
            "weather_split": weather_split, "unplaced_points": unplaced, "datasets": rows,
            "units": {k: {"unique_points": units[k], "planned_points": alloc[k], "share": alloc[k] / real_points if real_points else 0.0,
                          "epochs": alloc[k] / units[k] if units[k] else 0.0} for k in units},
            "targets": {d: r["share"] for d, r in rows.items()}}


def match_plan(unique: dict[str, float], first: dict, max_epochs: float | None = None) -> dict:
    """The planned points of every dataset taken from an earlier plan (`first`); `unique` (the new pool) only changes the epochs.
    Used so that a run with a bigger sampling pool differs from the first run in the pool only: same points, same shares, same loader
    targets for every dataset. The two plans must cover the same datasets."""
    old = first["datasets"]
    if set(old) != set(unique):
        raise SystemExit(f"--match-plan: the datasets differ. Only in the first plan: {sorted(set(old) - set(unique))}; "
                         f"only in the new pool: {sorted(set(unique) - set(old))}")
    real_points = float(first["real_points"])
    cap = float(max_epochs if max_epochs is not None else first["max_epochs"])
    rows = {}
    for d, u in unique.items():
        planned = float(old[d]["planned_points"])
        rows[d] = {"unique_points": float(u), "planned_points": planned, "share": planned / real_points if real_points else 0.0,
                   "epochs": planned / float(u) if u else 0.0, "unit": unit_of(d),
                   "unique_points_first_plan": float(old[d]["unique_points"]), "epochs_first_plan": float(old[d]["epochs"])}
    units: dict[str, dict] = {}
    for d, r in rows.items():
        k = r["unit"]
        g = units.setdefault(k, {"unique_points": 0.0, "planned_points": 0.0})
        g["unique_points"] += r["unique_points"]
        g["planned_points"] += r["planned_points"]
    for g in units.values():
        g["share"] = g["planned_points"] / real_points if real_points else 0.0
        g["epochs"] = g["planned_points"] / g["unique_points"] if g["unique_points"] else 0.0
    return {"real_points": real_points, "max_epochs": cap, "weather_max_share": first.get("weather_max_share"),
            "weather_split": first.get("weather_split"), "unplaced_points": float(first.get("unplaced_points", 0.0)),
            "matched_from": first.get("matched_from") or "first plan", "datasets": rows, "units": units,
            "targets": {d: r["share"] for d, r in rows.items()}}


def report(plan: dict, extra: dict[str, dict] | None = None) -> str:
    rows = plan["datasets"]
    cap = plan["max_epochs"]
    matched = any("epochs_first_plan" in r for r in rows.values())
    L = [f"REAL HALF: {plan['real_points'] / 1e9:.2f}B points planned over {sum(r['unique_points'] for r in rows.values()) / 1e9:.2f}B unique "
         f"points downloaded; cap {cap:g} epochs per non-weather dataset, weatherbench_* as one group"
         + (f"\nPLANNED POINTS AND SHARES REUSED from {plan.get('matched_from')}: only the sampling pool (and so the epochs) differs" if matched else ""),
         f"{'dataset':48s} {'unique (B)':>11s} {'planned (B)':>12s} {'share':>7s} {'epochs':>7s}" + (f" {'epochs 1st':>10s} {'unique 1st (B)':>14s}" if matched else "") + "  note"]
    for d, r in sorted(rows.items(), key=lambda kv: -kv[1]["planned_points"]):
        note = ""
        if r["epochs"] > cap + 1e-9:
            note = f"ABOVE {cap:g} EPOCHS"
        elif r["epochs"] > cap - 1e-9:
            note = "at the cap"
        if r["unit"] == "weather":
            note = (note + " " if note else "") + "(weather group)"
        first = f" {r['epochs_first_plan']:>10.2f} {r['unique_points_first_plan'] / 1e9:>14.3f}" if matched else ""
        L.append(f"{d:48s} {r['unique_points'] / 1e9:>11.3f} {r['planned_points'] / 1e9:>12.3f} {r['share']:>6.1%} {r['epochs']:>7.2f}{first}  {note}")
    u = plan["units"]
    if "weather" in u:
        w = u["weather"]
        L.append(f"{'weather group (all weatherbench_*)':48s} {w['unique_points'] / 1e9:>11.3f} {w['planned_points'] / 1e9:>12.3f} "
                 f"{w['share']:>6.1%} {w['epochs']:>7.2f}")
    tot_p = sum(r["planned_points"] for r in rows.values())
    L.append(f"{'TOTAL':48s} {sum(r['unique_points'] for r in rows.values()) / 1e9:>11.3f} {tot_p / 1e9:>12.3f} {tot_p / plan['real_points']:>6.1%}")
    if plan["unplaced_points"] > 1:
        L.append(f"WARNING: {plan['unplaced_points'] / 1e9:.2f}B points of the real half cannot be placed within the caps")
    big = [d for d, r in rows.items() if r["epochs"] > cap + 1e-9]
    if big:
        L.append("WARNING: above the epoch cap: " + ", ".join(big))
    return "\n".join(L)


def unique_from_manifest(path: Path) -> dict[str, float]:
    """Unique points per dataset from manifest.json (points as downloaded, NaN included) or from shards/index.json (VALID points
    after conversion: what the epoch caps count)."""
    man = json.loads(Path(path).read_text())
    return {d: float(v["total_points"]) for d, v in man["datasets"].items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", default=None, help="manifest.json, or shards/index.json (valid points: use this one for the "
                    "final numbers); default $SCRATCH/tidemark/real/shards/index.json if it exists, else manifest.json")
    ap.add_argument("--real-points", type=float, default=10e9)
    ap.add_argument("--max-epochs", type=float, default=4.0)
    ap.add_argument("--weather-max-share", type=float, default=None)
    ap.add_argument("--weather-split", default="0.5,0.4,0.1", help="hourly,daily,weekly shares of the weather group "
                    "(default 0.5,0.4,0.1; 'none' = in proportion to unique points)")
    ap.add_argument("--non-weather-at-cap", action="store_true", help="fill every non-weather dataset to its epoch cap first and give the rest of the "
                    "real half to the weather group (real200b: 100B real, non-weather 4 epochs, weather about 1.2 passes). Without it the real half is "
                    "split in proportion to unique points.")
    ap.add_argument("--match-plan", default=None, metavar="PLAN_JSON", help="reuse every dataset's planned points (and so its share and its loader "
                    "target) from this earlier plan; --manifest then only gives the unique points of the new sampling pool, which change the "
                    "epochs. The other planning options are ignored.")
    ap.add_argument("--out", default=None, help="write the plan JSON here (the loader reads its `targets`)")
    a = ap.parse_args(argv)
    import os
    base = Path(os.environ.get("SCRATCH", ".")) / "tidemark" / "real"
    man = a.manifest or str(base / "shards" / "index.json" if (base / "shards" / "index.json").exists() else base / "manifest.json")
    print(f"unique points from {man}" + ("  (valid points, NaN excluded)" if man.endswith("index.json") else "  (as downloaded: NaN counted; convert first for the final numbers)"))
    split = None if a.weather_split == "none" else dict(zip(("hourly", "daily", "weekly"), (float(x) for x in a.weather_split.split(","))))
    if a.match_plan:
        first = json.loads(Path(a.match_plan).read_text())
        plan = match_plan(unique_from_manifest(Path(man)), first, a.max_epochs if "--max-epochs" in (argv or sys.argv) else None)
        plan["matched_from"] = str(a.match_plan)
    else:
        uniq = unique_from_manifest(Path(man))
        weights = None
        if a.non_weather_at_cap:                                 # a weight so large that the non-weather units reach their caps before weather gets any share
            weights = {}
            for d, u in uniq.items():
                weights[unit_of(d)] = weights.get(unit_of(d), 0.0) + (float(u) if unit_of(d) == "weather" else 1e6 * float(u))
        plan = build_plan(uniq, a.real_points, a.max_epochs, weights, a.weather_max_share, split)
    print(report(plan))
    if a.out:
        Path(a.out).write_text(json.dumps(plan, indent=1))
        print(f"plan written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
