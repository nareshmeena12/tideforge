"""X1 stacking, X2 leave-one-role-out, X4 (Phase 3 tools, owner OK 2026-10-29). Writes experiment configs; never runs anything.

    python -m tideforge.campaign.stack x1 --winners G4.3 G1.8 G6.1      # ranked by gain, best first
    python -m tideforge.campaign.stack x1 --board runs/board/board.csv  # or: the board's confirmed wins, by dev gain
    python -m tideforge.campaign.stack x2 --from X1.3
    python -m tideforge.campaign.stack x4 --from X1.3 [--points 2e11]
    python -m tideforge.campaign.stack report --winners G4.3 G1.8 G6.1 --board runs/board/board.csv --noise e07.json

x1: X1.k = the union of the first k winners (switches, gen changes, train items, model options; seeds [0]). A conflict raises:
    a different base; the same switch with other (resolved) parameters; the same gen field, gen addition, train key or model
    option with another value; two switches overriding the same generator layer, setting the same profile key or adding the
    same group kind (tideforge.gen.switches.plan). Group-kind shares are weights relative to the v0 kinds, so a union cannot
    overflow a budget: more scenarios dilute each other, which X1's gain check shows.
x2: X2.<group>.<from> = the config with one role group's data items removed (configs/campaign/roles.yaml); the model is held
    fixed (M1 stays on, owner 2026-10-29).
x4: X4 = the final recipe's config at --points (default 200B; the LR schedule follows target_points, so its 100B arm is the
    final recipe's own 100B run, not this run's 100B milestone).
report: per X1.k the predicted gain (sum of the single gains of its winners) against the observed one; |observed - predicted|
    above E0.7's range is flagged as an interaction.
Existing files are never overwritten (--force to replace).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

from tideforge import expcfg
from tideforge.gen import switches as SW

REPO = Path(__file__).resolve().parents[2]
ROLES = REPO / "configs" / "campaign" / "roles.yaml"


class Conflict(ValueError):
    pass


def roles(path: Path = ROLES) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def check_roles(r: dict | None = None) -> None:
    """Every registered switch is in exactly one group or in controls; amount switches exist."""
    r = r or roles()
    seen: dict = {}
    for g, d in r["groups"].items():
        for s in d.get("switches", []):
            if s in seen:
                raise ValueError(f"switch {s} in groups {seen[s]} and {g}")
            seen[s] = g
    for s in r.get("controls", []):
        if s in seen:
            raise ValueError(f"switch {s} is a control and in group {seen[s]}")
        seen[s] = "controls"
    missing = sorted(set(SW.REGISTRY) - set(seen))
    unknown = sorted(set(seen) - set(SW.REGISTRY))
    if missing or unknown:
        raise ValueError(f"roles.yaml: switches without a group {missing}; unknown switches {unknown}")
    for param, names in r.get("amount", {}).items():
        for n in names:
            if param not in SW.REGISTRY[n].params:
                raise ValueError(f"roles.yaml amount: {n} has no parameter {param!r}")


def raw(exp_id: str) -> dict:
    return dict(expcfg.load_exp(exp_id).raw)


def _put(dst: dict, key: str, val, where: str) -> None:
    if key in dst and dst[key] != val:
        raise Conflict(f"{where} {key}: {dst[key]!r} vs {val!r}")
    dst[key] = val


def union(ids: list[str]) -> dict:
    """The union of the configs' data and model settings (see the module docstring for what conflicts)."""
    out = dict(base=None, gen={}, switches={}, train={}, model={})
    resolved: dict = {}
    for i in ids:
        d = raw(i)
        base = d.get("base", "b0")
        if out["base"] not in (None, base):
            raise Conflict(f"{i}: base {base} vs {out['base']}")
        out["base"] = base
        for k, v in (d.get("gen") or {}).items():
            if k == "gen_additions":
                for ak, av in dict(v).items():
                    _put(out["gen"].setdefault("gen_additions", {}), ak, av, f"{i}: gen_additions")
            else:
                _put(out["gen"], k, v, f"{i}: gen")
        for name, params in (d.get("switches") or {}).items():
            (n, p), = SW.normalize({name: params})
            if n in resolved and resolved[n] != p:
                x, y = dict(resolved[n]), dict(p)
                diff = {k: (x[k], y[k]) for k in x if x[k] != y[k]}
                raise Conflict(f"{i}: switch {n} with other parameters (earlier, this): {diff}")
            resolved[n] = p
            out["switches"][n] = params
        for k, v in (d.get("train") or {}).items():
            _put(out["train"], k, v, f"{i}: train")
        for k, v in expcfg.check_model(d.get("model")).items():
            _put(out["model"], k, v, f"{i}: model")
    plan_conflicts(out["switches"])
    return {k: v for k, v in out.items() if v not in ({}, None)} | {"base": out["base"]}


def plan_conflicts(switches: dict) -> None:
    owner: dict = {}
    for e in SW.plan(switches):
        if e[0] in ("layer", "profile", "group_kind"):
            k = (e[0], e[2])
            if k in owner and owner[k] != e[1]:
                raise Conflict(f"switches {owner[k]} and {e[1]} both change {e[0]} {e[2]!r}")
            owner[k] = e[1]


def write(cfg_id: str, d: dict, notes: str, out_dir: Path, force: bool = False) -> Path:
    p = out_dir / f"{cfg_id}.yaml"
    if p.exists() and not force:
        raise FileExistsError(f"{p} exists (--force to replace)")
    body = {"id": cfg_id, "base": d.get("base", "b0")}
    for k in ("gen", "switches", "train", "model"):
        if d.get(k):
            body[k] = d[k]
    body.update(seeds=[0], notes=notes)
    p.write_text("# GENERATED by tideforge.campaign.stack\n" + yaml.safe_dump(body, sort_keys=False, width=120),
                 encoding="utf-8", newline="\n")
    expcfg.load_exp(p)                                  # it must load (unknown keys, switches, params all raise)
    return p


def x1(winners: list[str], out_dir: Path, force=False) -> list[Path]:
    out = []
    for k in range(1, len(winners) + 1):
        d = union(winners[:k])
        out.append(write(f"X1.{k}", d, f"X1 step {k}: union of {', '.join(winners[:k])} (ranked by gain, best first)",
                         out_dir, force))
    return out


def remove_group(d: dict, group: str, r: dict | None = None) -> dict:
    g = (r or roles())["groups"][group]
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in d.items()}
    sw = {k: v for k, v in (out.get("switches") or {}).items() if k not in set(g.get("switches", []))}
    gen = dict(out.get("gen") or {})
    for key in g.get("gen", []):
        if key.startswith("gen_additions."):
            ga = dict(gen.get("gen_additions") or {})
            ga.pop(key.split(".", 1)[1], None)
            gen["gen_additions"] = ga
            if not ga:
                gen.pop("gen_additions")
        else:
            gen.pop(key, None)
    train = {k: v for k, v in (out.get("train") or {}).items() if k not in set(g.get("train", []))}
    out.update(switches=sw, gen=gen, train=train)          # model: unchanged (held fixed)
    return {k: v for k, v in out.items() if v not in ({}, None)} | {"base": d.get("base", "b0")}


def groups_present(d: dict, r: dict | None = None) -> list[str]:
    r = r or roles()
    keys = set(d.get("switches") or {}) | set(d.get("train") or {})
    keys |= {k for k in (d.get("gen") or {}) if k != "gen_additions"}
    keys |= {f"gen_additions.{k}" for k in (d.get("gen") or {}).get("gen_additions", {})}
    return [g for g, v in r["groups"].items() if keys & set(v.get("switches", []) + v.get("gen", []) + v.get("train", []))]


def x2(from_id: str, out_dir: Path, force=False) -> list[Path]:
    r = roles()
    d = raw(from_id)
    d = {k: d[k] for k in ("base", "gen", "switches", "train", "model") if k in d}
    return [write(f"X2.{g}.{from_id}", remove_group(d, g, r), f"X2: {from_id} without the {g} role group (model held fixed)",
                  out_dir, force) for g in groups_present(d, r)]


def x4(from_id: str, out_dir: Path, points: float = 2e11, force=False) -> Path:
    d = raw(from_id)
    d = {k: d[k] for k in ("base", "gen", "switches", "train", "model") if k in d}
    d["train"] = dict(d.get("train") or {}, target_points=float(points))
    return write("X4", d, f"X4: the final recipe ({from_id}) at {points / 1e9:.0f}B points", out_dir, force)


def winners_from_board(board_csv: str) -> list[str]:
    b = pd.read_csv(board_csv)
    w = b[(b["verdict"] == "win") & ~b["exp"].astype(str).str.startswith(("X", "E0."))]
    return w.sort_values("dev_gain", ascending=False)["exp"].tolist()


def report(winners: list[str], board_csv: str, noise_path: str | None) -> pd.DataFrame:
    import json
    from tideforge.eval import noise as NZ
    b = pd.read_csv(board_csv).set_index("exp")
    fl = NZ.floor(json.loads(Path(noise_path).read_text(encoding="utf-8")) if noise_path else None, "MASE")
    rows = []
    for k in range(1, len(winners) + 1):
        pred = float(sum(b.loc[w, "dev_gain"] for w in winners[:k]))
        obs = float(b.loc[f"X1.{k}", "dev_gain"]) if f"X1.{k}" in b.index else float("nan")
        diff = obs - pred
        flag = "" if fl is None or diff != diff else ("interaction" if abs(diff) > fl else "adds up")
        rows.append(dict(step=f"X1.{k}", added=winners[k - 1], predicted=pred, observed=obs, diff=diff, check=flag))
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("x1", "x2", "x4", "report"))
    ap.add_argument("--winners", nargs="+")
    ap.add_argument("--board")
    ap.add_argument("--from", dest="from_")
    ap.add_argument("--points", type=float, default=2e11)
    ap.add_argument("--noise")
    ap.add_argument("--out", default=str(expcfg.EXP_DIR))
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    win = a.winners or (winners_from_board(a.board) if a.board and a.cmd == "x1" else None)
    if a.cmd == "x1":
        if not win:
            raise SystemExit("x1 needs --winners or --board")
        for p in x1(win, out, a.force):
            print(f"wrote {p}")
    elif a.cmd == "x2":
        for p in x2(a.from_, out, a.force):
            print(f"wrote {p}")
    elif a.cmd == "x4":
        print(f"wrote {x4(a.from_, out, a.points, a.force)}")
    else:
        print(report(a.winners, a.board, a.noise).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
