"""G2: build the calendar tables from the `holidays` package (offline; owner-approved dependency).

    python tools/build_calendars.py                 # writes tideforge/gen/ext/calendars/calendars_v1.npz + .json + report
    python tools/build_calendars.py --check         # rebuild in memory, fail if the sha256 differs from the manifest
    python tools/build_calendars.py --diff DIR      # rebuild into DIR (nothing committed is touched) and print a CONTENT
                                                    # diff against the committed table: cells, calendars, names, metadata

Per calendar (country, or country-subdivision) and per day from 1990-01-01 to 2040-12-31:
  code     int8 holiday type (TYPES below; 0 = none), from the English holiday name (language en_US where the package has
           it). Several holidays on one day: the first type in PRIORITY that matches. Unmatched names -> "other" (counted).
  weekend  int8 1 on the calendar's weekend days (WEEKEND rules, with their real changes, e.g. the UAE in 2022)
  dst      int8 1 while daylight saving time is in force at the end of the day (rules in DST, no tz database)
Plus per calendar: valid years (the longest run of years in which every moving-festival type of that calendar is present,
so a row never shows a missing festival as a fact), hemisphere, held-out flag (G2.4 only). The .npz is written with fixed
zip timestamps, so the bytes (and the sha256) depend only on the package version and this file.
The report (calendars_v1_report.txt) lists every holiday name -> type per calendar and the unmatched names.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "tideforge" / "gen" / "ext" / "calendars"
VERSION = "v1"
START, END = dt.date(1990, 1, 1), dt.date(2040, 12, 31)
NDAYS = (END - START).days + 1

TYPES = ["none", "new_year", "civic", "easter", "christmas", "islamic", "hindu_buddhist", "chinese", "jewish",
         "other_religious", "observed", "later_day", "other"]
T = {n: i for i, n in enumerate(TYPES)}
MOVING = ("easter", "islamic", "hindu_buddhist", "chinese", "jewish")
KEYS = {
    "observed": ["observed", "substitut", "day off", "bridge", "in lieu", "alternative holiday", "compensat", "replacement"],
    "chinese": ["lunar new year", "chinese new year", "spring festival", "seollal", "korean new year", "tet$", "mid-autumn",
                "chuseok", "dragon boat", "qingming", "tomb-sweeping", "tomb sweeping", "lantern", "double ninth",
                "chung yeung", "ching ming", "tuen ng", "29 of lunar"],
    "islamic": ["eid", "ramadan", "arafah", "arafat", "ashura", "islamic new year", "hijri", "prophet", "mawlid", "isra",
                "miraj", "mi'raj", "lailat", "muharram", "sacrifice", "kurban", "bakr", "idul", "hari raya", "nuzul",
                "laylat", "maulid", "milad"],
    "hindu_buddhist": ["diwali", "deepavali", "holi$", "dussehra", "dashain", "tihar", "bhai tika", "vesak", "buddha", "poya",
                       "ganesh", "navaratri", "janmashtami", "pongal", "onam", "thaipusam", "makha bucha", "visakha",
                       "songkran", "ram navami", "shivratri", "raksha", "chhath", "ugadi", "tamil", "sinhala", "bihu",
                       "vaisakhi", "guru", "durga", "ekadashi", "dashami", "duwadashi", "purnima", "poornima", "losar",
                       "asalha", "khao phansa", "magha", "wesak", "mahavir", "gandhi", "bhai", "teej", "lhosar", "dashera"],
    "jewish": ["rosh hashanah", "yom kippur", "sukkot", "pesach", "passover", "shavuot", "simchat", "hanukkah", "purim",
               "shemini", "yom ha"],
    "easter": ["easter", "good friday", "holy thursday", "maundy", "ascension", "pentecost", "whit", "corpus christi",
               "carnival", "shrove", "ash wednesday", "holy saturday", "holy week", "sacred heart"],
    "christmas": ["christmas", "boxing day", "st. stephen", "saint stephen", "st stephen"],
    "new_year": ["new year"],
    "other_religious": ["saint", "st.", "assumption", "all saints", "all souls", "immaculate", "epiphany", "reformation",
                        "orthodox", "annunciation", "our lady", "feast", "transfiguration", "coptic", "virgin", "three kings",
                        "candlemas", "nativity", "theophany", "lady of", "corpus", "baptism", "cross"],
    "civic": ["independence", "national", "republic", "constitution", "labor", "labour", "workers", "victory", "liberation",
              "memorial", "veterans", "armed forces", "king", "queen", "emperor", "president", "revolution", "unity",
              "thanksgiving", "mother", "children", "youth", "martin luther", "washington", "columbus", "election",
              "flag", "heritage", "foundation", "founding", "statehood", "reconciliation", "freedom", "democracy",
              "human rights", "heroes", "martyr", "revolt", "culture", "birthday", "bank holiday", "anniversary",
              "remembrance", "commemoration", "sovereignty", "people", "women", "teachers", "senior", "citizen", "army",
              "day of", "health", "sports", "mountain", "marine", "showa", "greenery", "respect for the aged", "autumnal",
              "vernal", "equinox", "solstice", "culture", "kingdom", "monarch", "royal", "coronation", "accession",
              "chulalongkorn", "regent", "patriot", "civic", "proclamation", "battle", "uprising", "restoration", "state",
              "jubilee", "family", "hangul", "victoria", "canada", "australia", "waitangi", "anzac", "juneteenth",
              "emancipation", "lincoln", "cesar", "pioneer", "admission", "kamehameha", "patriots", "evacuation",
              "rizal", "bonifacio", "ninoy", "valor", "araw", "seaweed", "tiradentes", "zumbi", "black awareness",
              "aboriginal", "truth", "general", "shinto"],
}
LATER = ["second day", "third day", "fourth day", "fifth day", "day after", "2nd day", "3rd day", "eve$"]
PRIORITY = ["observed", "chinese", "islamic", "hindu_buddhist", "jewish", "easter", "christmas", "new_year", "other_religious",
            "civic"]

# weekend rules: day-of-week sets (Mon = 0); changes listed as (from date, rule)
RULES = {"sat_sun": (5, 6), "fri_sat": (4, 5), "thu_fri": (3, 4), "sun": (6,), "fri": (4,), "sat": (5,)}
WEEKEND = {
    "AE": [(START, "thu_fri"), (dt.date(2006, 9, 1), "fri_sat"), (dt.date(2022, 1, 1), "sat_sun")],
    "SA": [(START, "thu_fri"), (dt.date(2013, 6, 29), "fri_sat")],
    "QA": [(START, "thu_fri"), (dt.date(2003, 9, 1), "fri_sat")],
    "EG": [(START, "fri_sat")], "IL": [(START, "fri_sat")], "BD": [(START, "fri_sat")], "NP": [(START, "sat")],
}
# DST rules (no tz database): (start rule, end rule) per period; a rule = (month, weekday, n) with n = -1 for the last
EU = [((3, 6, -1), (10, 6, -1))]
US = [(1990, (4, 6, 1), (10, 6, -1)), (2007, (3, 6, 2), (11, 6, 1))]
AU_S = [(1990, (10, 6, -1), (3, 6, -1)), (2008, (10, 6, 1), (4, 6, 1))]
NZ = [(1990, (10, 6, 1), (3, 6, 3)), (2007, (9, 6, -1), (4, 6, 1))]
DST = {**{c: "EU" for c in ("GB", "IE", "FR", "DE", "ES", "IT", "NL", "BE", "CH", "AT", "PL", "CZ", "SE", "NO", "DK", "FI",
                            "PT", "GR", "UA")},
       "US": "US", "CA": "US", "AU": "AU_S", "NZ": "NZ"}
SOUTH = {"AU", "NZ", "ZA", "BR", "AR", "CL"}

POOL = ["US", "CA", "MX", "BR", "AR", "CL", "CO", "GB", "IE", "FR", "DE", "ES", "IT", "NL", "BE", "CH", "AT", "PL", "CZ", "SE",
        "NO", "DK", "FI", "PT", "GR", "RU", "UA", "ZA", "NG", "KE", "AU", "NZ", "PH", "SA", "AE", "QA", "EG", "MA", "TR", "PK",
        "BD", "ID", "IN", "NP", "LK", "TH", "CN", "HK", "TW", "KR", "VN", "SG", "MY", "IL", "JP"]
SUBDIVS = [("US", "CA"), ("US", "TX"), ("DE", "BY"), ("DE", "BE"), ("ES", "CT"), ("CA", "QC"), ("AU", "NSW"), ("AU", "VIC"),
           ("IN", "MH"), ("CH", "ZH"), ("BR", "SP"), ("MY", "10")]
HELD_OUT = ("GB", "IN", "KR", "AU", "BR")          # G2.4 only (owner 2026-10-28)


def _has(n: str, keys) -> bool:
    """A key matches at a word start; a trailing $ means the whole word ("holi$" is not "holiday")."""
    return any(re.search(r"(?<![a-z])" + re.escape(k[:-1]) + r"(?![a-z])" if k.endswith("$") else r"(?<![a-z])" + re.escape(k), n)
               for k in keys)


def classify(name: str) -> int:
    n = name.lower()
    hit = None
    for t in PRIORITY:
        if _has(n, KEYS[t]):
            hit = t
            break
    if hit != "observed" and _has(n, LATER[:-1]):          # 2nd.. day of a multi-day festival; an "eve" keeps its type
        return T["later_day"]
    return T[hit] if hit else T["other"]


def nth_weekday(year, month, weekday, n):
    if n > 0:
        d = dt.date(year, month, 1)
        d += dt.timedelta(days=(weekday - d.weekday()) % 7)
        return d + dt.timedelta(weeks=n - 1)
    d = dt.date(year + (month == 12), month % 12 + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def dst_days(rule: str) -> np.ndarray:
    out = np.zeros(NDAYS, np.int8)
    if rule == "EU":
        periods = [(1990, *EU[0])]
    else:
        periods = {"US": US, "AU_S": AU_S, "NZ": NZ}[rule]
    for y in range(START.year, END.year + 1):
        p = [q for q in periods if q[0] <= y][-1]
        a, b = nth_weekday(y, *p[1]), nth_weekday(y, *p[2])
        ia, ib = (a - START).days, (b - START).days
        if a < b:                                   # northern: on from a to b
            out[ia:ib] = 1
        else:                                       # southern: on from a to the end of the year, and from Jan 1 to b
            out[ia:min(NDAYS, (dt.date(y + 1, 1, 1) - START).days)] = 1
            out[(dt.date(y, 1, 1) - START).days:ib] = 1
    return out


def weekend_days(country: str) -> np.ndarray:
    ch = WEEKEND.get(country, [(START, "sat_sun")])
    days = [START + dt.timedelta(days=i) for i in range(NDAYS)]
    out = np.zeros(NDAYS, np.int8)
    for i, d in enumerate(days):
        rule = [r for f, r in ch if f <= d][-1]
        out[i] = d.weekday() in RULES[rule]
    return out


def holidays_for(country, subdiv):
    import holidays
    yrs = range(START.year, END.year + 1)
    try:
        return holidays.country_holidays(country, subdiv=subdiv, years=yrs, language="en_US"), True
    except Exception:
        return holidays.country_holidays(country, subdiv=subdiv, years=yrs), False


FESTIVALS = {"diwali": ["diwali", "deepavali"], "lunar_new_year": ["lunar new year", "chinese new year", "korean new year",
              "seollal", "tet$", "spring festival"], "eid_fitr": ["eid al-fitr", "hari raya puasa", "idul fitri"],
              "eid_adha": ["eid al-adha", "hari raya haji", "idul adha"], "rosh_hashanah": ["rosh hashanah"],
              "yom_kippur": ["yom kippur"], "pesach": ["pesach", "passover"], "vesak": ["vesak", "wesak", "buddha"]}


def festival(name: str) -> str | None:
    n = name.lower()
    return next((f for f, keys in FESTIVALS.items() if _has(n, keys)), None)


def valid_years(names_by_year: dict) -> tuple[int, int]:
    """Longest run of years in which every key lunisolar / Islamic / Jewish festival (FESTIVALS) of the calendar is present.
    Only festivals present in >= 30% of the years count: rarer ones are changes in the law (California's Diwali from 2026),
    not missing data. Other holidays are taken as the package gives them (law changes included)."""
    yrs = list(range(START.year, END.year + 1))
    alln = set().union(*names_by_year.values()) if names_by_year else set()
    core = [n for n in alln if sum(n in names_by_year.get(y, set()) for y in yrs) >= 0.3 * len(yrs)]
    ok = [all(n in names_by_year.get(y, set()) for n in core) for y in yrs]
    best, cur, bs = (0, 0), 0, 0
    for i, o in enumerate(ok + [False]):
        if o:
            if cur == 0:
                bs = i
            cur += 1
        else:
            if cur > best[1] - best[0]:
                best = (bs, bs + cur)
            cur = 0
    return yrs[best[0]], yrs[best[1] - 1]


def _base(name: str) -> str:
    return re.sub(r"\s*\((observed|estimated)[^)]*\)", "", name).strip()


def build():
    import holidays
    warnings.filterwarnings("ignore")
    cals = [(c, None) for c in POOL] + list(SUBDIVS)
    code = np.zeros((len(cals), NDAYS), np.int8)
    wk = np.zeros_like(code)
    dst = np.zeros_like(code)
    meta, report, unmatched = [], [], {}
    for k, (c, sd) in enumerate(cals):
        h, en = holidays_for(c, sd)
        names, by_year = {}, {}
        for d, nm in sorted(h.items()):
            if not (START <= d <= END):
                continue
            t = classify(nm)
            code[k, (d - START).days] = t
            for part in nm.split("; "):
                names.setdefault(part, classify(part))
                f = festival(part)
                if f:
                    by_year.setdefault(d.year, set()).add(f)
        wk[k] = weekend_days(c)
        if c in DST:
            dst[k] = dst_days(DST[c])
        y0, y1 = valid_years(by_year)
        cid = c if sd is None else f"{c}-{sd}"
        meta.append(dict(id=cid, country=c, subdiv=sd, english=en, valid=[y0, y1], south=c in SOUTH,
                         dst=DST.get(c), held_out=c in HELD_OUT, weekend=[(str(f), r) for f, r in WEEKEND.get(c, [(START, "sat_sun")])]))
        report.append(f"\n== {cid}  (english names: {en}; valid {y0}-{y1}; weekend {meta[-1]['weekend']}; dst {DST.get(c)})")
        for nm, t in sorted(names.items()):
            report.append(f"   {TYPES[t]:15s} {nm}")
            if t == T["other"]:
                unmatched.setdefault(cid, []).append(nm)
    n_un = sum(len(v) for v in unmatched.values())
    n_all = sum(1 for line in report if line.startswith("   "))
    head = [f"calendars {VERSION}: {len(cals)} calendars, {NDAYS} days {START}..{END}, holidays {holidays.__version__}",
            f"holiday names: {n_all}, unmatched -> other: {n_un}"] + [f"   {k}: {len(v)}" for k, v in unmatched.items()]
    return dict(code=code, weekend=wk, dst=dst), meta, "\n".join(head + report) + "\n", holidays.__version__


def npz_bytes(arrays: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name in sorted(arrays):
            b = io.BytesIO()
            np.save(b, arrays[name], allow_pickle=False)
            zi = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(zi, b.getvalue())
    return buf.getvalue()


def content_sha(arrays: dict) -> str:
    """sha256 over each array's name, dtype, shape and raw bytes (independent of zip / deflate / .npy header bytes)."""
    h = hashlib.sha256()
    for name in sorted(arrays):
        x = np.ascontiguousarray(arrays[name])
        h.update(f"{name}|{x.dtype.str}|{x.shape}|".encode())
        h.update(x.tobytes())
    return h.hexdigest()


def versions() -> str:
    import platform
    import zlib
    import holidays
    return (f"python {platform.python_version()}, numpy {np.__version__}, zlib {zlib.ZLIB_RUNTIME_VERSION}, "
            f"holidays {holidays.__version__}")


def diff(out_dir: Path, arrays: dict, meta: list, report: str, data: bytes) -> int:
    """Rebuild written to out_dir; prints the content diff against the committed table. 0 = same contents."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"calendars_{VERSION}.npz").write_bytes(data)
    (out_dir / f"calendars_{VERSION}_report.txt").write_text(report, encoding="utf-8", newline="\n")
    (out_dir / f"calendars_{VERSION}_meta.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    with np.load(OUT / f"calendars_{VERSION}.npz") as z:
        old = {k: z[k] for k in z.files}
    m = json.loads((OUT / f"calendars_{VERSION}.json").read_text(encoding="utf-8"))
    print(f"this machine: {versions()}")
    print(f"file sha256   committed {m['sha256']}  rebuilt {hashlib.sha256(data).hexdigest()}")
    print(f"content sha256 committed {content_sha(old)}  rebuilt {content_sha(arrays)}")
    ids = [c["id"] for c in meta]
    bad = 0
    for name in sorted(set(old) | set(arrays)):
        a, b = old.get(name), arrays.get(name)
        if a is None or b is None or a.shape != b.shape:
            print(f"{name}: shape committed {None if a is None else a.shape} rebuilt {None if b is None else b.shape}")
            bad += 1
            continue
        cells = np.argwhere(a != b)
        print(f"{name}: {len(cells)} differing (calendar, day) cells")
        bad += len(cells)
        by_cal: dict = {}
        for k, d in cells:
            by_cal.setdefault(ids[k], []).append(int(d))
        for cid, days in sorted(by_cal.items()):
            first = [str(START + dt.timedelta(days=d)) for d in days[:5]]
            print(f"   {cid}: {len(days)} days, first {first}")
            if name == "code":
                c, sd = (cid.split("-", 1) + [None])[:2]
                h, _ = holidays_for(c, sd)
                for d in days[:10]:
                    day = START + dt.timedelta(days=d)
                    print(f"      {day}: committed {TYPES[a[ids.index(cid), d]]} rebuilt {TYPES[b[ids.index(cid), d]]} "
                          f"names {h.get(day)!r}")
    old_meta = {c["id"]: c for c in m["calendars"]}
    for c in meta:
        o = old_meta.get(c["id"])
        if o is None:
            print(f"meta {c['id']}: new calendar"); bad += 1
        else:
            for k in c:
                if json.loads(json.dumps(c[k])) != o.get(k):
                    print(f"meta {c['id']}.{k}: committed {o.get(k)} rebuilt {c[k]}"); bad += 1
    for cid in sorted(set(old_meta) - set(ids)):
        print(f"meta {cid}: missing in the rebuild"); bad += 1
    rep_old = set((OUT / f"calendars_{VERSION}_report.txt").read_text(encoding="utf-8").splitlines()[2:])
    rep_new = set(report.splitlines()[2:])
    for line in sorted(rep_old - rep_new)[:40]:
        print(f"report only committed: {line.strip()}")
    for line in sorted(rep_new - rep_old)[:40]:
        print(f"report only rebuilt:   {line.strip()}")
    bad += len(rep_old ^ rep_new)
    same_bytes = hashlib.sha256(data).hexdigest() == m["sha256"]
    if bad == 0:
        print("CONTENTS EQUAL" + (" (file bytes equal too)" if same_bytes else
                                  " (the file bytes differ: zip / deflate / .npy header encoding only)"))
    else:
        print(f"CONTENTS DIFFER: {bad} differences (cells + metadata + report lines); rebuild in {out_dir}")
    return 0 if bad == 0 else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--diff", metavar="DIR", help="rebuild into DIR and print a content diff against the committed table")
    a = ap.parse_args(argv)
    arrays, meta, report, ver = build()
    data = npz_bytes(arrays)
    sha = hashlib.sha256(data).hexdigest()
    man = OUT / f"calendars_{VERSION}.json"
    if a.diff:
        return diff(Path(a.diff), arrays, meta, report, data)
    if a.check:
        m = json.loads(man.read_text(encoding="utf-8"))
        if m["sha256"] != sha:
            print(f"FAIL: rebuild sha256 {sha} != manifest {m['sha256']} (holidays {ver}, manifest built with "
                  f"{m['holidays_version']})")
            return 1
        print(f"OK: {sha}")
        return 0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"calendars_{VERSION}.npz").write_bytes(data)
    man.write_text(json.dumps(dict(version=VERSION, sha256=sha, holidays_version=ver, start=str(START), end=str(END),
                                   types=TYPES, rules=RULES, calendars=meta), indent=1) + "\n", encoding="utf-8", newline="\n")
    (OUT / f"calendars_{VERSION}_report.txt").write_text(report, encoding="utf-8", newline="\n")
    print(report.split("\n== ")[0])
    print(f"wrote {OUT} ({len(data):,} bytes, sha256 {sha})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
