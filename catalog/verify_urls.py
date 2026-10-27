"""Check every catalog URL: does it resolve, and does the page mention the dataset?

Run on your own machine (stdlib only, ~1-2 min for ~380 URLs):
    python3 verify_urls.py dataset_catalog_v2.csv
Writes:
    url_check.csv                 one row per catalog row: status class, HTTP code, final URL, page title, keyword hits
    dataset_catalog_v3_checked.csv catalog with url_status / url_http / url_title added
Status classes:
    ok            HTTP 2xx and the page text contains a distinctive word from the dataset name
    ok_weak       HTTP 2xx but no distinctive word found (landing page, JS-rendered page, wrong page) -> look manually
    blocked       401/403/429/503 (bot protection, login, rate limit): the site exists, contents not confirmed
    dead          404/410 or DNS failure -> fix the URL
    error         timeout / TLS / other
"""
import csv, re, sys, html, ssl, socket, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

SRC = sys.argv[1] if len(sys.argv) > 1 else "dataset_catalog_v2.csv"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0 Safari/537.36")
GENERIC = set("""data dataset datasets open public time series daily hourly monthly weekly index indices national global
the and for from with system systems traffic sales price prices energy load records record traces trace counts
database collection network station stations portal statistics stats survey archive level levels""".split())

def keywords(name):
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]{3,}", name.lower())
    return [t for t in toks if t not in GENERIC][:6]

def fetch(row):
    url = row["url"].strip()
    out = dict(id=row["id"], name=row["name"], url=url, status="error", http="", final_url="", title="", kw_hits="", err="")
    if not url.startswith("http"):
        out["err"] = "no url"; return out
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
                                               "Accept-Language": "en"})
    try:
        with urllib.request.urlopen(req, timeout=25, context=ssl.create_default_context()) as r:
            out["http"] = r.status; out["final_url"] = r.geturl()
            body = r.read(400_000).decode("utf-8", "ignore")
    except urllib.error.HTTPError as e:
        out["http"] = e.code
        out["status"] = "dead" if e.code in (404, 410) else "blocked" if e.code in (401, 403, 405, 406, 429, 451, 503) else "error"
        out["err"] = str(e.reason)[:80]; return out
    except (urllib.error.URLError, socket.timeout, ssl.SSLError, ConnectionError, OSError) as e:
        msg = str(getattr(e, "reason", e))
        out["status"] = "dead" if ("Name or service" in msg or "nodename" in msg or "getaddrinfo" in msg) else "error"
        out["err"] = msg[:80]; return out
    m = re.search(r"<title[^>]*>(.*?)</title>", body, re.I | re.S)
    out["title"] = html.unescape(m.group(1)).strip()[:120] if m else ""
    text = html.unescape(re.sub(r"<[^>]+>", " ", body)).lower()
    kws = keywords(row["name"])
    hits = [k for k in kws if k in text]
    out["kw_hits"] = f"{len(hits)}/{len(kws)}:" + ",".join(hits)
    out["status"] = "ok" if hits else "ok_weak"
    return out

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
with ThreadPoolExecutor(max_workers=24) as ex:
    res = list(ex.map(fetch, rows))

cols = ["id", "name", "url", "status", "http", "final_url", "title", "kw_hits", "err"]
with open("url_check.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(res)
byid = {r["id"]: r for r in res}
for row in rows:
    r = byid[row["id"]]; row["url_status"] = r["status"]; row["url_http"] = r["http"]; row["url_title"] = r["title"]
with open("dataset_catalog_v3_checked.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

from collections import Counter
c = Counter(r["status"] for r in res)
print("checked:", len(res), dict(c))
for s in ("dead", "error", "ok_weak"):
    bad = [r for r in res if r["status"] == s]
    if bad:
        print(f"\n{s} ({len(bad)}):")
        for r in bad[:60]: print(f"  {r['id']} {r['name'][:45]:45s} {r['http']} {r['err'] or r['title'][:50]}")
