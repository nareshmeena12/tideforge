"""Downloads (stdlib urllib; run on a Perlmutter login / transfer node or the owner's machine, never on compute nodes).

- URL lists: fetch.urls, or fetch.url_template expanded over fetch.vars (lists, {range: [a, b]}, {months: [start, end],
  fmt}, or "index" = the selected series). Template values are URL-quoted.
- Files go to <raw>/<slug>/files/<name>; a download is written to <name>.part and renamed when complete, so a rerun skips
  finished files and redoes partial ones. Every file gets a row in <raw>/<slug>/fetch_log.jsonl (url, file, bytes, sha256,
  status, time). A 404 is logged and skipped (a station-year that does not exist), other errors are retried, then raised.
- Auth: fetch.auth = {env: NAME, param: p}, {env: NAME, header: h} or {env: NAME, basic: true} (HTTP basic auth with the
  value as the user name and an empty password, e.g. USDA MyMarketNews); the value is read from the environment at request
  time and never written to the log, a file name or an error message (URLs are logged with the parameter redacted).
- probe(): the first bytes of the index and of the first file only, for checking a source's layout before a full fetch.
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import itertools
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UA = "tideforge-research/0.1 (time-series dev set; contact: repository owner)"


def expand_vars(spec: dict, series: list[str] | None = None) -> list[dict]:
    """fetch.vars -> list of {name: value} combinations (cartesian product, in sorted name order)."""
    names, values = [], []
    for k in sorted(spec or {}):
        v = spec[k]
        if v == "index":
            if series is None:
                raise ValueError(f"var {k} = index needs the selected series")
            vals = list(series)
        elif isinstance(v, dict) and "range" in v:
            a, b = v["range"]
            vals = list(range(int(a), int(b) + 1))
        elif isinstance(v, dict) and "months" in v:
            a, b = v["months"]
            vals = [d.strftime(v.get("fmt", "%Y%m")) for d in
                    __import__("pandas").period_range(a, b, freq="M").to_timestamp()]
        elif isinstance(v, list):
            vals = list(v)
        else:
            vals = [v]
        names.append(k)
        values.append(vals)
    return [dict(zip(names, combo)) for combo in itertools.product(*values)]


def render(template: str, vars_: dict) -> str:
    return template.format(**{k: urllib.parse.quote(str(v), safe="") for k, v in vars_.items()})


def file_list(fetch: dict, series: list[str] | None = None) -> list[tuple[str, dict]]:
    """[(url, vars)] for a fetch / index spec."""
    if fetch.get("urls"):
        return [(u, {}) for u in fetch["urls"]]
    if fetch.get("url"):
        return [(fetch["url"], {})]
    return [(render(fetch["url_template"], v), v) for v in expand_vars(fetch.get("vars", {}), series)]


def local_name(url: str) -> str:
    """A stable, safe file name for a URL (path tail + hash of the full URL, so query strings do not collide)."""
    p = urllib.parse.urlparse(url)
    tail = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(p.path).name or "index")[-80:]
    return f"{hashlib.sha1(url.encode()).hexdigest()[:10]}_{tail}"


def _with_auth(url: str, headers: dict, auth: dict | None) -> tuple[str, dict]:
    if not auth:
        return url, headers
    val = os.environ.get(auth["env"])
    if not val:
        raise RuntimeError(f"environment variable {auth['env']} is not set")
    if auth.get("basic"):
        import base64
        return url, dict(headers, Authorization="Basic " + base64.b64encode(f"{val}:".encode()).decode())
    if "header" in auth:
        return url, dict(headers, **{auth["header"]: val})
    p = urllib.parse.urlparse(url)
    q = urllib.parse.parse_qsl(p.query) + [(auth["param"], val)]
    return urllib.parse.urlunparse(p._replace(query=urllib.parse.urlencode(q))), headers


def redact(url: str, auth: dict | None) -> str:
    if not auth or "param" not in auth:
        return url
    return re.sub(rf"([?&]{re.escape(auth['param'])}=)[^&]*", r"\1<redacted>", url)


def _open(url: str, headers: dict, auth: dict | None, timeout: float, extra: dict | None = None):
    full, h = _with_auth(url, dict({"User-Agent": UA}, **(headers or {}), **(extra or {})), auth)
    return urllib.request.urlopen(urllib.request.Request(full, headers=h), timeout=timeout)


def download(url: str, dest: Path, headers: dict | None = None, auth: dict | None = None, retries: int = 3,
             timeout: float = 120.0, log: Path | None = None) -> dict:
    """url -> dest (skipped if dest exists). Returns the log row."""
    if dest.exists():
        return {"url": redact(url, auth), "file": dest.name, "status": "exists"}
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    err = None
    for attempt in range(retries):
        try:
            h = hashlib.sha256()
            n = 0
            with _open(url, headers, auth, timeout) as r, open(part, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
                    h.update(b)
                    n += len(b)
            part.replace(dest)
            row = {"url": redact(url, auth), "file": dest.name, "bytes": n, "sha256": h.hexdigest(), "status": "ok"}
            break
        except urllib.error.HTTPError as e:
            if e.code == 404:
                row = {"url": redact(url, auth), "file": dest.name, "status": "404"}
                break
            err = f"HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            err = type(e).__name__
        time.sleep(2 ** attempt)
    else:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"download failed after {retries} tries ({err}): {redact(url, auth)}")
    row["time"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if log:
        with open(log, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    return row


def fetch_all(urls: list[tuple[str, dict]], folder: Path, fetch: dict, log: Path, max_files: int | None = None,
              say=print) -> list[tuple[Path, dict]]:
    """Download every url (rate-limited); returns [(path, vars)] of the files that exist afterwards."""
    out = []
    for i, (u, v) in enumerate(urls[:max_files] if max_files else urls):
        dest = folder / local_name(u)
        row = download(u, dest, fetch.get("headers"), fetch.get("auth"), retries=int(fetch.get("retries", 3)),
                       timeout=float(fetch.get("timeout_s", 120)), log=log)
        if row["status"] != "exists":
            time.sleep(float(fetch.get("rate_limit_s", 0.2)))
        if dest.exists():
            out.append((dest, v))
        if i % 50 == 0 or row["status"] not in ("ok", "exists"):
            say(f"  [{i + 1}/{len(urls)}] {row['status']:6s} {row['url'][:140]}")
    return out


def probe(url: str, fetch: dict, n: int = 65536) -> str:
    """Status, headers and the first lines of a URL (first n bytes only; gzip is decompressed as far as it goes)."""
    auth = fetch.get("auth")
    try:
        with _open(url, fetch.get("headers"), auth, 60, {"Range": f"bytes=0-{n - 1}"}) as r:
            head = r.read(n)
            meta = f"HTTP {r.status} {r.headers.get('Content-Type')} length {r.headers.get('Content-Length')}"
    except urllib.error.HTTPError as e:
        return f"HTTP {e.code} {redact(url, auth)}"
    except Exception as e:      # noqa: BLE001
        return f"{type(e).__name__}: {e} {redact(url, auth)}"
    if head[:2] == b"\x1f\x8b":
        try:
            head = gzip.decompress(head)
        except (EOFError, OSError):
            d = __import__("zlib").decompressobj(16 + 15)
            head = d.decompress(head)
    if head[:2] == b"PK":
        return f"{meta}\n  (zip archive; members are listed after the full download)"
    text = head.decode("utf-8", "replace")
    if "<ListBucketResult" in text:                                   # S3 listing: show the keys
        keys = re.findall(r"<Key>([^<]+)</Key>", text)
        return meta + f"\n  S3 listing, {len(keys)} keys in the first {n} bytes:\n" + "\n".join("  | " + k for k in keys[:25])
    lines = text.splitlines()
    body = [x for x in lines if not x.startswith("#")]                 # skip comment headers (USGS RDB)
    shown = (body or lines)[:8]
    skipped = f"  ({len(lines) - len(body)} comment lines skipped)\n" if body and len(body) < len(lines) else ""
    return meta + "\n" + skipped + "\n".join("  | " + x[:400] for x in shown)
