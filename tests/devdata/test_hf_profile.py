"""Hugging Face catalog profiler: unit tests on recorded responses, no network.

Fixtures in tests/devdata/fixtures/hf were recorded on 2026-11-12 (anonymous calls): Hub API repo info, viewer /is-valid,
/size, /info, /statistics, /parquet, one RMISC folder meta.json, and two parquet FOOTERS (metadata only, no rows).
A fake session stands in for `requests`; any URL it does not know fails the test.
"""
import argparse
import datetime as dt
import io
import json
import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tideforge.devdata import hf_derive as D
from tideforge.devdata import hf_profile as P
from tideforge.devdata.hf_guard import BudgetError, Fetcher, FetchError, GuardError, classify, retry_after_seconds

FX = Path(__file__).parent / "fixtures" / "hf"
WFP = "electricsheepafrica/wfp-nigeria-food-prices"
HUB, VIEWER = "https://huggingface.co", "https://datasets-server.huggingface.co"


def fx_bytes(name):
    return (FX / name).read_bytes()


def fx_json(name):
    return json.loads(fx_bytes(name).decode("utf-8"))


class FakeRaw:
    def __init__(self, n):
        self.n = n

    def tell(self):
        return self.n


class FakeResp:
    def __init__(self, status, body=b"", headers=None):
        self.status_code, self._body = status, body
        self.headers = {"Content-Length": str(len(body)), **(headers or {})}
        self.raw = FakeRaw(len(body))
        self.closed = False

    def iter_content(self, n):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]

    def close(self):
        self.closed = True


class FakeSession:
    """routes: {url: response | [responses in order] | callable(headers) -> response}; a file is {"file": bytes}."""

    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, headers=None, stream=False, allow_redirects=True, timeout=None):
        headers = headers or {}
        self.calls.append((url, dict(headers)))
        assert stream and not allow_redirects
        if url not in self.routes:
            raise AssertionError(f"unexpected network call: {url}")
        r = self.routes[url]
        if isinstance(r, list):
            r = r.pop(0) if len(r) > 1 else r[0]
        if callable(r):
            r = r(headers)
        if isinstance(r, dict) and "file" in r:  # a remote file that honours suffix ranges
            data, rng = r["file"], headers.get("Range")
            if rng is None:
                return FakeResp(200, data)
            n = int(rng.split("=-")[1])
            part = data[-n:]
            return FakeResp(206, part, {"Content-Range": f"bytes {len(data) - len(part)}-{len(data) - 1}/{len(data)}"})
        return r


def J(obj, status=200, headers=None):
    return FakeResp(status, json.dumps(obj).encode(), headers)


def make_fx(tmp_path, routes, **kw):
    sleeps = []
    s = FakeSession(routes)
    f = Fetcher(tmp_path / "cache", session=s, sleep=sleeps.append, **kw)
    return f, s, sleeps


def padded(footer: bytes, size: int) -> bytes:
    """A stand-in remote parquet file: zeros where the row data would be, then the recorded footer."""
    return b"\0" * (size - len(footer)) + footer


OPTS = argparse.Namespace(max_footers=50, footer_tail=16384, max_expand=300, max_stats_calls=400)


# ---- the fetch guard ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("url,rng,kind", [
    (f"{HUB}/api/datasets/{WFP}?blobs=true", None, "hub_api"),
    (f"{HUB}/api/datasets/{WFP}/tree/main/data?recursive=true", None, "hub_api"),
    (f"{VIEWER}/size?dataset={WFP}", None, "viewer"),
    (f"{VIEWER}/statistics?dataset={WFP}&config=default&split=train", None, "viewer"),
    (f"{HUB}/datasets/{WFP}/resolve/main/train.parquet", "bytes=-16384", "footer"),
    (f"{HUB}/datasets/{WFP}/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet", "bytes=-8", "footer"),
    (f"{HUB}/datasets/{WFP}/resolve/main/README.md", None, "text"),
    (f"{HUB}/datasets/nju-zhangsq/RMISC/resolve/main/SP500/meta.json", None, "text"),
    (f"{HUB}/datasets/Salesforce/lotsa_data/resolve/main/HZMETRO/dataset_info.json", None, "text"),
])
def test_guard_allows_the_four_kinds(url, rng, kind):
    assert classify(url, rng) == kind


@pytest.mark.parametrize("url,rng,size", [
    (f"{HUB}/datasets/{WFP}/resolve/main/train.parquet", None, None),  # a data file, whole
    (f"{HUB}/datasets/{WFP}/resolve/main/train.parquet", "bytes=0-65535", None),  # not a footer: the file's start
    (f"{HUB}/datasets/{WFP}/resolve/main/train.parquet", "bytes=1000-2000", None),  # not a footer: a row group
    (f"{HUB}/datasets/{WFP}/resolve/main/train.parquet", "bytes=421951-", None),  # open-ended range
    (f"{HUB}/datasets/{WFP}/resolve/main/train.parquet", f"bytes=-{9 * 2 ** 20}", None),  # above the footer cap
    (f"{HUB}/datasets/{WFP}/resolve/main/train.parquet", "bytes=-487487", 487487),  # suffix = the whole file
    (f"{HUB}/datasets/{WFP}/resolve/main/wfp_markets_nga.csv", None, 5438),  # csv data file
    (f"{HUB}/datasets/{WFP}/resolve/main/wfp_markets_nga.csv", "bytes=-100", None),  # range on a non-parquet file
    (f"{HUB}/datasets/x/y/resolve/main/data.zip", None, None),
    (f"{HUB}/datasets/x/y/resolve/main/a/data-00000-of-00001.arrow", None, None),
    (f"{HUB}/datasets/x/y/resolve/main/arr.npy", None, None),
    (f"{HUB}/datasets/x/y/resolve/main/metadata.json", None, 5_000_000),  # "metadata" but 5 MB
    (f"{VIEWER}/rows?dataset={WFP}&config=default&split=train", None, None),  # rows are data
    (f"{VIEWER}/first-rows?dataset={WFP}", None, None),
    (f"http://huggingface.co/api/datasets/{WFP}", None, None),
    ("https://example.com/api/datasets/a/b", None, None),
    (f"{HUB}/api/models/a/b", None, None),
])
def test_guard_refuses_everything_else(url, rng, size):
    with pytest.raises(GuardError):
        classify(url, rng, size)


def test_refused_request_never_reaches_the_network(tmp_path):
    f, s, _ = make_fx(tmp_path, {})
    with pytest.raises(GuardError):
        f.get(f"{HUB}/datasets/{WFP}/resolve/main/train.parquet")
    with pytest.raises(GuardError):
        f.get(f"{HUB}/datasets/{WFP}/resolve/main/train.parquet", "bytes=0-1023")
    assert s.calls == [] and f.bytes_total == 0


def test_server_ignoring_the_range_is_refused(tmp_path):
    url = f"{HUB}/datasets/{WFP}/resolve/main/train.parquet"
    f, s, _ = make_fx(tmp_path, {url: FakeResp(200, b"x" * 500_000)})
    with pytest.raises(GuardError, match="ignored the range"):
        f.get(url, "bytes=-16384", 487487)
    assert f.bytes_total < 1000  # headers only: the body was not read


def test_text_file_over_1mb_is_refused(tmp_path):
    url = f"{HUB}/datasets/a/b/resolve/main/meta.json"
    f, _, _ = make_fx(tmp_path, {url: FakeResp(200, b"x" * 1_200_000)})
    with pytest.raises(GuardError, match="limit"):
        f.get(url)


def test_429_waits_for_retry_after(tmp_path):
    url = f"{VIEWER}/size?dataset={WFP}"
    f, s, sleeps = make_fx(tmp_path, {url: [FakeResp(429, b"{}", {"Retry-After": "7"}), FakeResp(429, b"{}"),
                                            J(fx_json("size_wfp.json"))]})
    r = f.get(url)
    assert r.status == 200 and sleeps == [7.0, 4.0] and len(s.calls) == 3 and f.n_429 == 2


def test_retry_after_forms():
    assert retry_after_seconds("12", 0) == 12.0
    assert retry_after_seconds(None, 0) == 2.0 and retry_after_seconds(None, 2) == 8.0
    assert retry_after_seconds("100000", 0) == 300.0  # capped
    now = dt.datetime(2026, 11, 12, 10, 0, 0, tzinfo=dt.timezone.utc).timestamp()
    assert retry_after_seconds("Thu, 12 Nov 2026 10:00:30 GMT", 0, now=now) == 30.0


def test_cache_makes_no_repeat_calls_and_refresh_redoes(tmp_path):
    url = f"{VIEWER}/size?dataset={WFP}"
    routes = {url: J(fx_json("size_wfp.json"))}
    f, s, _ = make_fx(tmp_path, routes, token="hf_SECRET_TOKEN")
    a = f.get(url)
    b = f.get(url)
    assert len(s.calls) == 1 and not a.from_cache and b.from_cache and a.body == b.body and f.n_cache_hits == 1
    f2, s2, _ = make_fx(tmp_path, routes)  # a new run: same cache directory
    assert f2.get(url).from_cache and s2.calls == [] and f2.bytes_total == 0
    f3, s3, _ = make_fx(tmp_path, routes, refresh=True)
    assert not f3.get(url).from_cache and len(s3.calls) == 1
    assert f3.get(url).from_cache and len(s3.calls) == 1  # refreshed once per run
    for p in (tmp_path / "cache").rglob("*"):  # the token is never written
        if p.is_file():
            assert b"hf_SECRET_TOKEN" not in p.read_bytes()


def test_cache_keeps_404_and_offline_misses_fail(tmp_path):
    url = f"{HUB}/api/datasets/nobody/none?blobs=true"
    f, s, _ = make_fx(tmp_path, {url: J({"error": "Repository not found"}, 404)})
    assert f.get(url).status == 404 and f.get(url).status == 404 and len(s.calls) == 1
    f2, _, _ = make_fx(tmp_path, {}, offline=True)
    assert f2.get(url).status == 404
    with pytest.raises(FetchError):
        f2.get(f"{VIEWER}/size?dataset=a/b")


def test_range_is_part_of_the_cache_key(tmp_path):
    url = f"{HUB}/datasets/{WFP}/resolve/main/train.parquet"
    f, s, _ = make_fx(tmp_path, {url: {"file": padded(fx_bytes("wfp_footer.bin"), 487487)}})
    assert len(f.get(url, "bytes=-8", 487487).body) == 8
    assert len(f.get(url, "bytes=-16384", 487487).body) == 16384
    assert len(s.calls) == 2


def test_token_goes_to_hf_hosts_only_and_redirects_keep_the_range(tmp_path):
    url = f"{HUB}/datasets/{WFP}/resolve/main/train.parquet"
    cdn = "https://us.aws.cdn.hf.co/xet-bridge-us/abc?sig=1"
    f, s, _ = make_fx(tmp_path, {url: FakeResp(302, b"", {"Location": cdn}),
                                 cdn: {"file": padded(fx_bytes("wfp_footer.bin"), 487487)}}, token="hf_SECRET_TOKEN")
    r = f.get(url, "bytes=-16384", 487487)
    assert r.status == 206 and len(r.body) == 16384
    assert s.calls[0][1]["Authorization"] == "Bearer hf_SECRET_TOKEN"
    assert "Authorization" not in s.calls[1][1] and s.calls[1][1]["Range"] == "bytes=-16384"
    assert "SECRET" not in repr(f)


def test_api_redirect_must_stay_on_metadata_endpoints(tmp_path):
    url = f"{HUB}/api/datasets/old/name?blobs=true"
    f, _, _ = make_fx(tmp_path, {url: FakeResp(307, b"", {"Location": "/datasets/old/name/resolve/main/data.csv"})})
    with pytest.raises(GuardError):
        f.get(url)
    new = f"{HUB}/api/datasets/new/name?blobs=true"
    f, _, _ = make_fx(tmp_path / "b", {url: FakeResp(307, b"", {"Location": "/api/datasets/new/name?blobs=true"}),
                                       new: J({"id": "new/name", "siblings": []})})
    assert f.get(url).json()["id"] == "new/name"  # a renamed repo


def test_bytes_are_counted_and_budget_aborts(tmp_path):
    u1, u2 = f"{VIEWER}/size?dataset=a/b", f"{VIEWER}/info?dataset=a/b"
    body = b"x" * 5000
    f, _, _ = make_fx(tmp_path, {u1: FakeResp(200, body), u2: FakeResp(200, body)}, budget=7000)
    f.get(u1)
    assert 5000 <= f.bytes_total < 5200 and f.summary()["by_kind"]["viewer"]["requests"] == 1
    with pytest.raises(BudgetError):
        f.get(u2)
    with pytest.raises(BudgetError):  # and nothing more is sent afterwards
        f.get(f"{VIEWER}/parquet?dataset=a/b")


def test_fetcher_with_real_requests_on_localhost(tmp_path):
    """The real `requests` objects (headers, raw.tell, streaming, redirects) against a local server: no outside network.
    `_once` is called directly because `get` allows the two Hugging Face hosts only."""
    import gzip
    import http.server
    import threading

    pytest.importorskip("requests")
    footer = fx_bytes("wfp_footer.bin")
    blob = padded(footer, 100_000)
    payload = json.dumps(fx_json("size_wfp.json")).encode()

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/moved.parquet":
                self.send_response(302)
                self.send_header("Location", "/x.parquet")
                self.send_header("Content-Length", "0")
                self.end_headers()
            elif self.path == "/x.parquet":
                n = int(self.headers["Range"].split("=-")[1])
                part = blob[-n:]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {len(blob) - n}-{len(blob) - 1}/{len(blob)}")
                self.send_header("Content-Length", str(len(part)))
                self.end_headers()
                self.wfile.write(part)
            elif self.path == "/ignores-range.parquet":
                self.send_response(200)
                self.send_header("Content-Length", str(len(blob)))
                self.end_headers()
                self.wfile.write(blob)
            else:
                body = gzip.compress(payload)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Encoding", "gzip")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        f = Fetcher(tmp_path / "cache", token="hf_SECRET_TOKEN")
        r = f._once(base + "/size", None, "viewer")
        assert r.status == 200 and r.json()["size"]["dataset"]["num_rows"] == 54883
        assert 0 < f.bytes_total < len(payload) + 400  # wire bytes: the gzip body + headers, not the decoded size
        before = f.bytes_total
        r = f._once(base + "/x.parquet", "bytes=-16384", "footer")
        assert r.status == 206 and len(r.body) == 16384 and r.body.endswith(b"PAR1")
        assert r.headers["content-range"].endswith("/100000") and 16384 <= f.bytes_total - before < 16384 + 400
        assert P.parse_footer(r.body[-len(footer):])["rows"] == 54883
        with pytest.raises(GuardError, match="non-https"):  # a redirect must stay on https
            f._once(base + "/moved.parquet", "bytes=-16384", "footer")
        before = f.bytes_total
        with pytest.raises(GuardError, match="ignored the range"):
            f._once(base + "/ignores-range.parquet", "bytes=-16384", "footer")
        assert f.bytes_total - before < 1000  # the 100 kB body was not read
    finally:
        srv.shutdown()
        srv.server_close()


# ---- footers -----------------------------------------------------------------------------------------------------------
def test_date_range_from_recorded_footer_statistics():
    ft = P.parse_footer(fx_bytes("wfp_footer.bin"))
    assert ft["rows"] == 54883 and len(ft["leaves"]) == 16
    cols = D.cols_from_leaves(ft["leaves"])
    assert [c.kind for c in cols if c.name == "date"] == ["time"]
    t = {"cols": cols, "rows": ft["rows"], "stats": ft["stats"], "complete": True, "stats_src": "parquet footer"}
    dr = D.date_range([t], [D.assign_roles(cols)])
    assert dr["first"] == dt.datetime(2002, 1, 15) and dr["last"] == dt.datetime(2025, 11, 15)
    assert dr["conf"] == D.EXACT and 23.8 < dr["span_years"] < 23.9
    assert ft["stats"]["price"]["nulls"] == 0 and ft["stats"]["price"]["num_values"] == 54883
    t2 = dict(t, complete=False)  # sampled footers: the range is a lower bound -> estimated
    assert D.date_range([t2], [D.assign_roles(cols)])["conf"] == D.EST


def test_recorded_rmisc_footer_null_counts():
    ft = P.parse_footer(fx_bytes("rmisc_sp500_footer.bin"))
    assert ft["rows"] == 603027 and ft["stats"]["Volume"]["nulls"] == 1258


def _parquet_footer(table, **kw) -> tuple[bytes, bytes]:
    buf = io.BytesIO()
    pq.write_table(table, buf, **kw)
    data = buf.getvalue()
    n = struct.unpack("<I", data[-8:-4])[0] + 8
    return data, data[-n:]


def test_nested_points_from_leaf_value_counts():
    lens = [5, 7, 3]
    tbl = pa.table({"item_id": ["a", "b", "c"],
                    "start": pa.array([dt.datetime(2020, 1, 1)] * 3, pa.timestamp("s")),
                    "freq": ["15T"] * 3,
                    "target": pa.array([[1.0] * n for n in lens], pa.list_(pa.float32()))})
    _, footer = _parquet_footer(tbl)
    ft = P.parse_footer(footer)
    cols = D.cols_from_leaves(ft["leaves"])
    assert {c.name: c.kind for c in cols} == {"item_id": "str", "start": "time", "freq": "str", "target": "list_num"}
    assert ft["stats"]["target"]["num_values"] == sum(lens)
    assert ft["stats"]["freq"]["min"] == ft["stats"]["freq"]["max"] == "15T"


def test_footer_reads_only_the_tail(tmp_path):
    url = f"{HUB}/datasets/a/b/resolve/main/x.parquet"
    footer = fx_bytes("wfp_footer.bin")
    # large file: one suffix read of 16 KiB holds the 8 KB footer
    f, s, _ = make_fx(tmp_path / "1", {url: {"file": padded(footer, 487487)}})
    buf, err = P.footer_bytes(f, url, 487487)
    assert buf == footer and err == "" and [c[1]["Range"] for c in s.calls] == ["bytes=-16384"]
    # footer larger than the first read: a second, exact suffix read
    f, s, _ = make_fx(tmp_path / "2", {url: {"file": padded(footer, 487487)}})
    buf, _ = P.footer_bytes(f, url, 487487, tail=1024)
    assert buf == footer and [c[1]["Range"] for c in s.calls] == ["bytes=-1024", f"bytes=-{len(footer)}"]
    # file smaller than the first read: the 8-byte trailer, then exactly the footer (never the whole file)
    small = padded(footer, 12000)
    f, s, _ = make_fx(tmp_path / "3", {url: {"file": small}})
    buf, _ = P.footer_bytes(f, url, 12000)
    assert buf == footer and [c[1]["Range"] for c in s.calls] == ["bytes=-8", f"bytes=-{len(footer)}"]
    assert f.bytes_total < 12000
    # not parquet
    f, _, _ = make_fx(tmp_path / "4", {url: {"file": b"z" * 50000}})
    assert P.footer_bytes(f, url, 50000)[0] is None


# ---- layout rules --------------------------------------------------------------------------------------------------------
def C(name, kind, **kw):
    return D.Col(name, kind, **kw)


def layout_of(cols):
    return D.decide_layout(cols, D.assign_roles(cols))


def test_layout_nested_from_recorded_chronos_features():
    cols = D.cols_from_features(fx_json("vinfo_chronos.json")["dataset_info"]["features"])
    assert {c.name: c.kind for c in cols} == {"id": "str", "timestamp": "list_time", "target": "list_num"}
    lay, rule, conf = layout_of(cols)
    assert lay == "nested" and rule.startswith("R1") and conf == D.EXACT
    roles = D.assign_roles(cols)
    assert roles["time_col"] == "timestamp" and roles["target_cols"] == ["target"] and roles["id_cols"] == ["id"]


def test_layout_nested_2d_target_keeps_the_variate_count():
    cols = D.cols_from_features(fx_json("lotsa_hzmetro_dataset_info.json")["features"])
    tgt = next(c for c in cols if c.name == "target")
    assert tgt.kind == "list_num" and tgt.depth == 2 and tgt.width == 2
    assert D.assign_roles(cols)["freq_cols"] == ["freq"] and layout_of(cols)[0] == "nested"


def test_layout_long_from_recorded_wfp_features():
    cols = D.cols_from_features(fx_json("vinfo_wfp.json")["dataset_info"]["default"]["features"])
    roles = D.assign_roles(cols)
    lay, rule, conf = D.decide_layout(cols, roles)
    assert lay == "long" and rule.startswith("R4") and conf == D.EST
    assert roles["time_col"] == "date" and roles["id_cols"] == ["market", "commodity"]  # market_id = market
    assert roles["value_cols"] == ["price", "usdprice"]  # ids and latitude / longitude are not values
    assert "latitude" in roles["static_cols"] and "currency" in roles["static_cols"]


def test_layout_wide_events_unknown():
    wide = [C("timestamp", "time")] + [C(f"MT_{i:03d}", "num") for i in range(370)]
    lay, rule, _ = layout_of(wide)
    assert lay == "wide" and rule.startswith("R5")
    assert layout_of([C("a", "num"), C("b", "num")])[1].startswith("R6")
    ev = [C("created_at", "time"), C("title", "str"), C("body", "str")]
    assert layout_of(ev)[0] == "events" and layout_of(ev)[1].startswith("R2")
    trades = [C("timestamp", "time"), C("market_id", "str"), C("trade_id", "str"), C("price", "num"), C("size", "num")]
    assert layout_of(trades)[0] == "events" and layout_of(trades)[1].startswith("R3")
    assert layout_of([C("text", "str")])[0] == "unknown" and layout_of([])[0] == "unknown"
    m5 = [C("date", "time"), C("item_id", "str"), C("sales", "num"), C("event_name_1", "str"), C("snap", "bool")]
    assert layout_of(m5)[0] == "long"  # a covariate that mentions an event does not make an event table


def test_time_column_by_name_when_not_typed():
    roles = D.assign_roles([C("Date", "str"), C("Open", "num"), C("Close", "num")])
    assert roles["time_col"] == "Date" and "not a timestamp type" in roles["time_rule"]
    assert roles["value_cols"] == ["Open", "Close"] and roles["target_cols"] == ["Open", "Close"]


# ---- n_series / n_points -----------------------------------------------------------------------------------------------
def counts(cols, rows, stats=None, **t):
    table = {"cols": cols, "rows": rows, "rows_conf": D.EXACT, "stats": stats or {}, "complete": True, **t}
    roles = D.assign_roles(cols, table["stats"])
    return D.table_counts(table, roles, D.decide_layout(cols, roles)[0])


def test_counts_nested_from_viewer_list_statistics():
    cols = D.cols_from_features(fx_json("vinfo_chronos.json")["dataset_info"]["features"])
    table = {"cols": cols, "rows": 8, "rows_conf": D.EXACT, "stats": {}, "complete": True}
    P._merge_viewer_stats(table, [(8, P._viewer_col_stats(fx_json("stats_chronos.json")))])
    roles = D.assign_roles(cols, table["stats"])
    c = D.table_counts(table, roles, "nested")
    assert c["n_series"] == 8 and c["n_series_conf"] == D.EXACT
    assert c["n_points"] == 8 * 7588 and c["n_points_conf"] == D.EST and "mean list length" in c["n_points_src"]
    assert (c["len_min"], c["len_median"], c["len_max"]) == (7588, 7588.0, 7588) and c["len_conf"] == D.EXACT
    assert table["stats"]["id"]["n_unique"] == 8


def test_counts_nested_from_footer_leaf_counts_are_exact():
    cols = [C("item_id", "str"), C("start", "time"), C("target", "list_num", depth=1),
            C("past_feat_dynamic_real", "list_num", depth=1)]
    c = counts(cols, 10, {"target": {"num_values": 1000}, "past_feat_dynamic_real": {"num_values": 3000}})
    assert c["n_series"] == 10 and c["n_points"] == 4000 and c["n_points_target"] == 1000
    assert c["n_points_conf"] == D.EXACT and c["len_median"] == 100.0 and c["len_conf"] == D.EST
    c = counts(cols, 20, {"target": {"num_values": 1000}, "past_feat_dynamic_real": {"num_values": 3000}},
               complete=False, scale=2.0, rows_conf=D.EST)  # half of the footers read: scaled, estimated
    assert c["n_points"] == 8000 and c["n_points_conf"] == D.EST


def test_counts_long():
    cols = [C("date", "time"), C("store", "str"), C("item", "str"), C("sales", "num"), C("price", "num")]
    c = counts(cols, 1000, {"store": {"n_unique": 10}})
    assert c["n_points"] == 2000 and c["n_points_target"] == 1000 and c["n_series"] is None  # item: no distinct count
    c = counts(cols, 1000, {"store": {"n_unique": 10}, "item": {"n_unique": 20}})
    assert c["n_series"] == 200 and c["n_series_conf"] == D.EST and "product" in c["n_series_src"]
    assert c["len_median"] == 5.0 and c["len_conf"] == D.EST
    c = counts(cols, 1000, {"store": {"n_unique": 100}, "item": {"n_unique": 100}})
    assert c["n_series"] == 1000  # the product is capped at the row count
    one = [C("ds", "time"), C("unique_id", "str"), C("y", "num")]
    c = counts(one, 5000, {"unique_id": {"n_unique": 50}})
    assert (c["n_series"], c["n_series_conf"], c["n_points"], c["n_points_conf"]) == (50, D.EXACT, 5000, D.EXACT)
    c = counts([C("ds", "time"), C("station_id", "num"), C("y", "num")], 5000, {"station_id": {"min": 1, "max": 25}})
    assert c["n_series"] == 25 and c["n_series_conf"] == D.EST and "max-min+1" in c["n_series_src"]


def test_counts_wide():
    cols = [C("timestamp", "time")] + [C(f"MT_{i}", "num") for i in range(370)]
    c = counts(cols, 140256)
    assert c["n_series"] == 370 and c["n_points"] == 370 * 140256 == c["n_points_target"]
    assert c["n_points_conf"] == D.EXACT and c["len_median"] == 140256


# ---- frequency, bands, slices, ingest ------------------------------------------------------------------------------------
@pytest.mark.parametrize("name,sec", [
    ("electricity_15min", 900), ("taxi_30min", 1800), ("solar_1h", 3600), ("monash_m1_monthly", 2629800),
    ("weatherbench_daily", 86400), ("ds-10-T", 60), ("ds-1030-5T", 300), ("candles_1m", 60), ("ohlc_1M", 2629800),
    ("Energy-Load-Profiles/15min/Commercial/DGS/DGS-15min", 900), ("Hourly/Commercial/BDG-2", 3600), ("hour", 3600),
    ("min", 60), ("uber_tlc_hourly", 3600), ("BEIJING_SUBWAY_30MIN", 1800), ("x_10s", 10), ("data_1d", 86400),
    ("M_DENSE", None), ("exchange_rate", None), ("largest_2017", None), ("bdg-2_bear", None), ("5G-Network", None),
    ("hourly_and_daily", None), ("m4_weekly", 604800), ("us500-ticks", None), ("LOOP_SEATTLE", None),
])
def test_freq_from_name(name, sec):
    assert D.freq_from_name(name) == sec


def test_freq_alias_implied_and_bands():
    assert D.parse_freq_alias("15T") == 900 and D.parse_freq_alias("H") == 3600 and D.parse_freq_alias("W-SUN") == 604800
    assert D.parse_freq_alias("5min") == 300 and D.parse_freq_alias("MS") == 2629800 and D.parse_freq_alias("??") is None
    assert D.freq_implied(86400.0 * 364, 365)[0] == 86400.0
    assert D.freq_implied(3600.0 * 999, 1000)[0] == 3600.0
    assert D.freq_implied(86400.0 * 365.25 * 10, 121)[0] == 2629800  # monthly
    assert D.freq_implied(86400.0 * 365.25 * 5, 1259)[0] == 86400.0  # 252 trading days a year
    assert D.freq_implied(12345.0 * 99, 100)[0] is None
    assert [D.freq_band(s) for s in (1, 900, 3600, 86400, 604800, 2629800, 31557600, None)] == [
        "sub-minute", "minute", "hourly", "daily", "weekly", "monthly", "quarterly / yearly", "unknown"]
    assert D.freq_band(60, "events") == "irregular / events" and D.freq_band(None, mixed=True) == "mixed"
    assert [D.length_band(n) for n in (50, 100, 999, 5000, 50000, 100000, None)] == [
        "under 100", "100-1k", "100-1k", "1k-10k", "10k-100k", "over 100k", "unknown"]
    assert D.freq_label(900) == "15min" and D.freq_label(2629800) == "1M" and D.freq_label(7200) == "2h"


def test_slices():
    s = D.slice_estimates(n_series=10_000, n_points=10_000 * 100_000, bytes_all=10 ** 10)
    assert s["slice500_points"] == 500 * 16_000 and s["slice5k_points"] == 5_000 * 64_000
    assert s["slice500_bytes"] == 10 ** 10 * 8_000_000 // 10 ** 9 and s["slice_all_bytes"] == 10 ** 10
    s = D.slice_estimates(n_series=100, n_points=100 * 1000, bytes_all=10 ** 6)  # small: every slice is everything
    assert s["slice500_points"] == s["slice5k_points"] == 100_000 and s["slice500_bytes"] == 10 ** 6
    s = D.slice_estimates(10_000, 10 ** 9, 10 ** 10, partial_fetch=False)  # zip / csv: cannot be cut
    assert s["slice500_bytes"] == 10 ** 10
    assert D.slice_estimates(None, None, 5)["slice500_points"] is None


def test_ingest_difficulty():
    assert D.ingest_difficulty({"parquet": 10}, 1, "long")[0] == "easy"
    assert D.ingest_difficulty({"parquet": 10}, 1, "nested")[0] == "medium"
    assert D.ingest_difficulty({"parquet": 10}, 3, "long")[0] == "medium"
    assert D.ingest_difficulty({"csv": 10}, 0, "unknown")[0] == "medium"
    assert D.ingest_difficulty({"arrow": 10}, 1, "nested")[0] == "medium"
    for e in ("zip", "nc", "h5", "npy", "rar"):
        assert D.ingest_difficulty({e: 10, "parquet": 5}, 1, "long")[0] == "hard"
    assert D.ingest_difficulty({"parquet": 10}, 1, "events")[0] == "hard"
    assert D.ingest_difficulty({}, 0, "unknown") == ("hard", "no data files found")


def test_data_file_and_extension_rules():
    assert P.ext_of("a/b/x.csv.gz") == "csv.gz" and P.ext_of("a/data-00000-of-00001.arrow") == "arrow"
    assert P.is_data_file("train.parquet", 10) and P.is_data_file("wfp_markets_nga.csv", 10)
    for name in ("README.md", ".gitattributes", "SP500/meta.json", "HZMETRO/dataset_info.json", "HZMETRO/state.json",
                 "cloudops_tsf.py", "SP500/references.bib", "LICENSE", "ind-eval.png"):
        assert not P.is_data_file(name, 10), name


# ---- scopes ----------------------------------------------------------------------------------------------------------------
def F(*paths):
    return [{"path": p, "size": 100, "ext": P.ext_of(p), "data": P.is_data_file(p, 100)} for p in paths]


def test_resolve_scope():
    files = F("README.md", "exchange_rate/train-00000.parquet", "solar/train-0.parquet", "solar_1h/train-0.parquet",
              "Dataset_V0.0/Energy-Load-Profiles/15min/Commercial/DGS/DGS-15min.parquet",
              "Dataset_V0.0/Energy-Load-Profiles/Hourly/Commercial/DGS/DGS-Hourly.parquet", "data/silver/a.parquet",
              "data/gold/b.parquet", "smartmeter_001.csv", "smartmeter_002.csv")
    card = {"exchange_rate": ["exchange_rate/*.parquet"], "solar": ["solar/*.parquet"], "solar_1h": ["solar_1h/*.parquet"]}
    names = list(card)
    s = P.resolve_scope("exchange_rate", files, names, card)
    assert s.rule == "config:exchange_rate" and s.configs == ["exchange_rate"] and len(s.files) == 1 and not s.whole
    s = P.resolve_scope("solar / solar_1h", files, names, card)
    assert s.configs == ["solar", "solar_1h"] and len(s.files) == 2 and s.unmatched == []
    s = P.resolve_scope("DGS", files, names, card)
    assert s.rule == "path segment:DGS" and len(s.files) == 2 and s.configs == []
    s = P.resolve_scope("data/silver", files, names, card)
    assert s.rule == "folder:data/silver" and [f["path"] for f in s.files] == ["data/silver/a.parquet"]
    s = P.resolve_scope("smartmeter_*.csv", files, names, card)
    assert s.rule.startswith("glob:") and len(s.files) == 2
    s = P.resolve_scope("", files, names, card)
    assert s.whole and len(s.files) == len(files)
    s = P.resolve_scope("(whole repo: 2,807 folders named ds-<id>-<freq>)", files, names, card)
    assert s.whole and s.rule == "whole repo"
    s = P.resolve_scope("solar, no_such_thing", files, names, card)
    assert s.unmatched == ["no_such_thing"] and s.configs == ["solar"]
    s = P.resolve_scope("8 tables", files, names, card)
    assert s.whole and s.unmatched == ["8 tables"] and "not matched" in s.rule


def test_pick_evenly_keeps_first_and_last():
    items = list(range(1000))
    got = P.pick_evenly(items, 50)
    assert len(got) == 50 and got[0] == 0 and got[-1] == 999 and P.pick_evenly(items[:7], 50) == items[:7]


# ---- end to end on recorded responses (fake session) ---------------------------------------------------------------------
def cat_row(repo, subset="", **kw):
    base = {"_row": 7, "repo_id": repo, "config_or_subset": subset, "domain": "agriculture_nature", "status": "keep",
            "frequency": "M (15th of month)", "approx_series": "~1k", "approx_points": "54883", "format": "parquet",
            "licence": "other (source-specific)", "gated": "no", "repo_size_gb": ""}
    return {**base, **kw}


def wfp_routes():
    q = f"dataset={WFP.replace('/', '%2F')}"
    return {
        f"{HUB}/api/datasets/{WFP}?blobs=true": J(fx_json("info_wfp.json")),
        f"{VIEWER}/is-valid?{q}": J(fx_json("valid_wfp.json")),
        f"{VIEWER}/size?{q}": J(fx_json("size_wfp.json")),
        f"{VIEWER}/info?{q}": J(fx_json("vinfo_wfp.json")),
        f"{VIEWER}/statistics?{q}&config=default&split=train": J(fx_json("stats_wfp.json")),
        f"{HUB}/datasets/{WFP}/resolve/main/train.parquet": {"file": padded(fx_bytes("wfp_footer.bin"), 487487)},
    }


def test_profile_repo_long_parquet_end_to_end(tmp_path):
    f, s, _ = make_fx(tmp_path, wfp_routes())
    rows, frows = P.profile_repo(f, WFP, [cat_row(WFP)], set(), OPTS)
    assert len(rows) == 1
    r = rows[0]
    assert r["profile_level"] == "viewer+footers" and r["error"] == "" and r["hub_exists"] == "yes"
    assert r["gated"] == "no" and r["licence_hub"] == "other" and r["repo_files"] == 5
    assert r["scope_data_files"] == 3 and r["n_footers_read"] == 1 and r["footer_source"] == "repo parquet files"
    assert r["rows"] == 54883 and r["rows_conf"] == D.EXACT
    assert r["layout"] == "long" and r["layout_rule"].startswith("R4")
    assert r["time_col"] == "date" and r["id_cols"] == "market; commodity" and r["value_cols"] == "price; usdprice"
    assert r["n_points"] == 54883 * 2 and r["n_points_conf"] == D.EXACT and r["n_points_target"] == 54883
    assert r["n_series_conf"] == D.EST and 0 < r["n_series"] <= 54883 and "product" in r["n_series_src"]
    assert r["date_first"] == "2002-01-15 00:00:00" and r["date_last"] == "2025-11-15 00:00:00" and r["date_conf"] == D.EXACT
    assert r["missing_share"] == 0.0 and r["missing_conf"] == D.EXACT
    assert r["has_covariates"] == "yes" and r["covariate_cols"] == "usdprice"
    assert r["bytes_all"] == 487487 + 3799 + 5438 and r["slice_all_points"] == r["n_points"]
    assert r["ingest_difficulty"] == "medium"  # the scope also holds two csv files
    assert "non-parquet data files" in r["undetermined"]
    pq_row = next(x for x in frows if x["path"] == "train.parquet")
    assert pq_row["rows"] == 54883 and pq_row["date_first"].startswith("2002-01-15") and pq_row["footer_read"] == "yes"
    assert len(frows) == 3 and {x["format"] for x in frows} == {"parquet", "csv"}
    # only allowed calls were made, and a second run is served from the cache
    kinds = {classify(u, h.get("Range"), None) for u, h in s.calls}
    assert kinds <= {"hub_api", "viewer", "footer"}
    n = len(s.calls)
    f2 = Fetcher(tmp_path / "cache", session=s)
    rows2, _ = P.profile_repo(f2, WFP, [cat_row(WFP)], set(), OPTS)
    assert len(s.calls) == n and f2.bytes_total == 0 and rows2[0]["n_points"] == r["n_points"]


def test_profile_repo_rmisc_folder_copies_meta_json(tmp_path):
    repo = "nju-zhangsq/RMISC"
    q = "dataset=nju-zhangsq%2FRMISC"
    err = J({"error": "not ready"}, 500, {"X-Error-Code": "ResponseNotReady"})
    routes = {
        f"{HUB}/api/datasets/{repo}?blobs=true": J(fx_json("info_rmisc_trimmed.json")),
        f"{VIEWER}/is-valid?{q}": J({"viewer": False, "preview": False, "search": False, "filter": False,
                                     "statistics": False}),
        f"{VIEWER}/size?{q}": err, f"{VIEWER}/info?{q}": err,
        f"{HUB}/datasets/{repo}/resolve/main/SP500/part0.parquet": {"file": padded(fx_bytes("rmisc_sp500_footer.bin"),
                                                                                  9056884)},
        f"{HUB}/datasets/{repo}/resolve/main/SP500/meta.json": FakeResp(200, fx_bytes("rmisc_meta.json")),
    }
    f, s, _ = make_fx(tmp_path, routes)
    cat = cat_row(repo, "SP500", domain="economics_finance", frequency="1D", approx_points="3M", licence="mit")
    rows, frows = P.profile_repo(f, repo, [cat], set(), OPTS)
    r = rows[0]
    assert r["scope_rule"] == "folder:SP500" and r["profile_level"] == "footers" and r["scope_data_files"] == 1
    assert r["meta_num_series"] == 500 and r["meta_licence"] == "Unknown" and r["meta_source"].startswith("https://github")
    assert r["n_series"] == 500 and r["n_series_conf"] == D.EXACT and "meta.json" in r["n_series_src"]
    assert r["rows"] == 603027 and r["n_points"] == 603027 * 5 and r["target_cols"] == "Low; Open; High; Volume; Close"
    assert r["has_covariates"] == "no" and r["frequency"] == "" and r["freq_conf"] == D.UNK  # meta freq is null
    assert r["missing_share"] == round(1258 / (603027 * 5), 6)
    assert r["len_median"] == round(603027 / 500, 2) and r["len_band"] == "1k-10k"
    assert r["ingest_difficulty"] == "easy" and len(frows) == 1 and frows[0]["bytes"] == 9056884


def test_profile_repo_nested_via_converted_parquet(tmp_path):
    """An arrow repo: schema and counts come from the footers of the viewer's parquet copy; freq from the freq column."""
    repo = "Salesforce/lotsa_data"
    q = "dataset=Salesforce%2Flotsa_data"
    lens = [2380] * 80
    tbl = pa.table({"item_id": [str(i) for i in range(80)],
                    "start": pa.array([dt.datetime(2019, 1, 1)] * 80, pa.timestamp("s")), "freq": ["15T"] * 80,
                    "target": pa.array([[[0.5] * n, [1.5] * n] for n in lens], pa.list_(pa.list_(pa.float32())))})
    feats = fx_json("lotsa_hzmetro_dataset_info.json")["features"]  # `datasets` writes these into the parquet footer
    tbl = tbl.replace_schema_metadata({"huggingface": json.dumps({"info": {"features": feats}})})
    data, _footer = _parquet_footer(tbl)
    info = {"id": repo, "private": False, "gated": False, "lastModified": "2026-01-01T00:00:00.000Z",
            "cardData": {"license": "apache-2.0", "configs": [
                {"config_name": "HZMETRO", "data_files": [{"split": "train", "path": "HZMETRO/*.arrow"}]},
                {"config_name": "LOOP_SEATTLE", "data_files": [{"split": "train", "path": "LOOP_SEATTLE/*.arrow"}]}]},
            "siblings": [{"rfilename": "HZMETRO/data-00000-of-00001.arrow", "size": 1524752},
                         {"rfilename": "HZMETRO/dataset_info.json", "size": 965},
                         {"rfilename": "HZMETRO/state.json", "size": 250},
                         {"rfilename": "LOOP_SEATTLE/data-00000-of-00001.arrow", "size": 135824024}]}
    size = {"size": {"configs": [{"config": "HZMETRO", "num_rows": 80}, {"config": "LOOP_SEATTLE", "num_rows": 323}],
                     "splits": [{"config": "HZMETRO", "split": "train", "num_rows": 80, "num_bytes_parquet_files": len(data),
                                 "estimated_num_rows": None},
                                {"config": "LOOP_SEATTLE", "split": "train", "num_rows": 323,
                                 "num_bytes_parquet_files": 5, "estimated_num_rows": None}]}, "partial": False}
    pq_list = fx_json("parquet_lotsa.json")
    pq_list["parquet_files"][0]["size"] = len(data)
    routes = {
        f"{HUB}/api/datasets/{repo}?blobs=true": J(info),
        f"{VIEWER}/is-valid?{q}": J({"viewer": True}), f"{VIEWER}/size?{q}": J(size),
        f"{VIEWER}/parquet?{q}": J(pq_list),
        f"{VIEWER}/statistics?{q}&config=HZMETRO&split=train": J(fx_json("stats_lotsa.json")),
        pq_list["parquet_files"][0]["url"]: {"file": data},
    }
    f, s, _ = make_fx(tmp_path, routes)
    cat = cat_row(repo, "HZMETRO", domain="mobility", frequency="15T", approx_points="", format="arrow", licence="apache-2.0")
    rows, frows = P.profile_repo(f, repo, [cat], {"loopseattle"}, OPTS)
    assert len(rows) == 1
    r = rows[0]
    assert r["scope_rule"] == "config:HZMETRO" and r["profile_level"] == "viewer+footers"
    assert r["footer_source"].startswith("viewer converted parquet") and r["layout"] == "nested"
    assert r["n_series"] == 80 and r["n_points"] == 80 * 2 * 2380 and r["n_points_conf"] == D.EXACT
    assert r["frequency"] == "15min" and r["freq_conf"] == D.EXACT and r["freq_src"].startswith("freq column")
    assert r["freq_band"] == "minute" and r["len_median"] == 2380.0 and r["len_band"] == "1k-10k"
    assert r["bytes_all"] == 1524752 and r["ingest_difficulty"] == "medium" and r["problems"] == ""
    assert {x["revision"] for x in frows} == {"main", "refs/convert/parquet"}
    # the data file itself was never requested without a footer range
    assert all("Range" in h for u, h in s.calls if u.endswith(".parquet"))
    assert not any(u.endswith(".arrow") for u, _ in s.calls)


def test_missing_gated_and_failing_repos_do_not_stop(tmp_path):
    gone = f"{HUB}/api/datasets/nobody/gone?blobs=true"
    f, _, _ = make_fx(tmp_path, {gone: J({"error": "Invalid username or password."}, 401)})
    rows, frows = P.profile_repo(f, "nobody/gone", [cat_row("nobody/gone")], set(), OPTS)
    assert rows[0]["profile_level"] == "missing" and "HTTP 401" in rows[0]["error"] and frows == []
    repo, q = "org/locked", "dataset=org%2Flocked"
    denied = J({"error": "Access to dataset org/locked is restricted."}, 401)
    routes = {f"{HUB}/api/datasets/{repo}?blobs=true": J({"id": repo, "gated": "manual", "private": False, "cardData": {},
                                                          "siblings": [{"rfilename": "data/train.parquet", "size": 10 ** 6}]}),
              f"{VIEWER}/is-valid?{q}": denied, f"{VIEWER}/size?{q}": denied, f"{VIEWER}/info?{q}": denied,
              f"{HUB}/datasets/{repo}/resolve/main/data/train.parquet": denied}
    f, _, _ = make_fx(tmp_path / "g", routes)
    rows, frows = P.profile_repo(f, repo, [cat_row(repo, gated="manual")], set(), OPTS)
    r = rows[0]
    assert r["profile_level"] == "gated" and r["error"] == "gated, no access" and "gated, no access" in r["problems"]
    assert r["bytes_all"] == 10 ** 6 and len(frows) == 1 and frows[0]["footer_read"] == "no"


def test_run_writes_outputs_and_report(tmp_path, monkeypatch, capsys):
    cat = tmp_path / "cat.csv"
    head = ("repo_id,config_or_subset,domain,subdomain,frequency,approx_series,approx_points,variates,covariates,licence,"
            "gated,format,overlap,pretrain,status,repo_size_gb,hub_checked,original_source,evidence_url,notes")
    cat.write_text(head + f"\n{WFP},,agriculture_nature,x,M,~1k,54883,,none,other,no,parquet,none,no,keep,,,,,\n"
                   "nobody/gone,,energy,x,1H,1,1,,none,mit,no,csv,none,no,check,,,,,\n", encoding="utf-8")
    routes = wfp_routes()
    routes[f"{HUB}/api/datasets/nobody/gone?blobs=true"] = J({"error": "Invalid username or password."}, 401)
    session = FakeSession(routes)
    real = Fetcher.__init__

    def init(self, *a, **kw):
        real(self, *a, **{**kw, "session": session, "token": None})

    monkeypatch.setattr(Fetcher, "__init__", init)
    monkeypatch.setenv("HF_TOKEN", "hf_SECRET_TOKEN")
    out = tmp_path / "out"
    rc = P.main(["--catalog", str(cat), "--excluded", str(tmp_path / "none.csv"), "--out", str(out),
                 "--cache", str(tmp_path / "cache")])
    printed = capsys.readouterr().out
    assert rc == 0 and "TOTAL BYTES TRANSFERRED:" in printed and "hf_SECRET_TOKEN" not in printed
    prof = P.read_catalog(out / "hf_profile.csv")
    assert [r["profile_level"] for r in prof] == ["viewer+footers", "missing"] and prof[0]["n_points"] == str(54883 * 2)
    assert len(P.read_catalog(out / "hf_profile_files.csv")) == 3
    md = (out / "HF_PROFILE.md").read_text(encoding="utf-8")
    for i in range(1, 12):
        assert f"## {i}. " in md
    assert "nobody/gone" in md.split("## 10. Problems")[1] and "status = keep" in md and "hf_SECRET_TOKEN" not in md
    for p in out.rglob("*"):
        if p.is_file():
            assert b"hf_SECRET_TOKEN" not in p.read_bytes()
