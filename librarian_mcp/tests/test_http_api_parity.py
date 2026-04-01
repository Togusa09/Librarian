import asyncio
import json
import urllib.error
import urllib.request

import pytest

from librarian_mcp import server


def _run(coro):
    return asyncio.run(coro)


def _post_json(base, tool_name, payload):
    req = urllib.request.Request(f"{base}/tools/{tool_name}", method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, data=json.dumps(payload).encode("utf-8")) as r:
        return r.status, dict(r.headers), json.load(r)


def _post_raw(base, tool_name, raw_bytes):
    req = urllib.request.Request(f"{base}/tools/{tool_name}", method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, data=raw_bytes) as r:
        return r.status, dict(r.headers), json.load(r)


def _start_api(svc):
    svc.start_http_api(host="127.0.0.1", port=0)
    port = svc._api_server.server_address[1]
    return f"http://127.0.0.1:{port}"


def test_get_routes_parity(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    svc = server.GameDocServer(docs_root=str(root))
    base = _start_api(svc)

    try:
        for path in ["/.well-known/mcp.json", "/healthz", "/readyz", "/tools", "/cache_status"]:
            with urllib.request.urlopen(f"{base}{path}") as r:
                body = json.load(r)
                assert r.status == 200
                assert "content-type" in {k.lower(): v for k, v in dict(r.headers).items()}
                assert "application/json" in r.headers.get("Content-Type", "")
                if path != "/tools":
                    assert "request_id" in body or path == "/.well-known/mcp.json"
    finally:
        _run(svc.shutdown())


def test_options_and_head_parity(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    svc = server.GameDocServer(docs_root=str(root))
    base = _start_api(svc)

    try:
        req_options = urllib.request.Request(f"{base}/tools/list_files", method="OPTIONS")
        with urllib.request.urlopen(req_options) as r:
            assert r.status in (200, 204)
            assert r.headers.get("Access-Control-Allow-Origin") == "*"

        req_head = urllib.request.Request(f"{base}/healthz", method="HEAD")
        with urllib.request.urlopen(req_head) as r:
            assert r.status == 200
            assert "application/json" in r.headers.get("Content-Type", "")
    finally:
        _run(svc.shutdown())


def test_error_shape_parity(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    svc = server.GameDocServer(docs_root=str(root))
    base = _start_api(svc)

    try:
        req = urllib.request.Request(f"{base}/tools/unknown_tool", method="POST")
        req.add_header("Content-Type", "application/json")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, data=b"{}")
        err = exc_info.value
        body = json.loads(err.read().decode("utf-8"))
        assert err.code == 404
        assert body.get("error") in ("unknown_tool", "not_found")
        assert "request_id" in body

        req = urllib.request.Request(f"{base}/tools/read_document", method="POST")
        req.add_header("Content-Type", "application/json")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, data=b"{bad-json")
        err = exc_info.value
        body = json.loads(err.read().decode("utf-8"))
        assert err.code == 400
        assert body.get("error") == "invalid_json"
        assert "request_id" in body

        req = urllib.request.Request(f"{base}/tools/read_document", method="POST")
        req.add_header("Content-Type", "application/json")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req, data=json.dumps({}).encode("utf-8"))
        err = exc_info.value
        body = json.loads(err.read().decode("utf-8"))
        assert err.code == 400
        assert body.get("error") == "bad_request"
        assert "request_id" in body
    finally:
        _run(svc.shutdown())


def test_search_alias_parity(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "one.md").write_text("alpha beta gamma")

    svc = server.GameDocServer(docs_root=str(root))
    base = _start_api(svc)

    try:
        status, _headers, body_q = _post_json(base, "search_knowledge_base", {"q": "alpha", "top_k": 1})
        assert status == 200
        assert "request_id" in body_q
        assert isinstance(body_q.get("result", {}).get("results"), list)

        status, _headers, body_query = _post_json(base, "search_knowledge_base", {"query": "alpha", "n": 1})
        assert status == 200
        assert "request_id" in body_query
        assert isinstance(body_query.get("result", {}).get("results"), list)
    finally:
        _run(svc.shutdown())
