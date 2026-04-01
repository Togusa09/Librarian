import json
import urllib.request
import urllib.error
from pathlib import Path

from librarian_mcp import server


def _start_api(svc):
    svc.start_http_api(host="127.0.0.1", port=0)
    # Wait for server to bind
    port = svc._api_server.server_address[1]
    return f"http://127.0.0.1:{port}"


def test_discovery_and_health(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    svc = server.GameDocServer(docs_root=str(root))
    base = _start_api(svc)

    # discovery
    with urllib.request.urlopen(f"{base}/.well-known/mcp.json") as r:
        body = json.load(r)
    assert "base_url" in body
    assert "tools" in body

    # health
    with urllib.request.urlopen(f"{base}/healthz") as r:
        body = json.load(r)
    assert body.get("status") == "ok"

    # shutdown
    svc._api_server.shutdown()
    svc._api_server.server_close()


def test_list_and_read_and_read_binary(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    text = root / "one.md"
    text.write_text("hello-api")
    small = root / "small.bin"
    data = b"abc123"
    small.write_bytes(data)

    svc = server.GameDocServer(docs_root=str(root))
    base = _start_api(svc)

    # list_files
    req = urllib.request.Request(f"{base}/tools/list_files", method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, data=json.dumps({"path": ""}).encode("utf-8")) as r:
        body = json.load(r)
    files = body.get("result", {}).get("files", [])
    # files now contain metadata objects; check that our file appears by path
    paths = [f.get("path") if isinstance(f, dict) else f for f in files]
    assert "one.md" in paths

    # read_document
    req = urllib.request.Request(f"{base}/tools/read_document", method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, data=json.dumps({"path": "one.md"}).encode("utf-8")) as r:
        body = json.load(r)
    assert "hello-api" in body.get("result", {}).get("text", "")

    # read_binary (inline)
    req = urllib.request.Request(f"{base}/tools/read_binary", method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, data=json.dumps({"path": "small.bin"}).encode("utf-8")) as r:
        body = json.load(r)
    res = body.get("result", {})
    assert res.get("method") == "inline"
    assert "data_b64" in res

    # shutdown
    svc._api_server.shutdown()
    svc._api_server.server_close()
