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


def _post_json(base, tool_name, payload):
    req = urllib.request.Request(f"{base}/tools/{tool_name}", method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, data=json.dumps(payload).encode("utf-8")) as r:
        return json.load(r)


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


def test_summarize_context_uses_http_memory_cache(monkeypatch, tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "one.md").write_text("http cache test\n" * 12)
    data_path = tmp_path / "data"
    monkeypatch.setenv("DATA_PATH", str(data_path))
    monkeypatch.setattr(server.GameDocServer, "_start_ollama_prewarm", lambda self: None)
    monkeypatch.setattr(server.GameDocServer, "_start_summary_background", lambda self: None)

    calls = {"count": 0}

    def fake_generate(model, prompt, keep_alive=None):
        calls["count"] += 1
        return "http-cached-summary"

    monkeypatch.setattr(server, "_call_ollama_generate", fake_generate)

    svc = server.GameDocServer(docs_root=str(root))
    base = _start_api(svc)
    try:
        body1 = _post_json(base, "summarize_context", {"scope": "file", "target": "one.md", "depth": "concise"})
        body2 = _post_json(base, "summarize_context", {"scope": "file", "target": "one.md", "depth": "concise"})

        assert body1["result"]["summary"] == "http-cached-summary"
        assert body2["result"]["summary"] == "http-cached-summary"
        assert calls["count"] == 1

        status = svc.get_cache_status()
        assert status["cache_hits_mem"] == 1
        assert status["cache_misses"] == 1
    finally:
        svc.shutdown()


def test_summarize_context_uses_http_sqlite_cache_across_instances(monkeypatch, tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "one.md").write_text("persisted http cache test\n" * 12)
    data_path = tmp_path / "data"
    monkeypatch.setenv("DATA_PATH", str(data_path))
    monkeypatch.setattr(server.GameDocServer, "_start_ollama_prewarm", lambda self: None)
    monkeypatch.setattr(server.GameDocServer, "_start_summary_background", lambda self: None)

    calls = {"count": 0}

    def fake_generate(model, prompt, keep_alive=None):
        calls["count"] += 1
        return "http-persisted-summary"

    monkeypatch.setattr(server, "_call_ollama_generate", fake_generate)

    svc1 = server.GameDocServer(docs_root=str(root))
    base1 = _start_api(svc1)
    try:
        body = _post_json(base1, "summarize_context", {"scope": "file", "target": "one.md", "depth": "concise"})
        assert body["result"]["summary"] == "http-persisted-summary"
        assert calls["count"] == 1
    finally:
        svc1.shutdown()

    def should_not_run(model, prompt, keep_alive=None):
        raise AssertionError("LLM should not run when summarize_context is served from sqlite cache")

    monkeypatch.setattr(server, "_call_ollama_generate", should_not_run)

    svc2 = server.GameDocServer(docs_root=str(root))
    base2 = _start_api(svc2)
    try:
        body = _post_json(base2, "summarize_context", {"scope": "file", "target": "one.md", "depth": "concise"})
        assert body["result"]["summary"] == "http-persisted-summary"

        status = svc2.get_cache_status()
        assert status["cache_hits_sqlite"] == 1
        assert status["cache_computes"] == 0
    finally:
        svc2.shutdown()
