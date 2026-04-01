import threading
import time
from concurrent.futures import ThreadPoolExecutor

from librarian_mcp import server


def _create_docs(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "one.md").write_text("alpha beta gamma\n" * 8)
    return docs


def test_summary_cache_hits_memory(monkeypatch, tmp_path):
    docs = _create_docs(tmp_path)
    data_path = tmp_path / "data"
    monkeypatch.setenv("DATA_PATH", str(data_path))

    calls = {"count": 0}

    def fake_generate(model, prompt, keep_alive=None):
        calls["count"] += 1
        return "cached-summary"

    monkeypatch.setattr(server, "_call_ollama_generate", fake_generate)

    svc = server.GameDocServer(docs_root=str(docs))
    s1 = svc.summarize_context("file", "one.md", "concise")
    s2 = svc.summarize_context("file", "one.md", "concise")

    assert s1 == "cached-summary"
    assert s2 == "cached-summary"
    assert calls["count"] == 1

    status = svc.get_cache_status()
    assert status["cache_hits_mem"] == 1
    assert status["cache_misses"] == 1
    assert status["cache_computes"] == 1


def test_summary_cache_hits_sqlite_across_instances(monkeypatch, tmp_path):
    docs = _create_docs(tmp_path)
    data_path = tmp_path / "data"
    monkeypatch.setenv("DATA_PATH", str(data_path))

    calls = {"count": 0}

    def fake_generate(model, prompt, keep_alive=None):
        calls["count"] += 1
        return "persisted-summary"

    monkeypatch.setattr(server, "_call_ollama_generate", fake_generate)

    svc1 = server.GameDocServer(docs_root=str(docs))
    assert svc1.summarize_context("file", "one.md", "concise") == "persisted-summary"
    assert calls["count"] == 1

    def should_not_run(model, prompt, keep_alive=None):
        raise AssertionError("LLM should not be called when sqlite cache entry exists")

    monkeypatch.setattr(server, "_call_ollama_generate", should_not_run)

    svc2 = server.GameDocServer(docs_root=str(docs))
    assert svc2.summarize_context("file", "one.md", "concise") == "persisted-summary"

    status = svc2.get_cache_status()
    assert status["cache_hits_sqlite"] == 1
    assert status["cache_computes"] == 0


def test_summary_cache_expired_entry_recomputes(monkeypatch, tmp_path):
    docs = _create_docs(tmp_path)
    data_path = tmp_path / "data"
    monkeypatch.setenv("DATA_PATH", str(data_path))

    calls = {"count": 0}

    def fake_generate(model, prompt, keep_alive=None):
        calls["count"] += 1
        return f"summary-{calls['count']}"

    monkeypatch.setattr(server, "_call_ollama_generate", fake_generate)

    svc = server.GameDocServer(docs_root=str(docs))
    svc.summary_cache_ttl_sec = -1

    s1 = svc.summarize_context("file", "one.md", "concise")
    s2 = svc.summarize_context("file", "one.md", "concise")

    assert s1 == "summary-1"
    assert s2 == "summary-2"
    assert calls["count"] == 2


def test_summary_cache_dedupes_inflight_requests(monkeypatch, tmp_path):
    docs = _create_docs(tmp_path)
    data_path = tmp_path / "data"
    monkeypatch.setenv("DATA_PATH", str(data_path))

    calls = {"count": 0}
    lock = threading.Lock()

    def fake_generate(model, prompt, keep_alive=None):
        with lock:
            calls["count"] += 1
        time.sleep(0.2)
        return "shared-summary"

    monkeypatch.setattr(server, "_call_ollama_generate", fake_generate)

    svc = server.GameDocServer(docs_root=str(docs))

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(svc.summarize_context, "file", "one.md", "concise")
        b = pool.submit(svc.summarize_context, "file", "one.md", "concise")

    assert a.result() == "shared-summary"
    assert b.result() == "shared-summary"
    assert calls["count"] == 1
