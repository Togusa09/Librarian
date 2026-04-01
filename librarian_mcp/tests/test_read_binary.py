import base64
import urllib.request
from pathlib import Path

from librarian_mcp import server


def test_read_binary_inline(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    f = root / "small.bin"
    data = b"hello-world" * 100
    f.write_bytes(data)

    svc = server.GameDocServer(docs_root=str(root))

    resp = svc.read_binary("small.bin", max_inline_bytes=1024 * 1024)
    assert resp["method"] == "inline"
    assert "data_b64" in resp

    decoded = base64.b64decode(resp["data_b64"])
    assert decoded == data


def test_read_binary_http_fallback(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    f = root / "big.bin"
    data = b"A" * (8 * 1024)  # 8 KiB
    f.write_bytes(data)

    svc = server.GameDocServer(docs_root=str(root))

    # Force HTTP fallback by setting a small max_inline_bytes
    resp = svc.read_binary("big.bin", max_inline_bytes=1024)
    assert resp["method"] == "http"
    assert "url" in resp

    url = resp["url"]
    with urllib.request.urlopen(url) as r:
        body = r.read()

    assert body == data

    # Cleanly shutdown the adapter
    if getattr(svc, "_http_server", None):
        try:
            svc._http_server.shutdown()
            svc._http_server.server_close()
        except Exception:
            pass
