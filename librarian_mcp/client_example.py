"""Client helper: handle `read_binary` responses from GameDocServer.

- Handles inline base64 payloads and HTTP-fallback URLs.
- Example usage: adapt the MCP call to your environment and pass the tool response
  into `handle_read_binary_response()`.
"""

import base64
import mimetypes
import os

try:
    import requests
except Exception:
    requests = None


def save_inline(resp: dict, out_path: str | None = None) -> str:
    content_type = resp.get("content_type", "application/octet-stream")
    ext = mimetypes.guess_extension(content_type.split(";")[0]) or ""
    filename = out_path or ("output" + ext)
    data = base64.b64decode(resp["data_b64"])
    with open(filename, "wb") as f:
        f.write(data)
    print("Saved inline ->", filename)
    return filename


def download_http(resp: dict, out_path: str | None = None) -> str:
    if requests is None:
        raise RuntimeError("The 'requests' package is required to download HTTP fallback files. Install with: pip install requests")
    url = resp["url"]
    r = requests.get(url, stream=True)
    r.raise_for_status()
    content_type = resp.get("content_type") or r.headers.get("Content-Type")
    ext = mimetypes.guess_extension((content_type or "").split(";")[0]) or ""
    filename = out_path or ("output" + ext)
    with open(filename, "wb") as f:
        for chunk in r.iter_content(8192):
            if chunk:
                f.write(chunk)
    print("Downloaded via HTTP ->", filename)
    return filename


def handle_read_binary_response(resp: dict, out_path: str | None = None) -> str:
    """Process a `read_binary` response and save to disk.

    - For `method == 'inline'`: decodes base64 and writes bytes.
    - For `method == 'http'`: downloads from the provided URL.

    Returns the saved filename.
    """
    method = resp.get("method")
    if method == "inline":
        return save_inline(resp, out_path)
    if method == "http":
        return download_http(resp, out_path)
    raise ValueError(f"Unknown response method: {method}")


if __name__ == "__main__":
    # MOCK INLINE example (uses a small slice of this file as the payload)
    with open(__file__, "rb") as fh:
        sample = fh.read(1024)
    mock_inline = {
        "method": "inline",
        "content_type": "text/plain",
        "data_b64": base64.b64encode(sample).decode("ascii"),
        "size": len(sample),
    }
    handle_read_binary_response(mock_inline, "sample_inline.txt")

    # MOCK HTTP example (requires the server HTTP adapter to be running and reachable)
    mock_http = {
        "method": "http",
        "url": "http://127.0.0.1:8001/README.md",
        "content_type": "text/plain",
        "size": 123,
    }
    # To test HTTP fallback, ensure the GameDocServer HTTP adapter is running and
    # the URL above is reachable, then uncomment the following line:
    # handle_read_binary_response(mock_http, "sample_http.txt")

    # Example: how an MCP client might call the tool (pseudo-code; adapt to your MCP client):
    # from mcp import MCPClient
    # client = MCPClient(host='127.0.0.1', port=8000)
    # resp = client.call_tool('read_binary', {'file_path': 'images/logo.png', 'max_inline_bytes': 5*1024*1024})
    # handle_read_binary_response(resp, 'logo.png')
