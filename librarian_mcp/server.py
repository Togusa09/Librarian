"""
Placeholder MCP server for local documentation.

Run:
    python server.py

This file contains simple startup skeletons and TODOs for MCP handlers.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Any
import json
import subprocess
import base64
import mimetypes
import threading
import http.server
import socketserver
import socket
import urllib.parse
import os
import json as _json
import uuid
import logging

# MCP server import (try FastMCP first)
try:
    from mcp import FastMCP
except Exception:
    try:
        from mcp import MCPServer as FastMCP
    except Exception:
        FastMCP = None

try:
    from .indexer import DocIndexer
except Exception:
    from indexer import DocIndexer


def _call_ollama_generate(model: str, prompt: str) -> str:
    """Generate text with Ollama.

    Tries the `ollama` python package, otherwise raises an informative error. CLI
    fallbacks are intentionally omitted because CLI args vary; if you prefer CLI,
    adjust this function to call `ollama run` or similar.
    """
    try:
        import ollama  # type: ignore
    except Exception:
        raise RuntimeError("ollama python package not available; install it or adjust _call_ollama_generate to use the CLI")

    # Attempt a few common client method names
    if hasattr(ollama, "completion"):
        return ollama.completion(model=model, prompt=prompt)
    if hasattr(ollama, "chat"):
        return ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    if hasattr(ollama, "run"):
        return ollama.run(model, prompt)

    raise RuntimeError("ollama client present but no known generation method found; please adapt _call_ollama_generate for your ollama client")


def _resolve_within_root(root_p: Path, requested: str) -> Path:
    """Resolve a requested (relative) path within the given root and prevent escape.

    `requested` may be either a relative path or an absolute path. The final resolved
    path will always be checked to ensure it's inside `root_p`.
    """
    root_resolved = root_p.resolve()
    req = Path(requested)
    # If requested is absolute, resolve it; otherwise join with root
    if req.is_absolute():
        req_resolved = req.resolve()
    else:
        req_resolved = (root_resolved / req).resolve()

    try:
        # Python 3.9+: use is_relative_to for clarity
        if hasattr(req_resolved, "is_relative_to"):
            if not req_resolved.is_relative_to(root_resolved):
                raise PermissionError(f"Requested path outside configured docs root: {requested}")
        else:
            # Fallback for older Python: compare commonpath
            import os

            if os.path.commonpath([str(root_resolved)]) != os.path.commonpath([str(root_resolved), str(req_resolved)]):
                raise PermissionError(f"Requested path outside configured docs root: {requested}")
    except Exception:
        raise

    return req_resolved


class GameDocServer:
    """FastMCP server wrapper registering tools for documentation QA.

    Tools implemented:
    - list_files(directory)
    - read_document(file_path)
    - search_knowledge_base(query)
    - summarize_context(scope, target)
    """

    def __init__(self, name: str = "GameDocServer", docs_root: str = None):
        """Create the server.

        Configuration precedence (per setting): constructor arg > specific env var > sensible default

        - Documentation root: `docs_root` arg > `DOCUMENT_ARCHIVE_PATH` env var > './docs'
        - Data path (for DBs, caches): `DATA_PATH` env var > './data'
        - Bind host: `BIND_HOST` env var > '127.0.0.1'
        - Port: `PORT` env var > 8000
        """
        import os

        self.name = name
        self.server = None
        if FastMCP is not None:
            try:
                self.server = FastMCP(name=self.name)
            except Exception:
                self.server = None

        # Paths and network configuration (exposed for Docker/compose)
        docs_env = os.environ.get("DOCUMENT_ARCHIVE_PATH") or os.environ.get("DOCS_ROOT")
        chosen = docs_root or docs_env or "./docs"
        self.docs_root = Path(chosen).resolve()

        self.data_path = os.environ.get("DATA_PATH") or "./data"
        # normalize to resolved Path when used

        self.bind_host = os.environ.get("BIND_HOST") or "127.0.0.1"
        try:
            self.port = int(os.environ.get("PORT", "8000"))
        except Exception:
            self.port = 8000

        # DocIndexer will be created lazily to avoid requiring chromadb/ollama at import time
        self.indexer = None

        # HTTP adapter for serving large files (started on demand)
        self._http_server = None
        self._http_thread = None
        self._http_port = None
        # HTTP API server (for LLM-friendly JSON POST tool API)
        self._api_server = None
        self._api_thread = None
        # Simple tool schema registry used by the HTTP API
        self._tool_schemas = {
            "list_files": {"input": {"directory": "string", "limit": "int?", "offset": "int?"}, "output": {"files": "list"}},
            "read_document": {"input": {"file_path": "string"}, "output": {"text": "string", "mime": "string?"}},
            "search_knowledge_base": {"input": {"q": "string", "top_k": "int?"}, "output": {"results": "list"}},
            "read_binary": {"input": {"file_path": "string", "prefer": "string?", "max_inline_bytes": "int?"}, "output": {"method": "string", "data_b64": "string?", "url": "string?", "content_type": "string", "size": "int"}},
            "summarize_context": {"input": {"scope": "string", "target": "string"}, "output": {"summary": "string"}},
        }

    # Tool implementations
    def list_files(self, directory: str = "") -> List[str]:
        # Directory is interpreted relative to configured docs_root unless absolute.
        p = _resolve_within_root(self.docs_root, directory)
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")
        files: List[str] = []
        for f in p.rglob("*"):
            if f.is_file():
                # Return path relative to docs_root to avoid exposing system paths
                files.append(str(f.relative_to(self.docs_root)))
        return files

    def read_document(self, file_path: str) -> str:
        # file_path is interpreted relative to docs_root unless absolute
        p = _resolve_within_root(self.docs_root, file_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")
        return p.read_text(encoding="utf-8", errors="ignore")

    def search_knowledge_base(self, query: str, n: int = 5) -> List[Dict[str, Any]]:
        # Lazy-create indexer when first needed
        if self.indexer is None:
            try:
                # Place persistent DB under configured `data_path` so it can be volume-mounted
                persist_dir = str(Path(self.data_path) / "chroma_db")
                self.indexer = DocIndexer(persist_directory=persist_dir, collection_name="game_docs", docs_root=str(self.docs_root))
            except Exception as e:
                raise RuntimeError(f"Failed to initialize DocIndexer: {e}")

        return self.indexer.semantic_search(query, n=n)

    def summarize_context(self, scope: str, target: str) -> str:
        """Summarize content according to `scope`:

        - 'file': `target` is file path; summarize the file.
        - 'directory': `target` is directory path; summarize each file then give feature-level overview.
        - 'snippet': `target` is the text to summarize.
        """
        model = "llama3"

        if scope == "file":
            text = self.read_document(target)
            prompt = f"Summarize the following document, focusing on feature-level descriptions and key points:\n\n{text}"
            return _call_ollama_generate(model, prompt)

        if scope == "snippet":
            prompt = f"Summarize this snippet concisely:\n\n{target}"
            return _call_ollama_generate(model, prompt)

        if scope == "directory":
            # List all files (relative paths) and produce per-file mini-summaries using the first 2000 tokens.
            files = self.list_files(target)
            mini_summaries: List[str] = []

            def _truncate_to_tokens(text: str, max_tokens: int) -> str:
                tokens = text.split()
                if len(tokens) <= max_tokens:
                    return text
                return " ".join(tokens[:max_tokens])

            for rel in files:
                if not rel.lower().endswith((".md", ".txt")):
                    continue
                try:
                    text = self.read_document(rel)
                except Exception:
                    continue

                truncated = _truncate_to_tokens(text, 2000)
                prompt = (
                    "Mini-summary (2-3 sentences) of the following document. Use the first 2000 tokens only:\n\n"
                    + truncated
                )
                try:
                    summary = _call_ollama_generate(model, prompt)
                except Exception as e:
                    summary = f"(mini-summary failed: {e})"

                mini_summaries.append(f"File: {rel}\nMini-summary: {summary}")

            # Aggregate all mini-summaries and ask Ollama for a high-level architectural overview
            aggregation_prompt = (
                "You are given many mini-summaries extracted from documentation files for a single feature. "
                "Produce a High-level Architectural Overview of the feature, grouping related capabilities, describing key components, "
                "their interactions, dependencies, and any notable gaps or risks. Use the mini-summaries below as the source:\n\n"
                + "\n\n".join(mini_summaries)
            )

            return _call_ollama_generate(model, aggregation_prompt)

        raise ValueError("Unknown scope. Must be one of: 'file', 'directory', 'snippet'")

    def read_binary(self, file_path: str, max_inline_bytes: int = 5 * 1024 * 1024) -> Dict[str, Any]:
        """Return binary content for `file_path`.

        - If the file is smaller than `max_inline_bytes`, return a base64 payload:
            {"method": "inline", "content_type": "image/png", "data_b64": "...", "size": 123}

        - If larger, ensure the HTTP adapter is running and return a URL:
            {"method": "http", "url": "http://host:port/rel/path", "content_type": "...", "size": 123}

        This avoids sending huge blobs over MCP while still providing a fallback.
        """
        p = _resolve_within_root(self.docs_root, file_path)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"File not found: {file_path}")

        size = p.stat().st_size
        content_type = mimetypes.guess_type(str(p))[0] or "application/octet-stream"

        if size <= max_inline_bytes:
            data = p.read_bytes()
            b64 = base64.b64encode(data).decode("ascii")
            return {"method": "inline", "content_type": content_type, "data_b64": b64, "size": size}

        # Otherwise, provide an HTTP URL via the adapter
        url = self._ensure_http_adapter_and_url(str(p.relative_to(self.docs_root)))
        return {"method": "http", "url": url, "content_type": content_type, "size": size}

    def _ensure_http_adapter_and_url(self, rel_path: str) -> str:
        """Start the HTTP adapter if needed and return a URL for `rel_path` (relative to docs_root)."""
        # Start server if not running
        if self._http_server is None:
            # Create a request handler that only serves files under docs_root
            docs_root = str(self.docs_root)

            class _Handler(http.server.BaseHTTPRequestHandler):
                def do_GET(self_inner):
                    parsed = urllib.parse.urlparse(self_inner.path)
                    rel = urllib.parse.unquote(parsed.path).lstrip("/")
                    target = Path(docs_root) / rel
                    try:
                        target_resolved = target.resolve()
                    except Exception:
                        self_inner.send_error(404)
                        return

                    # Ensure containment
                    root_resolved = Path(docs_root).resolve()
                    try:
                        if hasattr(target_resolved, "is_relative_to"):
                            if not target_resolved.is_relative_to(root_resolved):
                                self_inner.send_error(403)
                                return
                        else:
                            if os.path.commonpath([str(root_resolved)]) != os.path.commonpath([str(root_resolved), str(target_resolved)]):
                                self_inner.send_error(403)
                                return
                    except Exception:
                        self_inner.send_error(403)
                        return

                    if not target_resolved.exists() or not target_resolved.is_file():
                        self_inner.send_error(404)
                        return

                    # Serve file
                    try:
                        content_type = mimetypes.guess_type(str(target_resolved))[0] or "application/octet-stream"
                        self_inner.send_response(200)
                        self_inner.send_header("Content-Type", content_type)
                        self_inner.send_header("Content-Length", str(target_resolved.stat().st_size))
                        self_inner.end_headers()
                        with open(target_resolved, "rb") as fh:
                            shutil_copyfileobj(fh, self_inner.wfile)
                    except Exception:
                        self_inner.send_error(500)

                def log_message(self_inner, format, *args):
                    # Silence default logging; keep noisy output out of server logs
                    return

            # Helper to copy file-like objects in chunks without importing shutil in many places
            def shutil_copyfileobj(fsrc, fdst, length=16 * 1024):
                while True:
                    buf = fsrc.read(length)
                    if not buf:
                        break
                    fdst.write(buf)

            # Bind to an ephemeral port on configured host
            class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
                daemon_threads = True

            # Try to bind; allow the OS to pick an available port (port 0)
            server = None
            for attempt in range(3):
                try:
                    srv = _ThreadingHTTPServer((self.bind_host, 0), _Handler)
                    server = srv
                    break
                except OSError:
                    # Try again briefly
                    continue

            if server is None:
                raise RuntimeError("Failed to start HTTP adapter for serving files")

            # Save and run in background thread
            self._http_server = server
            self._http_port = server.server_address[1]

            def _serve():
                try:
                    server.serve_forever()
                finally:
                    try:
                        server.server_close()
                    except Exception:
                        pass

            t = threading.Thread(target=_serve, daemon=True)
            t.start()
            self._http_thread = t

        # Construct URL for the relative path
        host = self.bind_host
        port = self._http_port
        # Ensure rel_path does not start with a slash
        rel_path_clean = rel_path.lstrip("/")
        return f"http://{host}:{port}/{urllib.parse.quote(rel_path_clean)}"

    # Registration helper for FastMCP-compatible servers
    def register_tools(self):
        if self.server is None:
            print("FastMCP not available; tools are available via GameDocServer instance methods.")
            return

        # Try a few register patterns
        if hasattr(self.server, "register_tool"):
            self.server.register_tool("list_files", self.list_files)
            self.server.register_tool("read_document", self.read_document)
            self.server.register_tool("search_knowledge_base", self.search_knowledge_base)
            self.server.register_tool("summarize_context", self.summarize_context)
            # Image/binary support (inline base64 or HTTP fallback)
            self.server.register_tool("read_binary", self.read_binary)
        elif hasattr(self.server, "add_tool"):
            self.server.add_tool("list_files", self.list_files)
            self.server.add_tool("read_document", self.read_document)
            self.server.add_tool("search_knowledge_base", self.search_knowledge_base)
            self.server.add_tool("summarize_context", self.summarize_context)
            self.server.add_tool("read_binary", self.read_binary)
        else:
            # Best-effort: attach as attributes
            setattr(self.server, "list_files", self.list_files)
            setattr(self.server, "read_document", self.read_document)
            setattr(self.server, "search_knowledge_base", self.search_knowledge_base)
            setattr(self.server, "summarize_context", self.summarize_context)
            setattr(self.server, "read_binary", self.read_binary)

    # Minimal HTTP JSON API for LLM-friendly tool calls
    def start_http_api(self, host: str = None, port: int = None):
        """Start a small HTTP server exposing JSON POST endpoints for tools.

        Exposed endpoints:
        - GET /.well-known/mcp.json
        - GET /healthz, GET /readyz
        - GET /tools -> returns available tools and input schemas
        - POST /tools/<tool_name> -> invoke tool with JSON body
        """
        if self._api_server is not None:
            return

        host = host or self.bind_host
        port = port or (int(os.environ.get("PORT", "8000")) + 1)

        # Setup basic logging
        logger = logging.getLogger("GameDocServer.API")
        logger.setLevel(logging.INFO)

        docs_root = str(self.docs_root)

        class _APIHandler(http.server.BaseHTTPRequestHandler):
            def _set_cors_headers(self_inner, status=200, extra_headers=None):
                self_inner.send_response(status)
                self_inner.send_header("Content-Type", "application/json")
                self_inner.send_header("Access-Control-Allow-Origin", "*")
                self_inner.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, HEAD")
                self_inner.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
                if extra_headers:
                    for k, v in extra_headers.items():
                        self_inner.send_header(k, v)
                self_inner.end_headers()

            def do_OPTIONS(self_inner):
                self_inner._set_cors_headers(status=204)

            def do_HEAD(self_inner):
                self_inner._set_cors_headers(status=200)

            def do_GET(self_inner):
                parsed = urllib.parse.urlparse(self_inner.path)
                path = parsed.path
                req_id = str(uuid.uuid4())

                try:
                    if path == "/.well-known/mcp.json":
                        base = f"http://{host}:{port}"
                        body = {"name": self.name, "base_url": base, "tools": list(self._tool_schemas.keys())}
                        self_inner._set_cors_headers(200)
                        self_inner.wfile.write(_json.dumps(body).encode("utf-8"))
                        return

                    if path == "/healthz":
                        body = {"status": "ok", "request_id": req_id}
                        self_inner._set_cors_headers(200)
                        self_inner.wfile.write(_json.dumps(body).encode("utf-8"))
                        return

                    if path == "/readyz":
                        # For now ready when server is up; indexing may be async
                        body = {"status": "ok", "indexed": bool(self.indexer is not None), "request_id": req_id}
                        self_inner._set_cors_headers(200)
                        self_inner.wfile.write(_json.dumps(body).encode("utf-8"))
                        return

                    if path == "/tools":
                        body = {"tools": self._tool_schemas}
                        self_inner._set_cors_headers(200)
                        self_inner.wfile.write(_json.dumps(body).encode("utf-8"))
                        return

                    # Fallback: 404
                    body = {"error": "not_found", "request_id": req_id}
                    self_inner._set_cors_headers(404)
                    self_inner.wfile.write(_json.dumps(body).encode("utf-8"))
                except Exception as e:
                    logger.exception("GET handler error")
                    body = {"error": "internal", "message": str(e), "request_id": req_id}
                    self_inner._set_cors_headers(500)
                    self_inner.wfile.write(_json.dumps(body).encode("utf-8"))

            def do_POST(self_inner):
                parsed = urllib.parse.urlparse(self_inner.path)
                path = parsed.path
                req_id = str(uuid.uuid4())
                logger.info("Request %s %s %s", req_id, self_inner.command, path)

                # Expect /tools/<tool_name>
                parts = path.strip("/").split("/")
                if len(parts) != 2 or parts[0] != "tools":
                    body = {"error": "not_found", "request_id": req_id}
                    self_inner._set_cors_headers(404)
                    self_inner.wfile.write(_json.dumps(body).encode("utf-8"))
                    return

                tool_name = parts[1]
                if tool_name not in self._tool_schemas:
                    body = {"error": "unknown_tool", "request_id": req_id}
                    self_inner._set_cors_headers(404)
                    self_inner.wfile.write(_json.dumps(body).encode("utf-8"))
                    return

                # Read JSON body
                length = int(self_inner.headers.get("Content-Length", "0"))
                if length == 0:
                    body = {"error": "empty_body", "request_id": req_id}
                    self_inner._set_cors_headers(400)
                    self_inner.wfile.write(_json.dumps(body).encode("utf-8"))
                    return

                raw = self_inner.rfile.read(length)
                try:
                    payload = _json.loads(raw.decode("utf-8"))
                except Exception:
                    body = {"error": "invalid_json", "request_id": req_id}
                    self_inner._set_cors_headers(400)
                    self_inner.wfile.write(_json.dumps(body).encode("utf-8"))
                    return

                # Map payload keys to local method parameter names
                try:
                    result = None
                    if tool_name == "list_files":
                        directory = payload.get("path") or payload.get("directory") or ""
                        result = {"files": self.list_files(directory)}

                    elif tool_name == "read_document":
                        fp = payload.get("path") or payload.get("file_path") or payload.get("file")
                        if not fp:
                            raise ValueError("missing path")
                        txt = self.read_document(fp)
                        result = {"text": txt, "mime": "text/markdown"}

                    elif tool_name == "search_knowledge_base":
                        q = payload.get("q") or payload.get("query")
                        top_k = int(payload.get("top_k", payload.get("n", 5)))
                        if not q:
                            raise ValueError("missing q")
                        res = self.search_knowledge_base(q, n=top_k)
                        # Normalize results to include path/title/snippet when possible
                        normalized = []
                        for r in res:
                            md = r.get("metadata") or {}
                            src = md.get("source") or md.get("path") or None
                            snippet = r.get("document")
                            normalized.append({"path": src, "score": r.get("distance"), "snippet": snippet, "metadata": md})
                        result = {"results": normalized}

                    elif tool_name == "read_binary":
                        fp = payload.get("path") or payload.get("file_path") or payload.get("file")
                        if not fp:
                            raise ValueError("missing path")
                        prefer = payload.get("prefer")
                        max_inline = payload.get("max_inline_bytes")
                        resp = self.read_binary(fp, max_inline_bytes=max_inline if max_inline is not None else 5 * 1024 * 1024)
                        # honor prefer
                        if prefer == "redirect" and resp.get("method") == "inline":
                            # force http adapter
                            resp = self.read_binary(fp, max_inline_bytes=0)
                        if prefer == "base64" and resp.get("method") == "http":
                            # try to inline
                            resp = self.read_binary(fp, max_inline_bytes=10 * 1024 * 1024)
                        result = resp

                    elif tool_name == "summarize_context":
                        scope = payload.get("scope")
                        target = payload.get("target")
                        if not scope or not target:
                            raise ValueError("missing scope/target")
                        summary = self.summarize_context(scope, target)
                        result = {"summary": summary}

                    else:
                        raise NotImplementedError("Tool not implemented in HTTP API")

                    body = {"request_id": req_id, "result": result}
                    self_inner._set_cors_headers(200)
                    self_inner.wfile.write(_json.dumps(body).encode("utf-8"))
                except FileNotFoundError as e:
                    body = {"error": "not_found", "message": str(e), "request_id": req_id}
                    self_inner._set_cors_headers(404)
                    self_inner.wfile.write(_json.dumps(body).encode("utf-8"))
                except PermissionError as e:
                    body = {"error": "forbidden", "message": str(e), "request_id": req_id}
                    self_inner._set_cors_headers(403)
                    self_inner.wfile.write(_json.dumps(body).encode("utf-8"))
                except ValueError as e:
                    body = {"error": "bad_request", "message": str(e), "request_id": req_id}
                    self_inner._set_cors_headers(400)
                    self_inner.wfile.write(_json.dumps(body).encode("utf-8"))
                except Exception as e:
                    logger.exception("Tool invocation failed")
                    body = {"error": "internal", "message": str(e), "request_id": req_id}
                    self_inner._set_cors_headers(500)
                    self_inner.wfile.write(_json.dumps(body).encode("utf-8"))

            def log_message(self_inner, format, *args):
                # route to logger
                logger.info(format % args)

        class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True

        srv = _ThreadingHTTPServer((host, port), _APIHandler)
        self._api_server = srv

        def _serve_api():
            try:
                srv.serve_forever()
            finally:
                try:
                    srv.server_close()
                except Exception:
                    pass

        t = threading.Thread(target=_serve_api, daemon=True)
        t.start()
        self._api_thread = t
        logging.getLogger("GameDocServer.API").info("HTTP API started at http://%s:%s", host, port)


async def start_server():
    svc = GameDocServer()
    svc.register_tools()
    print("GameDocServer initialized. Tools registered.")


if __name__ == "__main__":
    # Provide a small, non-sensitive discovery payload at the base URL so humans
    # and MCP-capable clients can discover the service without guessing routes.
    from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

    class _RootHandler(BaseHTTPRequestHandler):
        def _send_json(self, data, status=200):
            payload = json.dumps(data, indent=2).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            # Allow simple browser checks from local pages
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/":
                manifest = {
                    "name": "librarian_mcp",
                    "version": "0.1.0",
                    "description": "Lightweight MCP server for document lookup",
                    "endpoints": {
                        "manifest": " /.well-known/mcp.json",
                        "health": "/healthz",
                        "tools": "/tools"
                    },
                    "tools": ["list_files", "read_document", "search_knowledge_base", "read_binary"]
                }
                self._send_json(manifest)
                return

            if path == "/.well-known/mcp.json":
                # Minimal MCP-style manifest (non-sensitive)
                mcp_manifest = {
                    "name": "librarian_mcp",
                    "host": self.server.server_address[0],
                    "port": self.server.server_address[1],
                    "tls": False,
                    "tools": [
                        {"name": "list_files", "description": "List files under docs root"},
                        {"name": "read_document", "description": "Return document text"},
                        {"name": "search_knowledge_base", "description": "Semantic search for docs"},
                        {"name": "read_binary", "description": "Return binary (inline base64 or HTTP fallback)"}
                    ]
                }
                self._send_json(mcp_manifest)
                return

            if path == "/healthz":
                self._send_json({"status": "ok"})
                return

            # Default: 404
            self.send_response(404)
            self.end_headers()

        def log_message(self, format, *args):
            return

    host = os.environ.get("BIND_HOST") or "127.0.0.1"
    try:
        port = int(os.environ.get("PORT", "8000"))
    except Exception:
        port = 8000

    server = ThreadingHTTPServer((host, port), _RootHandler)
    print(f"Serving discovery endpoints on http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down")
        try:
            server.server_close()
        except Exception:
            pass

    # End of module when running as a discovery HTTP server.
