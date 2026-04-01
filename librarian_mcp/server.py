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
import sqlite3
import base64
import mimetypes
import threading
import queue
import http.server
import socketserver
import socket
import urllib.parse
import urllib.request
import os
import json as _json
import uuid
import logging
import hashlib
import math
import time
import importlib.util
from collections import OrderedDict

FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None and importlib.util.find_spec("uvicorn") is not None

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


def _call_ollama_generate(model: str, prompt: str, keep_alive: str = None) -> str:
    """Generate text with Ollama.

    Tries the `ollama` python package, otherwise raises an informative error. CLI
    fallbacks are intentionally omitted because CLI args vary; if you prefer CLI,
    adjust this function to call `ollama run` or similar.
    """
    try:
        import ollama  # type: ignore
    except Exception:
        raise RuntimeError("ollama python package not available; install it or adjust _call_ollama_generate to use the CLI")

    def _normalize_response(resp: Any) -> str:
        """Convert multiple ollama client response shapes to plain text."""
        if resp is None:
            return ""

        # Dict-style responses
        if isinstance(resp, dict):
            msg = resp.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
            if isinstance(resp.get("response"), str):
                return resp["response"]
            if isinstance(resp.get("content"), str):
                return resp["content"]
            return str(resp)

        # Object-style responses (e.g., ChatResponse)
        message = getattr(resp, "message", None)
        content = getattr(message, "content", None)
        if isinstance(content, str):
            return content
        response_text = getattr(resp, "response", None)
        if isinstance(response_text, str):
            return response_text
        content_text = getattr(resp, "content", None)
        if isinstance(content_text, str):
            return content_text

        return str(resp)

    def _call_with_optional_kwargs(fn, **kwargs):
        """Call client method with best-effort kwargs compatibility."""
        try:
            return fn(**kwargs)
        except TypeError:
            # Older clients may not support keep_alive.
            kwargs.pop("keep_alive", None)
            return fn(**kwargs)

    # Attempt a few common client method names
    if hasattr(ollama, "completion"):
        return _normalize_response(
            _call_with_optional_kwargs(
                ollama.completion,
                model=model,
                prompt=prompt,
                keep_alive=keep_alive,
            )
        )
    if hasattr(ollama, "chat"):
        return _normalize_response(
            _call_with_optional_kwargs(
                ollama.chat,
                model=model,
                messages=[{"role": "user", "content": prompt}],
                keep_alive=keep_alive,
            )
        )
    if hasattr(ollama, "run"):
        return _normalize_response(ollama.run(model, prompt))

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

        # Public-facing URL override (useful when container port is remapped)
        # - PUBLIC_URL: full base url, e.g. https://example.com:8001
        # - PUBLIC_HOST + PUBLIC_PORT: host/port pair to build http://host:port
        pub = os.environ.get("PUBLIC_URL")
        if pub:
            self.public_url = pub
        else:
            pub_h = os.environ.get("PUBLIC_HOST")
            pub_p = os.environ.get("PUBLIC_PORT")
            if pub_h and pub_p:
                self.public_url = f"http://{pub_h}:{pub_p}"
            else:
                self.public_url = None

        # DocIndexer will be created lazily to avoid requiring chromadb/ollama at import time
        self.indexer = None

        # HTTP adapter for serving large files (started on demand)
        self._http_server = None
        self._http_thread = None
        self._http_port = None
        # HTTP API server (for LLM-friendly JSON POST tool API)
        self._api_server = None
        self._api_thread = None
        self._api_impl = None
        # Simple tool schema registry used by the HTTP API
        self._tool_schemas = {
            "list_files": {"input": {"directory": "string", "limit": "int?", "offset": "int?"}, "output": {"files": "list"}},
            "read_document": {"input": {"file_path": "string"}, "output": {"text": "string", "mime": "string?"}},
            "search_knowledge_base": {"input": {"q": "string", "top_k": "int?"}, "output": {"results": "list"}},
            "read_binary": {"input": {"file_path": "string", "prefer": "string?", "max_inline_bytes": "int?"}, "output": {"method": "string", "data_b64": "string?", "url": "string?", "content_type": "string", "size": "int"}},
            "summarize_context": {"input": {"scope": "string", "target": "string"}, "output": {"summary": "string"}},
        }

        # Ollama runtime tuning (model residency and startup warmup).
        self.ollama_model = os.environ.get("OLLAMA_MODEL") or "llama3"
        self.ollama_keep_alive = os.environ.get("OLLAMA_KEEP_ALIVE") or "30m"
        prewarm_raw = (os.environ.get("OLLAMA_PREWARM") or "1").strip().lower()
        self.ollama_prewarm_enabled = prewarm_raw not in ("0", "false", "no", "off")
        try:
            self.ollama_prewarm_timeout_sec = int(os.environ.get("OLLAMA_PREWARM_TIMEOUT_SEC", "12"))
        except Exception:
            self.ollama_prewarm_timeout_sec = 12
        self._ollama_prewarm_started = False

        # Summary cache settings (phase 1: in-memory + in-flight dedupe).
        cache_enabled_raw = (os.environ.get("SUMMARY_CACHE_ENABLED") or "1").strip().lower()
        self.summary_cache_enabled = cache_enabled_raw not in ("0", "false", "no", "off")
        try:
            self.summary_cache_ttl_sec = int(os.environ.get("SUMMARY_CACHE_TTL_SEC", "86400"))
        except Exception:
            self.summary_cache_ttl_sec = 86400
        try:
            self.summary_cache_max_entries = int(os.environ.get("SUMMARY_CACHE_MAX_ENTRIES", "512"))
        except Exception:
            self.summary_cache_max_entries = 512

        # Persisted summary depth and live depth controls.
        self.live_summary_depth = (os.environ.get("SUMMARY_LIVE_DEPTH") or "concise").strip().lower()
        if self.live_summary_depth not in ("concise", "detailed"):
            self.live_summary_depth = "concise"
        self.persisted_summary_depth = (os.environ.get("SUMMARY_PERSISTED_DEPTH") or "detailed").strip().lower()
        if self.persisted_summary_depth not in ("concise", "detailed"):
            self.persisted_summary_depth = "detailed"

        # Prompt version lets us invalidate stale persisted summaries after prompt edits.
        self.summary_prompt_version = os.environ.get("SUMMARY_PROMPT_VERSION") or "v1"

        self._summary_cache_lock = threading.RLock()
        self._summary_mem_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._summary_inflight: Dict[str, threading.Event] = {}

        # Phase 2: persisted SQLite summary store.
        self._data_root = Path(self.data_path).resolve()
        self._summary_store_path = self._data_root / "summary_cache.sqlite3"

        # Queue + workers for pre-caching and refresh-on-edit.
        self._summary_task_queue: "queue.PriorityQueue[Any]" = queue.PriorityQueue()
        self._summary_task_keys = set()
        self._summary_task_keys_lock = threading.Lock()
        self._summary_task_seq = 0
        self._summary_worker_thread = None
        self._summary_monitor_thread = None
        self._stop_event = threading.Event()
        self._summary_background_started = False

        # Runtime stats for cache observability.
        self._cache_stats_lock = threading.Lock()
        self._cache_hits_mem = 0
        self._cache_hits_sqlite = 0
        self._cache_misses = 0
        self._cache_computes = 0
        self._cache_errors = 0
        self._worker_tasks_completed = 0
        self._worker_tasks_failed = 0
        self._worker_last_success_ts = None
        self._worker_last_error = None
        self._monitor_last_scan_ts = None
        self._monitor_last_changes = 0

        try:
            self.summary_worker_delay_ms = int(os.environ.get("SUMMARY_WORKER_DELAY_MS", "150"))
        except Exception:
            self.summary_worker_delay_ms = 150
        try:
            self.summary_monitor_interval_sec = int(os.environ.get("SUMMARY_MONITOR_INTERVAL_SEC", "15"))
        except Exception:
            self.summary_monitor_interval_sec = 15
        try:
            self.summary_status_log_interval_sec = int(os.environ.get("SUMMARY_STATUS_LOG_INTERVAL_SEC", "30"))
        except Exception:
            self.summary_status_log_interval_sec = 30
        try:
            self.summary_precache_max_files = int(os.environ.get("SUMMARY_PRECACHE_MAX_FILES", "2000"))
        except Exception:
            self.summary_precache_max_files = 2000
        try:
            self.summary_precache_max_dirs = int(os.environ.get("SUMMARY_PRECACHE_MAX_DIRS", "500"))
        except Exception:
            self.summary_precache_max_dirs = 500

        self._watched_file_sigs: Dict[str, str] = {}

        self._init_summary_store()

    def _sqlite_row_count(self) -> int:
        try:
            with sqlite3.connect(str(self._summary_store_path)) as conn:
                cur = conn.cursor()
                cur.execute("SELECT COUNT(*) FROM summaries")
                row = cur.fetchone()
                return int(row[0] if row else 0)
        except Exception:
            return 0

    def get_cache_status(self) -> Dict[str, Any]:
        now = time.time()
        with self._cache_stats_lock:
            stats = {
                "cache_hits_mem": self._cache_hits_mem,
                "cache_hits_sqlite": self._cache_hits_sqlite,
                "cache_misses": self._cache_misses,
                "cache_computes": self._cache_computes,
                "cache_errors": self._cache_errors,
                "worker_tasks_completed": self._worker_tasks_completed,
                "worker_tasks_failed": self._worker_tasks_failed,
                "worker_last_success_ts": self._worker_last_success_ts,
                "worker_last_error": self._worker_last_error,
                "monitor_last_scan_ts": self._monitor_last_scan_ts,
                "monitor_last_changes": self._monitor_last_changes,
            }

        with self._summary_cache_lock:
            mem_entries = len(self._summary_mem_cache)
            inflight = len(self._summary_inflight)
        with self._summary_task_keys_lock:
            queue_unique_keys = len(self._summary_task_keys)
        queue_depth = self._summary_task_queue.qsize()

        status = {
            "status": "ok",
            "timestamp": now,
            "cache_enabled": self.summary_cache_enabled,
            "summary_cache_ttl_sec": self.summary_cache_ttl_sec,
            "summary_cache_max_entries": self.summary_cache_max_entries,
            "mem_entries": mem_entries,
            "sqlite_rows": self._sqlite_row_count(),
            "queue_depth": queue_depth,
            "queue_unique_keys": queue_unique_keys,
            "inflight": inflight,
        }
        status.update(stats)

        total_requests = status["cache_hits_mem"] + status["cache_hits_sqlite"] + status["cache_misses"]
        if total_requests > 0:
            status["cache_hit_ratio"] = round((status["cache_hits_mem"] + status["cache_hits_sqlite"]) / total_requests, 4)
        else:
            status["cache_hit_ratio"] = 0.0
        return status

    def _log_cache_status(self) -> None:
        s = self.get_cache_status()
        logging.getLogger("GameDocServer.Cache").warning(
            "cache_status mem=%d sqlite=%d q=%d inflight=%d hit_mem=%d hit_sqlite=%d miss=%d computes=%d failed=%d hit_ratio=%.4f",
            s.get("mem_entries", 0),
            s.get("sqlite_rows", 0),
            s.get("queue_depth", 0),
            s.get("inflight", 0),
            s.get("cache_hits_mem", 0),
            s.get("cache_hits_sqlite", 0),
            s.get("cache_misses", 0),
            s.get("cache_computes", 0),
            s.get("worker_tasks_failed", 0),
            s.get("cache_hit_ratio", 0.0),
        )

    def _start_ollama_prewarm(self) -> None:
        """Best-effort background prewarm to reduce first summarize latency."""
        if not self.ollama_prewarm_enabled:
            return
        if self._ollama_prewarm_started:
            return
        self._ollama_prewarm_started = True

        t = threading.Thread(target=self._run_ollama_prewarm, daemon=True)
        t.start()

    def _run_ollama_prewarm(self) -> None:
        logger = logging.getLogger("GameDocServer.Ollama")
        base = (os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        url = f"{base}/api/chat"
        payload = {
            "model": self.ollama_model,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "keep_alive": self.ollama_keep_alive,
        }
        req = urllib.request.Request(
            url,
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.ollama_prewarm_timeout_sec) as resp:
                if resp.status >= 400:
                    logger.warning("Ollama prewarm failed with status %s", resp.status)
                else:
                    logger.info("Ollama prewarm completed for model '%s' (keep_alive=%s)", self.ollama_model, self.ollama_keep_alive)
        except Exception as e:
            logger.warning("Ollama prewarm skipped/failed: %s", e)

    def _init_summary_store(self) -> None:
        if not self.summary_cache_enabled:
            return
        try:
            self._data_root.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self._summary_store_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS summaries (
                        cache_key TEXT PRIMARY KEY,
                        scope TEXT NOT NULL,
                        target TEXT NOT NULL,
                        depth TEXT NOT NULL,
                        model TEXT NOT NULL,
                        content_sig TEXT NOT NULL,
                        prompt_version TEXT NOT NULL,
                        summary TEXT NOT NULL,
                        created_at REAL NOT NULL,
                        expires_at REAL NOT NULL
                    )
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_summaries_expires_at ON summaries(expires_at)")
                conn.commit()
        except Exception as e:
            logging.getLogger("GameDocServer.Cache").warning("Summary SQLite init failed; cache store disabled: %s", e)

    def _sqlite_get_summary(self, cache_key: str, now_ts: float) -> str:
        if not self.summary_cache_enabled:
            return None
        try:
            with sqlite3.connect(str(self._summary_store_path)) as conn:
                row = conn.execute(
                    "SELECT summary, expires_at FROM summaries WHERE cache_key = ?",
                    (cache_key,),
                ).fetchone()
                if not row:
                    return None
                summary, expires_at = row
                if expires_at < now_ts:
                    conn.execute("DELETE FROM summaries WHERE cache_key = ?", (cache_key,))
                    conn.commit()
                    return None
                return summary
        except Exception:
            return None

    def _sqlite_put_summary(
        self,
        cache_key: str,
        scope: str,
        target: str,
        depth: str,
        content_sig: str,
        summary: str,
        now_ts: float,
    ) -> None:
        if not self.summary_cache_enabled:
            return
        expires_at = now_ts + self.summary_cache_ttl_sec
        try:
            with sqlite3.connect(str(self._summary_store_path)) as conn:
                conn.execute(
                    """
                    INSERT INTO summaries(
                        cache_key, scope, target, depth, model, content_sig, prompt_version,
                        summary, created_at, expires_at
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(cache_key) DO UPDATE SET
                        scope=excluded.scope,
                        target=excluded.target,
                        depth=excluded.depth,
                        model=excluded.model,
                        content_sig=excluded.content_sig,
                        prompt_version=excluded.prompt_version,
                        summary=excluded.summary,
                        created_at=excluded.created_at,
                        expires_at=excluded.expires_at
                    """,
                    (
                        cache_key,
                        scope,
                        target,
                        depth,
                        self.ollama_model,
                        content_sig,
                        self.summary_prompt_version,
                        summary,
                        now_ts,
                        expires_at,
                    ),
                )
                conn.commit()
        except Exception as e:
            logging.getLogger("GameDocServer.Cache").warning("Summary SQLite write failed: %s", e)

    def _mem_get_summary(self, cache_key: str, now_ts: float) -> str:
        with self._summary_cache_lock:
            entry = self._summary_mem_cache.get(cache_key)
            if not entry:
                return None
            if entry.get("expires_at", 0) < now_ts:
                self._summary_mem_cache.pop(cache_key, None)
                return None
            self._summary_mem_cache.move_to_end(cache_key)
            return entry.get("summary")

    def _mem_put_summary(self, cache_key: str, summary: str, now_ts: float) -> None:
        with self._summary_cache_lock:
            self._summary_mem_cache[cache_key] = {
                "summary": summary,
                "expires_at": now_ts + self.summary_cache_ttl_sec,
            }
            self._summary_mem_cache.move_to_end(cache_key)
            while len(self._summary_mem_cache) > self.summary_cache_max_entries:
                self._summary_mem_cache.popitem(last=False)

    def _cache_key(self, scope: str, target: str, depth: str, content_sig: str) -> str:
        raw = "|".join(
            [
                self.summary_prompt_version,
                self.ollama_model,
                self.ollama_keep_alive,
                scope,
                target,
                depth,
                content_sig,
            ]
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _truncate_tokens(self, text: str, max_tokens: int) -> str:
        tokens = text.split()
        if len(tokens) <= max_tokens:
            return text
        return " ".join(tokens[:max_tokens])

    def _file_signature(self, rel_path: str) -> str:
        p = _resolve_within_root(self.docs_root, rel_path)
        st = p.stat()
        return f"{rel_path}|{st.st_size}|{st.st_mtime_ns}"

    def _directory_signature(self, rel_dir: str) -> str:
        root = _resolve_within_root(self.docs_root, rel_dir)
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Directory not found: {rel_dir}")
        lines = []
        for f in root.rglob("*"):
            if not f.is_file():
                continue
            if f.suffix.lower() not in (".md", ".txt"):
                continue
            rel = str(f.relative_to(self.docs_root))
            st = f.stat()
            lines.append(f"{rel}|{st.st_size}|{st.st_mtime_ns}")
        lines.sort()
        manifest = "\n".join(lines)
        return hashlib.sha256(manifest.encode("utf-8")).hexdigest()

    def _summary_content_signature(self, scope: str, target: str) -> str:
        if scope == "file":
            return self._file_signature(target)
        if scope == "directory":
            return self._directory_signature(target)
        if scope == "snippet":
            return hashlib.sha256(target.encode("utf-8")).hexdigest()
        raise ValueError("Unknown scope. Must be one of: 'file', 'directory', 'snippet'")

    def _build_summary_prompt(self, scope: str, target: str, depth: str) -> str:
        if scope == "file":
            text = self.read_document(target)
            if depth == "concise":
                truncated = self._truncate_tokens(text, 900)
                return (
                    "Create a concise summary with 5-8 bullet points. Focus on intent, key capabilities, constraints, "
                    "and important interfaces. Keep it compact and practical.\n\n"
                    + truncated
                )
            truncated = self._truncate_tokens(text, 3000)
            return (
                "Create a detailed structured summary with sections: Overview, Key Components, Data Flows, "
                "Constraints, and Risks. Include concrete details and dependencies when present.\n\n"
                + truncated
            )

        if scope == "directory":
            files = self.list_files(target)
            doc_files = [p for p in files if p.lower().endswith((".md", ".txt"))]
            if not doc_files:
                raise FileNotFoundError(f"No text docs found in directory: {target}")

            if depth == "concise":
                selected = doc_files[: min(len(doc_files), 20)]
                chunks = []
                for rel in selected:
                    try:
                        t = self.read_document(rel)
                    except Exception:
                        continue
                    chunks.append(f"File: {rel}\n{self._truncate_tokens(t, 300)}")
                return (
                    "Provide a concise directory-level summary in 8-12 bullets. Mention important files, "
                    "capabilities, and high-risk gaps.\n\n"
                    + "\n\n".join(chunks)
                )

            selected = doc_files[: min(len(doc_files), 80)]
            mini_summaries = []
            for rel in selected:
                try:
                    t = self.read_document(rel)
                except Exception:
                    continue
                snippet = self._truncate_tokens(t, 1200)
                mini_prompt = (
                    "Summarize this file in 4-6 bullet points, emphasizing architecture, inputs/outputs, "
                    "and constraints.\n\n"
                    + snippet
                )
                try:
                    s = _call_ollama_generate(self.ollama_model, mini_prompt, keep_alive=self.ollama_keep_alive)
                except Exception as e:
                    s = f"(mini-summary failed: {e})"
                mini_summaries.append(f"File: {rel}\n{s}")

            return (
                "You are given per-file summaries from one documentation directory. Produce a detailed technical "
                "overview with sections: System Purpose, Components, Data/Control Flow, Operational Constraints, "
                "Known Risks, and Suggested Follow-ups.\n\n"
                + "\n\n".join(mini_summaries)
            )

        if scope == "snippet":
            if depth == "concise":
                return "Summarize this text in 3-5 bullets:\n\n" + self._truncate_tokens(target, 600)
            return "Provide a detailed summary with key insights and implications:\n\n" + self._truncate_tokens(target, 1800)

        raise ValueError("Unknown scope. Must be one of: 'file', 'directory', 'snippet'")

    def _compute_summary(self, scope: str, target: str, depth: str) -> str:
        prompt = self._build_summary_prompt(scope, target, depth)
        return _call_ollama_generate(self.ollama_model, prompt, keep_alive=self.ollama_keep_alive)

    def _get_or_compute_summary(self, scope: str, target: str, depth: str) -> str:
        content_sig = self._summary_content_signature(scope, target)
        cache_key = self._cache_key(scope, target, depth, content_sig)
        now_ts = time.time()

        if self.summary_cache_enabled:
            mem = self._mem_get_summary(cache_key, now_ts)
            if mem is not None:
                with self._cache_stats_lock:
                    self._cache_hits_mem += 1
                return mem
            disk = self._sqlite_get_summary(cache_key, now_ts)
            if disk is not None:
                with self._cache_stats_lock:
                    self._cache_hits_sqlite += 1
                self._mem_put_summary(cache_key, disk, now_ts)
                return disk

        with self._cache_stats_lock:
            self._cache_misses += 1

        waiter = None
        with self._summary_cache_lock:
            waiter = self._summary_inflight.get(cache_key)
            if waiter is None:
                waiter = threading.Event()
                self._summary_inflight[cache_key] = waiter
                is_owner = True
            else:
                is_owner = False

        if not is_owner:
            waiter.wait(timeout=180)
            now_ts = time.time()
            mem = self._mem_get_summary(cache_key, now_ts)
            if mem is not None:
                return mem
            disk = self._sqlite_get_summary(cache_key, now_ts)
            if disk is not None:
                self._mem_put_summary(cache_key, disk, now_ts)
                return disk
            # If the owner failed, fall back to direct compute.
            return self._compute_summary(scope, target, depth)

        try:
            summary = self._compute_summary(scope, target, depth)
            now_ts = time.time()
            with self._cache_stats_lock:
                self._cache_computes += 1
            if self.summary_cache_enabled:
                self._mem_put_summary(cache_key, summary, now_ts)
                self._sqlite_put_summary(cache_key, scope, target, depth, content_sig, summary, now_ts)
            return summary
        finally:
            with self._summary_cache_lock:
                ev = self._summary_inflight.pop(cache_key, None)
                if ev is not None:
                    ev.set()

    def _enqueue_summary_task(self, scope: str, target: str, depth: str, reason: str) -> None:
        priority_map = {
            "live_request": 0,
            "file_changed": 1,
            "dir_refresh": 2,
            "root_refresh": 3,
            "dir_refresh_removed": 3,
            "root_refresh_removed": 3,
            "startup_precache": 10,
        }
        prio = priority_map.get(reason, 5)
        task_key = f"{scope}|{target}|{depth}"
        with self._summary_task_keys_lock:
            if task_key in self._summary_task_keys:
                return
            self._summary_task_keys.add(task_key)
            self._summary_task_seq += 1
            seq = self._summary_task_seq
        self._summary_task_queue.put(
            (
                prio,
                seq,
                {
                    "scope": scope,
                    "target": target,
                    "depth": depth,
                    "reason": reason,
                },
            )
        )

    def _summary_worker_loop(self) -> None:
        logger = logging.getLogger("GameDocServer.SummaryWorker")
        while not self._stop_event.is_set():
            try:
                _prio, _seq, task = self._summary_task_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            scope = task.get("scope")
            target = task.get("target")
            depth = task.get("depth")
            reason = task.get("reason")
            task_key = f"{scope}|{target}|{depth}"
            try:
                _ = self._get_or_compute_summary(scope, target, depth)
                logger.info("Summary task complete: %s (%s) [%s]", target, scope, reason)
                with self._cache_stats_lock:
                    self._worker_tasks_completed += 1
                    self._worker_last_success_ts = time.time()
            except Exception as e:
                logger.warning("Summary task failed: %s (%s) [%s]: %s", target, scope, reason, e)
                with self._cache_stats_lock:
                    self._worker_tasks_failed += 1
                    self._cache_errors += 1
                    self._worker_last_error = str(e)
            finally:
                with self._summary_task_keys_lock:
                    self._summary_task_keys.discard(task_key)
                try:
                    self._summary_task_queue.task_done()
                except Exception:
                    pass
                if self.summary_worker_delay_ms > 0:
                    time.sleep(self.summary_worker_delay_ms / 1000.0)

    def _all_summary_directories(self) -> List[str]:
        dirs = [""]
        seen = {""}
        count = 0
        for p in self.docs_root.rglob("*"):
            if not p.is_dir():
                continue
            rel = str(p.relative_to(self.docs_root))
            if rel in seen:
                continue
            seen.add(rel)
            dirs.append(rel)
            count += 1
            if count >= self.summary_precache_max_dirs:
                break
        return dirs

    def _all_summary_files(self) -> List[str]:
        out = []
        count = 0
        for p in self.docs_root.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in (".md", ".txt"):
                continue
            out.append(str(p.relative_to(self.docs_root)))
            count += 1
            if count >= self.summary_precache_max_files:
                break
        return out

    def _seed_precache_queue(self) -> None:
        for d in self._all_summary_directories():
            self._enqueue_summary_task("directory", d, self.persisted_summary_depth, "startup_precache")
        for f in self._all_summary_files():
            self._enqueue_summary_task("file", f, self.persisted_summary_depth, "startup_precache")

    def _snapshot_file_sigs(self) -> Dict[str, str]:
        sigs = {}
        for f in self._all_summary_files():
            try:
                sigs[f] = self._file_signature(f)
            except Exception:
                continue
        return sigs

    def _summary_monitor_loop(self) -> None:
        logger = logging.getLogger("GameDocServer.SummaryMonitor")
        self._watched_file_sigs = self._snapshot_file_sigs()
        next_status_log = time.time() + max(self.summary_status_log_interval_sec, 5)
        while not self._stop_event.is_set():
            time.sleep(max(self.summary_monitor_interval_sec, 3))
            try:
                current = self._snapshot_file_sigs()
                old_keys = set(self._watched_file_sigs.keys())
                new_keys = set(current.keys())

                changed = []
                for k in (old_keys & new_keys):
                    if self._watched_file_sigs.get(k) != current.get(k):
                        changed.append(k)
                added = list(new_keys - old_keys)
                removed = list(old_keys - new_keys)

                if changed or added or removed:
                    logger.info(
                        "Summary monitor changes: changed=%d added=%d removed=%d",
                        len(changed),
                        len(added),
                        len(removed),
                    )

                for rel in changed + added:
                    self._enqueue_summary_task("file", rel, self.persisted_summary_depth, "file_changed")
                    parent = str(Path(rel).parent)
                    if parent == ".":
                        parent = ""
                    self._enqueue_summary_task("directory", parent, self.persisted_summary_depth, "dir_refresh")
                    self._enqueue_summary_task("directory", "", self.persisted_summary_depth, "root_refresh")

                # Removed files still require parent/root directory refresh.
                for rel in removed:
                    parent = str(Path(rel).parent)
                    if parent == ".":
                        parent = ""
                    self._enqueue_summary_task("directory", parent, self.persisted_summary_depth, "dir_refresh_removed")
                    self._enqueue_summary_task("directory", "", self.persisted_summary_depth, "root_refresh_removed")

                self._watched_file_sigs = current
                with self._cache_stats_lock:
                    self._monitor_last_scan_ts = time.time()
                    self._monitor_last_changes = len(changed) + len(added) + len(removed)

                if time.time() >= next_status_log:
                    self._log_cache_status()
                    next_status_log = time.time() + max(self.summary_status_log_interval_sec, 5)
            except Exception as e:
                logger.warning("Summary monitor loop error: %s", e)
                with self._cache_stats_lock:
                    self._cache_errors += 1

    def _start_summary_background(self) -> None:
        if not self.summary_cache_enabled:
            return
        if self._summary_background_started:
            return
        self._summary_background_started = True

        self._summary_worker_thread = threading.Thread(target=self._summary_worker_loop, daemon=True)
        self._summary_worker_thread.start()

        self._summary_monitor_thread = threading.Thread(target=self._summary_monitor_loop, daemon=True)
        self._summary_monitor_thread.start()

        self._seed_precache_queue()

    def shutdown(self) -> None:
        self._stop_event.set()
        try:
            if self._api_server:
                if hasattr(self._api_server, "shutdown"):
                    self._api_server.shutdown()
                if hasattr(self._api_server, "server_close"):
                    self._api_server.server_close()
        except Exception:
            pass
        try:
            if self._http_server:
                self._http_server.shutdown()
                self._http_server.server_close()
        except Exception:
            pass

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

    def _search_placeholder_deterministic(self, query: str, n: int = 5) -> List[Dict[str, Any]]:
        """Deterministic placeholder fallback search.

        This fallback does not provide semantic retrieval quality. It uses stable
        hash-derived vectors to keep results deterministic in zero-dependency mode
        when the primary indexer is unavailable.
        """

        def embed(text: str) -> List[float]:
            h = hashlib.sha256(text.encode("utf-8")).digest()
            return [float(h[i % len(h)]) / 255.0 for i in range(768)]

        def cosine(a: List[float], b: List[float]) -> float:
            sa = sum(x * x for x in a)
            sb = sum(x * x for x in b)
            if sa == 0 or sb == 0:
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            return dot / (math.sqrt(sa) * math.sqrt(sb))

        q_emb = embed(query)
        candidates: List[Dict[str, Any]] = []
        for file in Path(self.docs_root).rglob("*"):
            if not file.is_file():
                continue
            if file.suffix.lower() not in (".md", ".txt"):
                continue
            try:
                text = file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            CHUNK_SIZE = 1000
            OVERLAP = max(int(CHUNK_SIZE * 0.1), 1)
            step = max(CHUNK_SIZE - OVERLAP, 1)
            for i in range(0, len(text), step):
                chunk = text[i:i + CHUNK_SIZE]
                if not chunk:
                    continue
                emb = embed(chunk)
                score = cosine(q_emb, emb)
                candidates.append({"score": score, "document": chunk, "metadata": {"source": str(file), "offset": i}})

        candidates.sort(key=lambda c: c["score"], reverse=True)
        out = []
        for c in candidates[:n]:
            out.append({"document": c["document"], "metadata": c["metadata"], "distance": 1.0 - c["score"]})
        return out

    def _normalize_search_results(self, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        for r in rows:
            md = r.get("metadata") or {}
            src = md.get("source") or md.get("path") or None
            snippet = r.get("document")
            normalized.append({"path": src, "score": r.get("distance"), "snippet": snippet, "metadata": md})
        return normalized

    def search_knowledge_base(self, query: str, n: int = 5) -> List[Dict[str, Any]]:
        # Lazy-create indexer when first needed. If DocIndexer fails to initialize
        # (migration issues, chroma compatibility, etc.) fall back to a simple
        # in-process deterministic placeholder search.
        if self.indexer is None:
            try:
                # Place persistent DB under configured `data_path` so it can be volume-mounted
                persist_dir = str(Path(self.data_path) / "chroma_db")
                self.indexer = DocIndexer(persist_directory=persist_dir, collection_name="game_docs", docs_root=str(self.docs_root))
            except Exception as e:
                # Log and continue with deterministic placeholder fallback.
                logging.getLogger("GameDocServer").warning(
                    "DocIndexer init failed, using deterministic placeholder fallback: %s",
                    e,
                )
                self.indexer = None

        if self.indexer is not None:
            try:
                return self.indexer.semantic_search(query, n=n)
            except Exception as e:
                logging.getLogger("GameDocServer").exception(
                    "DocIndexer.semantic_search failed, using deterministic placeholder fallback: %s",
                    e,
                )
                # Clear indexer so subsequent calls will rebuild or stay in fallback
                self.indexer = None

        return self._search_placeholder_deterministic(query, n=n)

    def summarize_context(self, scope: str, target: str, depth: str = None) -> str:
        """Summarize content according to `scope`:

        - 'file': `target` is file path; summarize the file.
        - 'directory': `target` is directory path; summarize each file then give feature-level overview.
        - 'snippet': `target` is the text to summarize.
        """
        chosen_depth = (depth or self.live_summary_depth or "concise").strip().lower()
        if chosen_depth not in ("concise", "detailed"):
            chosen_depth = "concise"

        summary = self._get_or_compute_summary(scope, target, chosen_depth)

        # Live requests return concise summaries by default, but queue deeper persisted summaries.
        if chosen_depth == "concise" and self.persisted_summary_depth == "detailed":
            self._enqueue_summary_task(scope, target, "detailed", "live_request")

        return summary

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
    def _invoke_http_tool(self, tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke one HTTP-exposed tool using parity-compatible payload mapping."""
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
            result = {"results": self._normalize_search_results(res)}

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
            depth = payload.get("depth") or self.live_summary_depth
            if not scope or not target:
                raise ValueError("missing scope/target")
            summary = self.summarize_context(scope, target, depth=depth)
            result = {"summary": summary}

        else:
            raise NotImplementedError("Tool not implemented in HTTP API")

        return result

    def _find_available_port(self, host: str) -> int:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((host, 0))
            return int(s.getsockname()[1])
        finally:
            s.close()

    def _start_http_api_legacy(self, host: str, port: int):
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
                        base = self.public_url or f"http://{host}:{port}"
                        body = {"name": self.name, "base_url": base, "public_url": base, "tools": list(self._tool_schemas.keys())}
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

                    if path == "/cache_status":
                        body = self.get_cache_status()
                        body["request_id"] = req_id
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
                    result = self._invoke_http_tool(tool_name, payload)
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
        self._api_impl = "legacy"
        logging.getLogger("GameDocServer.API").info("HTTP API started at http://%s:%s", host, port)

    def _start_http_api_fastapi(self, host: str, port: int):
        if not FASTAPI_AVAILABLE:
            raise RuntimeError("FastAPI backend requested but fastapi/uvicorn are not installed")

        from fastapi import FastAPI, Request  # type: ignore[import-not-found]
        from fastapi.middleware.cors import CORSMiddleware  # type: ignore[import-not-found]
        from fastapi.responses import JSONResponse  # type: ignore[import-not-found]
        import uvicorn  # type: ignore[import-not-found]
        from starlette.exceptions import HTTPException as StarletteHTTPException

        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["GET", "POST", "OPTIONS", "HEAD"],
            allow_headers=["Content-Type", "Accept"],
        )

        logger = logging.getLogger("GameDocServer.API")
        logger.setLevel(logging.INFO)

        @app.exception_handler(StarletteHTTPException)
        async def starlette_exception_handler(_request, exc):
            if exc.status_code == 404:
                return JSONResponse(status_code=404, content={"error": "not_found", "request_id": str(uuid.uuid4())})
            return JSONResponse(status_code=exc.status_code, content={"error": "internal", "message": str(exc.detail), "request_id": str(uuid.uuid4())})

        @app.options("/{path:path}")
        async def options_handler(path: str):
            _ = path
            return JSONResponse(status_code=204, content={})

        @app.get("/.well-known/mcp.json")
        async def discovery():
            base = self.public_url or f"http://{host}:{port}"
            body = {"name": self.name, "base_url": base, "public_url": base, "tools": list(self._tool_schemas.keys())}
            return JSONResponse(content=body)

        @app.get("/healthz")
        async def healthz():
            return JSONResponse(content={"status": "ok", "request_id": str(uuid.uuid4())})

        @app.get("/readyz")
        async def readyz():
            body = {"status": "ok", "indexed": bool(self.indexer is not None), "request_id": str(uuid.uuid4())}
            return JSONResponse(content=body)

        @app.get("/tools")
        async def tools():
            return JSONResponse(content={"tools": self._tool_schemas})

        @app.get("/cache_status")
        async def cache_status():
            body = self.get_cache_status()
            body["request_id"] = str(uuid.uuid4())
            return JSONResponse(content=body)

        @app.post("/tools/{tool_name}")
        async def post_tool(tool_name: str, request: Request):
            req_id = str(uuid.uuid4())
            logger.info("Request %s POST /tools/%s", req_id, tool_name)

            if tool_name not in self._tool_schemas:
                return JSONResponse(status_code=404, content={"error": "unknown_tool", "request_id": req_id})

            try:
                body_raw = await request.body()
                if not body_raw:
                    return JSONResponse(status_code=400, content={"error": "empty_body", "request_id": req_id})
                payload = _json.loads(body_raw.decode("utf-8"))
            except Exception:
                return JSONResponse(status_code=400, content={"error": "invalid_json", "request_id": req_id})

            try:
                result = self._invoke_http_tool(tool_name, payload)
                return JSONResponse(status_code=200, content={"request_id": req_id, "result": result})
            except FileNotFoundError as e:
                return JSONResponse(status_code=404, content={"error": "not_found", "message": str(e), "request_id": req_id})
            except PermissionError as e:
                return JSONResponse(status_code=403, content={"error": "forbidden", "message": str(e), "request_id": req_id})
            except ValueError as e:
                return JSONResponse(status_code=400, content={"error": "bad_request", "message": str(e), "request_id": req_id})
            except Exception as e:
                logger.exception("Tool invocation failed")
                return JSONResponse(status_code=500, content={"error": "internal", "message": str(e), "request_id": req_id})

        config = uvicorn.Config(app=app, host=host, port=port, log_level="warning")
        uv_server = uvicorn.Server(config=config)

        def _serve_api():
            uv_server.run()

        t = threading.Thread(target=_serve_api, daemon=True)
        t.start()
        self._api_thread = t

        # Wait briefly for server startup.
        deadline = time.time() + 5.0
        while time.time() < deadline:
            if getattr(uv_server, "started", False):
                break
            time.sleep(0.01)

        class _UvicornHandle:
            def __init__(self, server_ref, thread_ref, bind_host, bind_port):
                self._server_ref = server_ref
                self._thread_ref = thread_ref
                self.server_address = (bind_host, bind_port)

            def shutdown(self):
                self._server_ref.should_exit = True
                if self._thread_ref is not None:
                    self._thread_ref.join(timeout=5.0)

            def server_close(self):
                return None

        self._api_server = _UvicornHandle(uv_server, t, host, port)
        self._api_impl = "fastapi"
        logging.getLogger("GameDocServer.API").info("HTTP API started at http://%s:%s", host, port)

    def start_http_api(self, host: str = None, port: int = None, impl: str = None):
        """Start HTTP API server with selectable backend implementation.

        Supported implementations:
        - legacy: built-in http.server-based implementation
        - fastapi: FastAPI + uvicorn implementation
        """
        if self._api_server is not None:
            return

        host = host or self.bind_host
        port = port if port is not None else (int(os.environ.get("PORT", "8000")) + 1)
        if int(port) == 0:
            port = self._find_available_port(host)

        api_impl = (impl or os.environ.get("HTTP_API_IMPL") or "legacy").strip().lower()
        if api_impl == "fastapi":
            self._start_http_api_fastapi(host, int(port))
        else:
            self._start_http_api_legacy(host, int(port))

        # Trigger non-blocking Ollama prewarm after API startup.
        self._start_ollama_prewarm()
        self._start_summary_background()


async def start_server():
    svc = GameDocServer()
    svc.register_tools()
    print("GameDocServer initialized. Tools registered.")


if __name__ == "__main__":
    # Start the full GameDocServer and expose the LLM-friendly HTTP API on
    # the configured bind host/port so POST /tools is available where
    # discovery advertises the service.
    import time

    host = os.environ.get("BIND_HOST") or "127.0.0.1"
    try:
        port = int(os.environ.get("PORT", "8000"))
    except Exception:
        port = 8000

    svc = GameDocServer()
    svc.register_tools()

    svc.start_http_api(host=host, port=port)
    print(f"GameDocServer HTTP API started at {svc.public_url or f'http://{host}:{port}'}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down GameDocServer")
        svc.shutdown()
