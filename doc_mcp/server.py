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

        docs_root precedence: constructor arg > `DOCS_ROOT` env var > './docs'
        """
        import os

        self.name = name
        self.server = None
        if FastMCP is not None:
            try:
                self.server = FastMCP(name=self.name)
            except Exception:
                self.server = None

        env_root = os.environ.get("DOCS_ROOT")
        chosen = docs_root or env_root or "./docs"
        self.docs_root = Path(chosen).resolve()

        # DocIndexer will be created lazily to avoid requiring chromadb/ollama at import time
        self.indexer = None

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
                self.indexer = DocIndexer(persist_directory="./chroma_db", collection_name="game_docs", docs_root=str(self.docs_root))
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
        elif hasattr(self.server, "add_tool"):
            self.server.add_tool("list_files", self.list_files)
            self.server.add_tool("read_document", self.read_document)
            self.server.add_tool("search_knowledge_base", self.search_knowledge_base)
            self.server.add_tool("summarize_context", self.summarize_context)
        else:
            # Best-effort: attach as attributes
            setattr(self.server, "list_files", self.list_files)
            setattr(self.server, "read_document", self.read_document)
            setattr(self.server, "search_knowledge_base", self.search_knowledge_base)
            setattr(self.server, "summarize_context", self.summarize_context)


async def start_server():
    svc = GameDocServer()
    svc.register_tools()
    print("GameDocServer initialized. Tools registered.")

    if svc.server is not None and hasattr(svc.server, "start"):
        # If FastMCP exposes an async start, call it
        maybe = svc.server.start()
        if asyncio.iscoroutine(maybe):
            await maybe
    else:
        # Keep the process alive to allow interactive use via imported instance
        print("Server running in local mode. Use the GameDocServer instance to call tools.")
        while True:
            await asyncio.sleep(3600)


def main():
    asyncio.run(start_server())


if __name__ == "__main__":
    main()
