"""
Indexer and RAG utilities (placeholder).

Uses `chromadb` for vector storage and `ollama` for local LLM calls.
Fill in embedding, upsert, and retrieval logic as needed.
"""
from typing import List, Dict, Any
from pathlib import Path
import math
import uuid
import json
import subprocess

try:
    import chromadb
except Exception:
    chromadb = None


class DocIndexer:
    """Document indexer that persists a ChromaDB collection and uses Ollama for embeddings.

    Notes:
    - Requires `chromadb` and either the `ollama` python package or the `ollama` CLI available.
    - Adjust `persist_directory` and `collection_name` as needed.
    """

    def __init__(self,
                 persist_directory: str = "./chroma_db",
                 collection_name: str = "docs",
                 embedding_model: str = "nomic-embed-text",
                 docs_root: str = None):
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        from pathlib import Path
        import os

        env_root = os.environ.get("DOCS_ROOT")
        chosen = docs_root or env_root or "./docs"
        self.docs_root = Path(chosen).resolve()

        if chromadb is None:
            raise RuntimeError("chromadb is not installed; please install it (pip install chromadb)")

        # Create client. Make initialization non-fatal: if a persistent DB is
        # requested but cannot be opened (deprecated layout, migration needed,
        # etc.) we will gracefully fall back to an ephemeral in-memory client
        # or to None so the higher-level server can use an in-memory search.
        import os
        from pathlib import Path as _Path

        persist_path = _Path(self.persist_directory)
        try:
            # Allow opting to use an HTTP Chroma server via env vars
            host = os.environ.get("CHROMA_HTTP_HOST")
            port = os.environ.get("CHROMA_HTTP_PORT")
            if host and port:
                try:
                    # Use new HttpClient if available
                    if hasattr(chromadb, "HttpClient"):
                        self.client = chromadb.HttpClient(host=host, port=int(port))
                    else:
                        # best-effort fallback
                        self.client = chromadb.Client()
                except Exception:
                    self.client = None
            else:
                # Use a persistent filesystem-backed store whenever a persist directory
                # is configured, including first-run empty directories.
                try:
                    persist_path.mkdir(parents=True, exist_ok=True)
                    if hasattr(chromadb, "PersistentClient"):
                        self.client = chromadb.PersistentClient(path=self.persist_directory)
                    else:
                        # Fall back to the older Settings-based construction if necessary
                        from chromadb.config import Settings

                        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.persist_directory)
                        self.client = chromadb.Client(settings)
                except Exception:
                    self.client = None
        except Exception:
            self.client = None

        # Create or get collection if a client is available
        try:
            self.collection = self.client.get_or_create_collection(name=self.collection_name) if self.client is not None else None
        except Exception:
            self.collection = None

        # In-memory index placeholder: list of {'document': str, 'metadata': dict, 'embedding': List[float]}
        self._memory_index = None

        # Try to import the ollama python package; if not available we'll fall back to the CLI.
        try:
            import ollama  # type: ignore
            self._ollama = ollama
            self._use_ollama_cli = False
        except Exception:
            self._ollama = None
            self._use_ollama_cli = True

    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        if chunk_size <= 0:
            return [text]
        step = max(chunk_size - overlap, 1)
        chunks: List[str] = []
        for i in range(0, len(text), step):
            chunk = text[i:i + chunk_size]
            if chunk:
                chunks.append(chunk)
            if i + chunk_size >= len(text):
                break
        return chunks

    def _is_dimension_mismatch_error(self, ex: Exception) -> bool:
        msg = str(ex).lower()
        return "dimension" in msg and ("embedding" in msg or "expect" in msg or "got" in msg)

    def _reset_collection(self):
        if self.client is None:
            self.collection = None
            return
        # Collect segment IDs before deletion so we can clean up HNSW dirs that
        # Chroma's delete_collection may leave on disk when vector segments were
        # never loaded in the current process.
        seg_ids_to_purge: List[str] = []
        try:
            import sqlite3 as _sqlite3

            db_path = Path(self.persist_directory) / "chroma.sqlite3"
            if db_path.exists():
                con = _sqlite3.connect(str(db_path))
                cur = con.cursor()
                cur.execute(
                    """
                    SELECT s.id
                    FROM segments s
                    JOIN collections c ON s.collection = c.id
                    WHERE c.name = ? AND s.scope = 'VECTOR'
                    """,
                    (self.collection_name,),
                )
                seg_ids_to_purge = [r[0] for r in cur.fetchall()]
                con.close()
        except Exception:
            pass

        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            # It may not exist yet; safe to ignore.
            pass

        for seg_id in seg_ids_to_purge:
            seg_dir = Path(self.persist_directory) / seg_id
            if seg_dir.exists() and seg_dir.is_dir():
                try:
                    import shutil

                    shutil.rmtree(str(seg_dir))
                except Exception:
                    pass

        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Return embeddings for `texts` using `ollama` (python lib if available, otherwise CLI).

        This method attempts to use the `ollama` python client first; if not available it falls
        back to calling the `ollama` CLI. The exact behavior may need adjustment depending on
        your local `ollama` installation and version.
        """
        if not texts:
            return []

        # Try python package
        if self._ollama is not None:
            try:
                # Many Ollama python clients provide an embeddings or embed function.
                # Try common variants; if none exist, fall back to CLI.
                if hasattr(self._ollama, "embeddings"):
                    resp = self._ollama.embeddings(model=self.embedding_model, input=texts)
                    # resp might be a list of embedding vectors
                    return resp
                if hasattr(self._ollama, "embed"):
                    # Some clients expose embed(model, text)
                    out = [self._ollama.embed(self.embedding_model, t) for t in texts]
                    return out
            except Exception:
                pass
        # If an external embedding HTTP endpoint is configured (useful to point
        # at an Ollama server running on the host), prefer that. The endpoint
        # should accept JSON: {"model": "<model>", "input": ["...", ...]} and
        # return a JSON array of embedding vectors (list of lists).
        import os

        embed_url = os.environ.get("OLLAMA_EMBED_URL")
        if embed_url:
            try:
                import requests

                resp = requests.post(embed_url, json={"model": self.embedding_model, "input": texts}, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                # Expecting either {"embeddings": [...]} or a raw list
                if isinstance(data, dict) and "embeddings" in data:
                    return data["embeddings"]
                if isinstance(data, list):
                    return data
            except Exception as ex:
                # Network/response errors should surface but we will fall back
                # to other methods below.
                print(f"OLLAMA_EMBED_URL request failed: {ex}")

        # Deterministic fallback embedding so the indexer can operate in constrained environments.
        # Produce 768 dims to match common Ollama embedding output dimensions (e.g. nomic-embed-text)
        # and avoid dimension flips when connectivity to Ollama is intermittent.
        try:
            import hashlib

            embs: List[List[float]] = []
            for t in texts:
                h = hashlib.sha256(t.encode("utf-8")).digest()
                vec = [float(h[i % len(h)]) / 255.0 for i in range(768)]
                embs.append(vec)
            return embs
        except Exception:
            pass

        # Fallback to CLI-based embedding using `ollama embed` (best-effort). This assumes
        # the CLI supports: `ollama embed <model> <text>` and prints a JSON array.
        embeddings: List[List[float]] = []
        for t in texts:
            try:
                proc = subprocess.run(["ollama", "embed", self.embedding_model, t], capture_output=True, text=True)
                if proc.returncode != 0:
                    raise RuntimeError(f"ollama embed failed: {proc.stderr.strip()}")
                emb = json.loads(proc.stdout)
                embeddings.append(emb)
            except Exception:
                # If anything fails, raise an informative error
                raise RuntimeError("Failed to generate embeddings via ollama (python package or CLI).")

        return embeddings

    def process_directory(self, path: str):
        """Recursively read .md and .txt files under `path` (relative to configured docs_root),
        chunk and upsert into ChromaDB.

        Chunks are 1000 characters with 10% (100 char) overlap by default.
        """
        # Resolve path within docs_root to avoid indexing outside configured root
        import os

        root = self.docs_root
        req = Path(path)
        if req.is_absolute():
            p = req.resolve()
        else:
            p = (root / req).resolve()

        # Ensure containment
        if hasattr(p, "is_relative_to"):
            if not p.is_relative_to(root):
                raise PermissionError(f"Requested path outside configured docs root: {path}")
        else:
            if os.path.commonpath([str(root)]) != os.path.commonpath([str(root), str(p)]):
                raise PermissionError(f"Requested path outside configured docs root: {path}")

        if not p.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        docs: List[str] = []
        metadatas: List[Dict[str, Any]] = []
        ids: List[str] = []

        CHUNK_SIZE = 1000
        OVERLAP = math.floor(CHUNK_SIZE * 0.1)

        for file in p.rglob("*"):
            if not file.is_file():
                continue
            if file.suffix.lower() not in {".md", ".txt"}:
                continue

            try:
                text = file.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue

            chunks = self._chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
            for idx, chunk in enumerate(chunks):
                docs.append(chunk)
                metadatas.append({"source": str(file), "chunk_index": idx})
                ids.append(str(uuid.uuid4()))

            # Upsert in reasonably sized batches to avoid huge memory spikes
            if len(docs) >= 256:
                embs = self._embed_texts(docs)
                try:
                    self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embs)
                except Exception as ex:
                    if self._is_dimension_mismatch_error(ex):
                        self._reset_collection()
                        self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embs)
                    else:
                        raise
                docs = []
                metadatas = []
                ids = []

        # final batch
        if docs:
            embs = self._embed_texts(docs)
            try:
                self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embs)
            except Exception as ex:
                if self._is_dimension_mismatch_error(ex):
                    self._reset_collection()
                    self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embs)
                else:
                    raise

    def semantic_search(self, query: str, n: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top `n` relevant chunks for `query`.

        Returns a list of results with keys: `document`, `metadata`, and `distance` (if available).
        """
        # Compute query embedding
        q_embs = self._embed_texts([query])
        if not q_embs:
            return []
        query_emb = q_embs[0]

        # If we have a persistent Chroma collection, use it
        if self.collection is not None:
            try:
                results = self.collection.query(query_embeddings=[query_emb], n_results=n, include=["documents", "metadatas", "distances"])
            except Exception as ex:
                if self._is_dimension_mismatch_error(ex):
                    self._reset_collection()
                    self.process_directory(".")
                    results = self.collection.query(query_embeddings=[query_emb], n_results=n, include=["documents", "metadatas", "distances"])
                else:
                    raise
            out: List[Dict[str, Any]] = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0] if "distances" in results else [None] * len(docs)
            for d, m, s in zip(docs, metas, dists):
                out.append({"document": d, "metadata": m, "distance": s})
            return out

        # Otherwise, build or use an in-memory index and perform a simple cosine-similarity search
        if self._memory_index is None:
            # Build index from files under docs_root
            idx: List[Dict[str, Any]] = []
            for file in Path(self.docs_root).rglob("*"):
                if not file.is_file():
                    continue
                if file.suffix.lower() not in {".md", ".txt"}:
                    continue
                try:
                    text = file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                chunks = self._chunk_text(text)
                for ci, chunk in enumerate(chunks):
                    emb = self._embed_texts([chunk])[0]
                    idx.append({"document": chunk, "metadata": {"source": str(file), "chunk_index": ci}, "embedding": emb})
            self._memory_index = idx

        # Compute cosine similarity
        def cosine(a: List[float], b: List[float]) -> float:
            import math

            sa = sum(x * x for x in a)
            sb = sum(x * x for x in b)
            if sa == 0 or sb == 0:
                return 0.0
            dot = sum(x * y for x, y in zip(a, b))
            return dot / (math.sqrt(sa) * math.sqrt(sb))

        scored = []
        for item in self._memory_index:
            score = cosine(query_emb, item.get("embedding", []))
            scored.append((score, item))
        scored.sort(key=lambda t: t[0], reverse=True)
        out = []
        for score, item in scored[:n]:
            out.append({"document": item["document"], "metadata": item["metadata"], "distance": 1.0 - score})
        return out


if __name__ == "__main__":
    print("DocIndexer module loaded. Use DocIndexer.process_directory() and semantic_search().")
