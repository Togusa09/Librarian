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
    from chromadb.config import Settings
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

        settings = Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.persist_directory)
        self.client = chromadb.Client(settings)
        # Create or get collection. We'll pass embeddings explicitly on upsert.
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

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
                self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embs)
                docs = []
                metadatas = []
                ids = []

        # final batch
        if docs:
            embs = self._embed_texts(docs)
            self.collection.upsert(ids=ids, documents=docs, metadatas=metadatas, embeddings=embs)

    def semantic_search(self, query: str, n: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top `n` relevant chunks for `query`.

        Returns a list of results with keys: `document`, `metadata`, and `distance` (if available).
        """
        q_embs = self._embed_texts([query])
        if not q_embs:
            return []
        query_emb = q_embs[0]

        results = self.collection.query(query_embeddings=[query_emb], n_results=n, include=["documents", "metadatas", "distances"])

        out: List[Dict[str, Any]] = []
        # Chromadb returns results keyed by query index; we asked one query so index 0
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0] if "distances" in results else [None] * len(docs)

        for d, m, s in zip(docs, metas, dists):
            out.append({"document": d, "metadata": m, "distance": s})

        return out


if __name__ == "__main__":
    print("DocIndexer module loaded. Use DocIndexer.process_directory() and semantic_search().")

