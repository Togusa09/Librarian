Librarian MCP Server (scaffold)

Quick start

- Create and activate a Python environment.
- Install dependencies:

```
python -m pip install -r requirements.txt
```

- Run the placeholder server:

```
python server.py
```

Files

- [librarian_mcp/server.py](librarian_mcp/server.py) — MCP server skeleton.
- [librarian_mcp/indexer.py](librarian_mcp/indexer.py) — RAG/indexing placeholders.
- [librarian_mcp/requirements.txt](librarian_mcp/requirements.txt) — required packages.

Next steps

- Implement MCP handler registration in `server.py` using your chosen `mcp` API.
- Implement embedding, upsert, and retrieval in `indexer.py` using `chromadb` and `ollama`.

Configuration

- Set the documentation root directory with the `DOCS_ROOT` environment variable, or pass `docs_root` when creating `GameDocServer`.
- The server enforces containment: clients may only access files inside the configured docs root. Paths returned by `list_files()` are relative to the docs root.

Tests

- Run tests with `pytest` from the `librarian_mcp` directory:

```powershell
python -m pip install -r requirements.txt
pytest -q
```
