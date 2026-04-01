Local Documentation MCP Server (scaffold)

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

- [doc_mcp/server.py](doc_mcp/server.py) — MCP server skeleton.
- [doc_mcp/indexer.py](doc_mcp/indexer.py) — RAG/indexing placeholders.
- [doc_mcp/requirements.txt](doc_mcp/requirements.txt) — required packages.

Next steps

- Implement MCP handler registration in `server.py` using your chosen `mcp` API.
- Implement embedding, upsert, and retrieval in `indexer.py` using `chromadb` and `ollama`.

Configuration

- Set the documentation root directory with the `DOCS_ROOT` environment variable, or pass `docs_root` when creating `GameDocServer`.
- The server enforces containment: clients may only access files inside the configured docs root. Paths returned by `list_files()` are relative to the docs root.

Tests

- Run tests with `pytest` from the `doc_mcp` directory:

```powershell
python -m pip install -r requirements.txt
pytest -q
```
