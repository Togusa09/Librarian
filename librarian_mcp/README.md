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

Docker

Build the image (from repository root):

```bash
docker build -t librarian_mcp:latest .
```

Run with local directories mounted for persistence:

```bash
docker run --rm -it \
	-e DOCUMENT_ARCHIVE_PATH=/docs \
	-e DATA_PATH=/data \
	-e BIND_HOST=0.0.0.0 \
	-e PORT=8000 \
	-v $(pwd)/docs:/docs \
	-v $(pwd)/data:/data \
	-p 8000:8000 \
	librarian_mcp:latest
```

Or use the included `docker-compose.yml` example from the repository root:

```bash
docker-compose up --build
```

Notes
- The container exposes `DOCUMENT_ARCHIVE_PATH` and `DATA_PATH` for mapping your document archive and persistent DB/cache files.
- The server will create the directories at container startup if they do not already exist.

**GitHub Copilot MCP Config**

You can provide a small MCP configuration file to help GitHub Copilot (or other MCP-capable clients) discover the server and available tools. Save the example below as `copilot-mcp.json` in the repository root and adapt `host`/`port` as needed.

```json
{
	"name": "librarian_mcp",
	"host": "127.0.0.1",
	"port": 8000,
	"tls": false,
	"tools": [
		{ "name": "list_files", "description": "List files under docs root" },
		{ "name": "read_document", "description": "Return document text" },
		{ "name": "search_knowledge_base", "description": "Semantic search for docs" },
		{ "name": "read_binary", "description": "Return binary (inline base64 or HTTP fallback)" }
	]
}
```

How to use

- Place `copilot-mcp.json` in your project and follow GitHub Copilot/extension docs to point the extension at the MCP server or config file. The exact integration steps depend on the Copilot build and your editor — consult Copilot documentation for "MCP" or "Model Context Protocol" for details.


Update image locally

Rebuild the local image after code changes from the repository root:

```bash
docker build -t librarian_mcp:latest .
```

If you prefer to tag with a version:

```bash
docker build -t librarian_mcp:0.1.0 .
```

Create container based on image

Run a container mapping your document archive and a host data directory for persistent DB/cache files:

```bash
docker run -d --name librarian_mcp \
	-e DOCUMENT_ARCHIVE_PATH=/docs \
	-e DATA_PATH=/data \
	-e BIND_HOST=0.0.0.0 \
	-e PORT=8000 \
	-v $(pwd)/docs:/docs \
	-v $(pwd)/data:/data \
	-p 8000:8000 \
	librarian_mcp:latest
```

Stop and remove the container:

```bash
docker stop librarian_mcp
docker rm librarian_mcp
```

Use `docker logs librarian_mcp` to inspect startup output and verify the server created the mapped directories.