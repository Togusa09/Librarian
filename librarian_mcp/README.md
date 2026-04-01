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

HTTP JSON API (LLM-friendly)

This server exposes a small HTTP JSON API intended for programmatic LLM tool invocations. Endpoints:

- `GET /.well-known/mcp.json` — discovery; includes `base_url` and tool list.
- `GET /healthz` — liveness probe returning `{"status":"ok"}`.
- `GET /readyz` — readiness; currently reports whether indexing is initialized.
- `GET /tools` — returns available tools and their simple input schemas.
- `POST /tools/<tool_name>` — invoke a tool with a JSON body. Use `Content-Type: application/json`.

Example curl commands:

```bash
# discovery + health
curl -sS http://localhost:8001/.well-known/mcp.json
curl -sS http://localhost:8001/healthz

# tool POST examples (adjust port to the server's API port)
curl -sS -X POST -H "Content-Type: application/json" -d '{"path":"/"}' http://localhost:8001/tools/list_files
curl -sS -X POST -H "Content-Type: application/json" -d '{"path":"example.md"}' http://localhost:8001/tools/read_document
curl -sS -X POST -H "Content-Type: application/json" -d '{"path":"/images/logo.png","prefer":"redirect"}' http://localhost:8001/tools/read_binary
```

Tiny OpenAPI-like fragment for tools

```yaml
openapi: 3.0.0
info:
	title: Librarian MCP Tools (minimal)
	version: '0.1'
paths:
	/.well-known/mcp.json:
		get: {}
	/healthz:
		get: {}
	/tools:
		get:
			responses:
				'200':
					description: tool schemas
	/tools/list_files:
		post:
			requestBody:
				content:
					application/json:
						schema:
							type: object
							properties:
								path:
									type: string
			responses:
				'200':
					description: file list
```


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