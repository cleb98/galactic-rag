# Galactic RAG

Pragmatic tooling that ingests the Hackapizza knowledge base with Datapizza AI and runs a Retrieval-Augmented Generation pipeline to answer every question.

## Quick start

1. Create and activate the virtual environment with uv (Python 3.12+ recommended):

   ```bash
   cd solution
   uv venv .venv
   source .venv/bin/activate
   ```

2. Install project dependencies from `pyproject.toml`:

   ```bash
   uv sync
   ```

   > Tip: set `OPENAI_API_KEY`/`GEMINI_API_KEY` before running any ingestion/RAG scripts.

3. (Optional) Install pre-commit hooks so formatting/linting runs automatically:

   ```bash
   uv pip install pre-commit
   pre-commit install
   ```

You can now develop inside `.venv`; every change under `src/galactic_rag` is immediately available because the project is installed in editable mode.

### Choose the provider once for ingestion + RAG

Set `PROVIDER` in `.env` (`openai` or `gemini`) before running any ingestion or answering command. The same setting drives both the embedder and the LLM client, so switching providers requires re-ingesting the menus with the new configuration. Each chunk written to Qdrant now carries `provider` in its metadata, which lets future RAG runs filter out embeddings produced by a different provider/encoder and prevents mixing Gemini vectors with OpenAI LLMs (and vice versa).

### Run Qdrant locally (optional)

Use the provided Docker Compose setup if you want a persistent Qdrant instance on `localhost:6333` (useful when experimenting from notebooks or external tools):

1. `cd src/galactic_rag/ingestion`
2. `cp .env.example .env` and set `QDRANT_API_KEY` to any secret token you prefer.
3. `docker compose up -d qdrant`

The service now listens on `http://localhost:6333` (REST) and `6334` (gRPC) and requires the API key you set. Stop it with `docker compose down`. When connecting from notebooks/scripts instantiate `QdrantVectorstore(host="localhost", port=6333, api_key=os.environ["QDRANT_API_KEY"])`.

## Menu ingestion

Run the Datapizza-style ingestion pipeline to parse every menu PDF and populate the local Qdrant collection:

```bash
source .venv/bin/activate
python -m galactic_rag.ingestion.ingestion --recreate
```

This command:

- parses `Dataset/knowledge_base/menu/**/*.pdf` with Docling,
- extracts logical sections (restaurant, chef, dishes) so every chunk maps to a single dish,
- attaches metadata such as chef, restaurant, page numbers and provider to each chunk,
- embeds both the raw dish text and a structured summary (two named vectors) with the configured provider, and
- upserts the chunks into `data/vectorstore` under the collection name from `config.Settings` (defaults to `galactic_menu`).

Use `--limit N` to ingest only the first `N` PDFs while debugging and `--batch-size` to tune the embedding throughput. Skip `--recreate` to append to the existing collection instead of dropping it.
Need to inspect the raw Docling output? Pass `--docling-json-dir data/docling_json` (or any path) and the parser will persist the intermediate JSON files so you can inspect what Docling extracted.
