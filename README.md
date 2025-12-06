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

   > Tip: set `OPENAI_API_KEY` before running any ingestion/RAG scripts.

3. (Optional) Install pre-commit hooks so formatting/linting runs automatically:

   ```bash
   uv pip install pre-commit
   pre-commit install
   ```

You can now develop inside `.venv`; every change under `src/galactic_rag` is immediately available because the project is installed in editable mode.
