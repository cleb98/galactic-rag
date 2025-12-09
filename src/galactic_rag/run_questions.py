"""CLI entrypoint to run the RAG pipeline on domande.csv."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import typer

from datapizza.vectorstores.qdrant import QdrantVectorstore

from galactic_rag import Settings, get_settings
from galactic_rag.ingestion.ingestion import _build_vectorstore
from galactic_rag.rag_pipeline import build_rag_pipeline

app = typer.Typer(help="Run the Galactic RAG pipeline over the provided questions CSV.")
logger = logging.getLogger(__name__)


def _ensure_vectorstore(settings: Settings) -> QdrantVectorstore:
    return _build_vectorstore(settings)


def _load_questions(path: Path, limit: int | None = None, skip: int = 0):
    with path.open("r", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for idx, row in enumerate(reader, start=1):
            if idx <= skip:
                continue
            yield idx, row
            if limit and idx >= skip + limit:
                break


@app.command()
def answer(
    questions_file: Path | None = typer.Option(
        None, help="Path to domande.csv (defaults to settings.domande_csv)."
    ),
    output_file: Path = typer.Option(Path("submission.csv"), help="CSV output path."),
    limit: int = typer.Option(0, help="Max number of questions to answer (0 = all)."),
    skip: int = typer.Option(0, help="Number of questions to skip from the top."),
    top_k: int = typer.Option(6, help="How many chunks to retrieve per query."),
    max_tokens: int = typer.Option(800, help="Max tokens for the answering model."),
):
    """Answer the questions with the configured RAG pipeline."""

    settings = get_settings()
    questions_path = questions_file or settings.domande_csv
    if not questions_path.exists():
        raise typer.BadParameter(f"Questions file not found: {questions_path}")

    vectorstore = _ensure_vectorstore(settings)
    pipeline = build_rag_pipeline(settings=settings, vectorstore=vectorstore)

    answered_rows: list[dict[str, Any]] = []
    total_limit = limit if limit > 0 else None
    typer.echo(f"Answering questions from {questions_path} → {output_file}")

    for row_id, row in _load_questions(questions_path, limit=total_limit, skip=skip):
        question = (row.get("domanda") or "").strip()
        difficulty = (row.get("difficoltà") or "").strip()
        if not question:
            logger.warning("Skipping empty question at row %s", row_id)
            continue

        pipeline_inputs = {
            "rewriter": {"user_prompt": question},
            "filter_builder": {"user_prompt": question},
            "query_embedder": {"fallback_prompt": question},
            "retriever": {
                "collection_name": settings.qdrant_collection,
                "k": top_k,
                "vector_name": settings.embedding_model,
            },
            "prompt_builder": {"user_prompt": question},
            "generator": {
                "input": "",
            },
            "postprocess": {
                "question": question,
                "difficulty": difficulty,
                "row_id": row_id,
            },
        }

        try:
            outputs = pipeline.run(pipeline_inputs)
            post = outputs.get("postprocess", {})
            result_value = post.get("result", "")
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to answer question %s: %s", row_id, exc)
            result_value = ""

        answered_rows.append({"row_id": row_id, "result": result_value})

    parent_dir = output_file.parent
    if parent_dir and not parent_dir.exists():
        parent_dir.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["row_id", "result"])
        writer.writeheader()
        writer.writerows(answered_rows)

    typer.echo(f"Wrote {len(answered_rows)} answers to {output_file}")


if __name__ == "__main__":  # pragma: no cover
    app()
