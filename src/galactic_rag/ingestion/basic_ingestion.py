from __future__ import annotations

import argparse
import logging
from pathlib import Path

from datapizza.core.vectorstore import Distance, VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.google import GoogleEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.modules.parsers.docling import (DoclingParser, OCREngine,
                                               OCROptions)
from datapizza.modules.splitters import NodeSplitter
from datapizza.pipeline import IngestionPipeline
from datapizza.type import EmbeddingFormat
from datapizza.vectorstores.qdrant import QdrantVectorstore

from galactic_rag.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _build_vectorstore(settings: Settings) -> QdrantVectorstore:
    """Return a remote Qdrant client when host is set, otherwise local storage."""
    if settings.qdrant_host:
        return QdrantVectorstore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=settings.qdrant_https,
        )
    return QdrantVectorstore(location=str(settings.vectorstore_path))


def _build_embedder(settings: Settings):
    if settings.provider == "openai":
        return OpenAIEmbedder(
            api_key=settings.api_key,
            model_name=settings.embedding_model,
            base_url=settings.base_url,
        )
    if settings.provider == "gemini":
        return GoogleEmbedder(
            api_key=settings.api_key,
            model_name=settings.embedding_model,
        )
    raise ValueError(f"Unsupported provider '{settings.provider}'")


def run_basic_ingestion(pdf_path: Path, ocr_engine: OCREngine) -> None:
    """Ingest a single PDF into Qdrant using the configured settings."""
    settings = get_settings()
    vectorstore = _build_vectorstore(settings)
    embedder_client = _build_embedder(settings)

    embedding_name = settings.embedding_model
    dims = settings.embedding_dimensions()

    vector_config = [
        VectorConfig(
            name=embedding_name,
            format=EmbeddingFormat.DENSE,
            dimensions=dims,
            distance=Distance.COSINE,
        )
    ]

    try:
        vectorstore.create_collection(
            settings.qdrant_collection,
            vector_config=vector_config,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info(
            "Could not create collection %s (it may already exist): %s",
            settings.qdrant_collection,
            exc,
        )

    ingestion_pipeline = IngestionPipeline(
        modules=[
            DoclingParser(
                json_output_dir=settings.docling_json_output_dir,
                ocr_options=OCROptions(engine=ocr_engine),
            ),
            NodeSplitter(max_char=settings.chunk_max_chars),
            ChunkEmbedder(
                client=embedder_client,
                model_name=settings.embedding_model,
                embedding_name=embedding_name,
            ),
        ],
        vector_store=vectorstore,
        collection_name=settings.qdrant_collection,
    )

    ingestion_pipeline.run(
        str(pdf_path),
        metadata={"source": str(pdf_path)},
    )

    res = vectorstore.search(
        collection_name=settings.qdrant_collection,
        query_vector=[0.0] * dims,
        vector_name=embedding_name,
        k=2,
    )
    print(res)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a minimal ingestion pipeline using project settings.",
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        default="sample.pdf",
        help="Path to the PDF to ingest (defaults to sample.pdf)",
    )
    parser.add_argument(
        "--ocr-engine",
        choices=[engine.value for engine in OCREngine],
        default=OCREngine.NONE.value,
        help="OCR engine used by Docling (default: none, fastest option)",
    )
    args = parser.parse_args()
    run_basic_ingestion(Path(args.pdf), OCREngine(args.ocr_engine))


if __name__ == "__main__":
    main()
