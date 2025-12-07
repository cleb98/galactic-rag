"""Scripts to ingest Galactic menu PDFs into a Qdrant vector store."""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Sequence
from pathlib import Path

import typer
from datapizza.core.embedder import BaseEmbedder
from datapizza.core.models import PipelineComponent
from datapizza.core.vectorstore import Distance, VectorConfig
from datapizza.embedders import ChunkEmbedder
from datapizza.embedders.google import GoogleEmbedder
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.clients.openai_like import OpenAILikeClient
from datapizza.modules.parsers.docling import DoclingParser
from datapizza.pipeline import IngestionPipeline
from datapizza.type import (Chunk, DenseEmbedding, EmbeddingFormat, Node,
                            NodeType)
from datapizza.vectorstores.qdrant import QdrantVectorstore

from galactic_rag.config import Settings, get_settings
from galactic_rag.ingestion.utils.constant import INFO_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

app = typer.Typer(help="Ingest the knowledge_base/menu PDFs into Qdrant")


def _embedding_names(settings: Settings) -> tuple[str, str]:
    base = settings.embedding_model
    return base, f"{base}-context"


def _clean_heading_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"^#+\s*", "", text).strip()


def _extract_heading_text(section: Node) -> str:
    raw = section.metadata.get("docling_raw") or {}
    heading = raw.get("text")
    if heading:
        return heading.strip()
    for child in section.children:
        if child.metadata.get("markdown_rendering") == "heading":
            return _clean_heading_text(child.content)
    return ""


def _section_plain_text(node: Node) -> str:
    parts: list[str] = []
    for child in node.children:
        if child.node_type == NodeType.SECTION:
            heading = _clean_heading_text(_extract_heading_text(child))
            if heading:
                parts.append(heading)
            text = _section_plain_text(child)
            if text:
                parts.append(text)
            continue
        if child.metadata.get("markdown_rendering") == "heading":
            continue
        content = (child.content or "").strip()
        if content:
            parts.append(content)
    return "\n\n".join(parts).strip()


def _section_body_text(node: Node) -> str:
    parts: list[str] = []
    for child in node.children:
        if child.node_type == NodeType.SECTION:
            parts.append(_section_plain_text(child))
        else:
            if child.metadata.get("markdown_rendering") == "heading":
                continue
            text = (child.content or "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(p for p in parts if p).strip()


def _extract_page_numbers(metadata: dict) -> list[int]:
    if "page_nos" in metadata and isinstance(metadata["page_nos"], list):
        return [int(p) for p in metadata["page_nos"] if isinstance(p, int)]
    page_no = metadata.get("page_no")
    if isinstance(page_no, int):
        return [page_no]
    return []


def _looks_like_metadata_section(heading: str) -> bool:
    lowered = heading.lower()
    non_dish = ("chef", "ristorante", "menu", "introduzione")
    return _classify_metadata_heading(lowered) is not None or any(
        key in lowered for key in non_dish
    )


def _classify_metadata_heading(heading: str) -> str | None:
    lowered = heading.lower()
    if "ingredient" in lowered:
        return "ingredients"
    if any(key in lowered for key in ("tecnica", "prepar", "proced", "cottura")):
        return "techniques"
    if "note" in lowered:
        return "notes"
    return None


# non esistono queste subsections, vanno trovate tra i fratelli successivi al massimo
def _collect_following_metadata(
    sections: Sequence[Node], start_idx: int, max_metadata_sections: int = 2
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {
        "ingredients": [],
        "techniques": [],
        "notes": [],
    }
    end_idx = min(len(sections), start_idx + 1 + max_metadata_sections)
    for idx in range(start_idx + 1, end_idx):
        section = sections[idx]
        heading = _clean_heading_text(_extract_heading_text(section))
        classification = _classify_metadata_heading(heading)
        text = _section_body_text(section)
        if classification:
            if text:
                result[classification].append(text)
            continue
        if heading:
            break
        if text:
            result["notes"].append(text)
    return result

# non esistono queste subsections, vanno trovate tra i fratelli successivi al massimo
def _build_structured_text(
    *,
    restaurant: str,
    chef: str,
    dish: str,
    body: str,
    pages: Sequence[int],
    subsections: dict[str, list[str]],
) -> str:
    parts = [
        f"Ristorante: {restaurant or 'n/d'}",
        f"Chef: {chef or 'n/d'}",
        f"Piatto: {dish}",
    ]
    if pages:
        parts.append("Pagine: " + ", ".join(str(p) for p in pages))
    if subsections["ingredients"]:
        parts.append("Ingredienti: " + " ".join(subsections["ingredients"]))
    if subsections["techniques"]:
        parts.append("Tecniche: " + " ".join(subsections["techniques"]))
    if subsections["notes"]:
        parts.append("Note: " + " ".join(subsections["notes"]))
    if body:
        parts.append("Dettagli: " + body)
    return "\n".join(parts)


def _origin_stem(document: Node) -> str:
    origin = document.metadata.get("origin") or {}
    file_path = origin.get("file_path") or origin.get("source") or ""
    if file_path:
        return Path(file_path).stem.replace("_", " ")
    return ""



class MenuSectionSplitter(PipelineComponent):
    """Custom splitter that extracts restaurant, chef and dishes as chunks."""
    def __init__(self, llm: OpenAILikeClient | None = None):
        self.llm = llm 

    def _extract_with_llm(
        self, *, dish_name: str, body: str
    ) -> dict[str, list[str] | str] | None:
        if not self.llm or not body:
            return None
        prompt = (
            "Estrai nome piatto, ingredienti e tecniche dal seguente testo.\n"
            "Rispondi solo con JSON: "
            '{"dish_name": "...", "ingredients": ["..."], "techniques": ["..."], "notes": ["..."]}\n'
            f"Titolo: {dish_name or 'n/d'}\n"
            f"Testo: {body}"
        )
        try:
            response = self.llm.invoke(input=prompt)
            text = response.first_text or response.text
            data = json.loads(text)
            return {
                "dish_name": str(data.get("dish_name", "")).strip(),
                "ingredients": [i.strip() for i in data.get("ingredients", []) if i],
                "techniques": [t.strip() for t in data.get("techniques", []) if t],
                "notes": [n.strip() for n in data.get("notes", []) if n],
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM extraction failed: %s", exc)
            return None

    def _run(self, document: Node) -> list[Chunk]:
        sections = [
            child for child in document.children if child.node_type == NodeType.SECTION
        ]
        if not sections:
            raise ValueError("Parsed document does not contain any sections")

        restaurant_section = sections[0]
        restaurant_name = (
            _clean_heading_text(_extract_heading_text(restaurant_section))
            or document.metadata.get("name")
            or _origin_stem(document)
        )

        chef_name = ""
        for section in sections:
            heading = _clean_heading_text(_extract_heading_text(section)).lower()
            if "chef" in heading:
                chef_name = re.sub(r"(?i)chef[:,\\-\\s]*", "", heading, count=1).strip()
                if not chef_name:
                    chef_name = (
                        _section_body_text(section).split("\n")[0].strip()
                        if _section_body_text(section)
                        else ""
                    )
                break
        if not chef_name and len(sections) > 1:
            candidate_text = _section_body_text(sections[1])
            match = re.search(r"(?i)chef[:\\-\\s]+([A-Za-z\\s']+)", candidate_text)
            if match:
                chef_name = match.group(1).strip()

        # Build dishes from sequential sections (metadata lives in following siblings)
        dish_sections: list[tuple[int, Node]] = []
        for idx, section in enumerate(sections):
            if idx == 0:
                continue
            heading = _clean_heading_text(_extract_heading_text(section))
            if not heading:
                continue
            if _looks_like_metadata_section(heading):
                continue
            dish_sections.append((idx, section))

        chunks: list[Chunk] = []
        # vedo sezione per sxeziojne se ha ingredienti e tecniche nel nome, in caso il contenuto lo metto nell'oggetto pydantic
        # due oggetti pydanticun per documento e uno per piatto (un piatto e' un chunk) (prima del piatto c'e' il primo section che da info sul cuoco e la cucina)
        # oggetto pydantic con dish name, description, ingredients, techniques
        for idx, dish_section in dish_sections:
            dish_name = _clean_heading_text(_extract_heading_text(dish_section))
            if not dish_name:
                continue
            pages = _extract_page_numbers(dish_section.metadata)
            raw_text = _section_body_text(dish_section)
            subsections = _collect_following_metadata(sections, idx)
            if not subsections["ingredients"] and not subsections["techniques"]:
                llm_data = self._extract_with_llm(dish_name=dish_name, body=raw_text)
                if llm_data:
                    if llm_data.get("dish_name"):
                        dish_name = llm_data["dish_name"]
                    subsections["ingredients"] = llm_data.get("ingredients", [])
                    subsections["techniques"] = llm_data.get("techniques", [])
                    subsections["notes"] = llm_data.get("notes", [])
            text_parts: list[str] = []
            if raw_text:
                text_parts.append(raw_text)
            for bucket in ("ingredients", "techniques", "notes"):
                text_parts.extend(subsections[bucket])
            if not any(text_parts):
                continue
            structured_text = _build_structured_text(
                restaurant=restaurant_name,
                chef=chef_name,
                dish=dish_name,
                body=raw_text or " ".join(text_parts),
                pages=pages,
                subsections=subsections,
            )
            
            chunk_metadata = {
                "restaurant": restaurant_name,
                "chef": chef_name,
                "dish_name": dish_name,
                "section_heading": dish_name,
                "pag_content": (
                    dish_section.content.strip() if dish_section.content else raw_text
                ),
                "num_pag": pages,
                "page_numbers": pages,
                "structured_text": structured_text,
            }
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=raw_text or " ".join(text_parts),
                    metadata=chunk_metadata,
                )
            )

        if not chunks:
            fallback = Chunk(
                id=str(uuid.uuid4()),
                text=document.content,
                metadata={
                    "restaurant": restaurant_name,
                    "chef": chef_name,
                    "structured_text": _build_structured_text(
                        restaurant=restaurant_name,
                        chef=chef_name,
                        dish=restaurant_name or "menu",
                        body=document.content,
                        pages=_extract_page_numbers(document.metadata),
                        subsections={"ingredients": [], "techniques": [], "notes": []},
                    ),
                },
            )
            return [fallback]

        return chunks


class StructuredEmbeddingAugmenter(PipelineComponent):
    def __init__(
        self,
        client: BaseEmbedder,
        model_name: str,
        embedding_name: str,
        batch_size: int = 64,
    ):
        self.client = client
        self.model_name = model_name
        self.embedding_name = embedding_name
        self.batch_size = batch_size

    def _run(self, chunks: list[Chunk]) -> list[Chunk]:
        payloads: list[tuple[Chunk, str]] = []
        for chunk in chunks:
            structured = chunk.metadata.get("structured_text")
            if structured:
                payloads.append((chunk, structured))
        if not payloads:
            return chunks

        for start in range(0, len(payloads), self.batch_size):
            batch = payloads[start : start + self.batch_size]
            texts = [text for _, text in batch]
            embeddings = self.client.embed(texts, self.model_name)
            for (chunk, _), vector in zip(batch, embeddings, strict=False):
                chunk.embeddings.append(
                    DenseEmbedding(name=self.embedding_name, vector=vector)
                )
        return chunks


def _build_embedder(settings: Settings) -> BaseEmbedder:
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
    raise ValueError(f"Unsupported provider {settings.provider}")


def _ensure_collection(
    *,
    settings: Settings,
    vectorstore: QdrantVectorstore,
    collection_name: str,
    embedding_names: Sequence[str],
    recreate: bool,
):
    """Create the Qdrant collection and optionally recreate it."""

    dims = settings.embedding_dimensions()
    vector_config = [
        VectorConfig(
            name=name,
            format=EmbeddingFormat.DENSE,
            dimensions=dims,
            distance=Distance.COSINE,
        )
        for name in embedding_names
    ]

    if recreate:
        try:
            vectorstore.delete_collection(collection_name)
            logger.info("Deleted existing collection %s", collection_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not delete collection %s: %s", collection_name, exc)

    vectorstore.create_collection(collection_name, vector_config)


def _build_pipeline(
    *,
    settings: Settings,
    vectorstore: QdrantVectorstore,
    batch_size: int,
    json_output_dir: Path | None = None,
) -> IngestionPipeline:
    """Assemble the ingestion pipeline with parser, section splitter, and dual embeddings."""

    parser = DoclingParser(
        json_output_dir=str(json_output_dir) if json_output_dir else None
    )
    llm = OpenAILikeClient(
        api_key=settings.api_key,
        model=settings.llm_model,
        system_prompt=INFO_EXTRACTION_PROMPT,
    )
    splitter = MenuSectionSplitter(llm=llm)
    embedder_client = _build_embedder(settings)
    raw_name, context_name = _embedding_names(settings)
    chunk_embedder = ChunkEmbedder(
        client=embedder_client,
        model_name=settings.embedding_model,
        embedding_name=raw_name,
        batch_size=batch_size,
    )
    contextual_embedder = StructuredEmbeddingAugmenter(
        client=embedder_client,
        model_name=settings.embedding_model,
        embedding_name=context_name,
        batch_size=batch_size,
    )

    return IngestionPipeline(
        modules=[parser, splitter, chunk_embedder, contextual_embedder],
        vector_store=vectorstore,
        collection_name=settings.qdrant_collection,
    )


def _build_vectorstore(settings: Settings) -> QdrantVectorstore:
    """Use remote Qdrant if host is set, otherwise fall back to local storage."""

    if settings.qdrant_host:
        return QdrantVectorstore(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            api_key=settings.qdrant_api_key,
            https=settings.qdrant_https,
        )
    return QdrantVectorstore(location=str(settings.vectorstore_path))


def _iter_pdfs(menu_dir: Path) -> Iterable[Path]:
    """Yield PDFs from the menu directory and nested subdirectories."""

    yield from sorted(menu_dir.glob("*.pdf"))
    for subdir in sorted(p for p in menu_dir.iterdir() if p.is_dir()):
        yield from sorted(subdir.rglob("*.pdf"))


@app.command()
def ingest_data(
    menu_dir: Path | None = typer.Option(
        None,
        help="Directory containing the menu PDFs (defaults to settings.knowledge_base/menu)",
    ),
    limit: int = typer.Option(0, help="Ingest only the first N PDFs (0 = all)"),
    recreate: bool = typer.Option(
        False,
        help="Drop and recreate the Qdrant collection before ingesting",
    ),
    batch_size: int = typer.Option(64, help="Embedding batch size"),
    docling_json_dir: Path | None = typer.Option(
        None,
        help="Optional directory where raw Docling JSON dumps will be stored for debugging",
    ),
):
    """Ingest menu PDFs into the configured Qdrant vector store."""

    settings = get_settings()
    target_dir = menu_dir or (settings.knowledge_base_dir / "menu")
    if not target_dir.exists():
        raise typer.BadParameter(f"Menu directory {target_dir} does not exist")

    pdfs = list(_iter_pdfs(target_dir))
    if not pdfs:
        raise typer.Exit("No PDF files found in menu directory")
    if limit > 0:
        pdfs = pdfs[:limit]

    vectorstore = _build_vectorstore(settings)
    embedding_names = _embedding_names(settings)
    _ensure_collection(
        settings=settings,
        vectorstore=vectorstore,
        collection_name=settings.qdrant_collection,
        embedding_names=embedding_names,
        recreate=recreate,
    )

    pipeline = _build_pipeline(
        settings=settings,
        vectorstore=vectorstore,
        batch_size=batch_size,
        json_output_dir=docling_json_dir,
    )

    typer.echo(
        f"Ingesting {len(pdfs)} PDF(s) into collection {settings.qdrant_collection}"
    )

    for path in pdfs:
        relative = path.relative_to(settings.dataset_root)
        metadata = {
            "source_path": str(relative),
            "category": "menu",
            "restaurant": path.parent.name,
            "provider": settings.provider,
        }
        typer.echo(f"→ {relative}")
        pipeline.run(str(path), metadata=metadata)

    typer.echo("Ingestion completed.")


if __name__ == "__main__":
    app()
