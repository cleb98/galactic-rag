"""Scripts to ingest Galactic menu PDFs into a Qdrant vector store."""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Iterable, List

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
from datapizza.type import Chunk, DenseEmbedding, EmbeddingFormat, Node, NodeType
from datapizza.vectorstores.qdrant import QdrantVectorstore
from pydantic import BaseModel, Field, field_validator

from galactic_rag.config import Settings, get_settings
from galactic_rag.ingestion.utils.constant import INFO_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)

app = typer.Typer(help="Ingest the knowledge_base/menu PDFs into Qdrant")


def _embedding_names(settings: Settings) -> tuple[str, str]:
    base = settings.embedding_model
    return base, f"{base}-context"


def _clean_heading_text(text: str | None) -> str:
    """Remove all leading markdown (#) heading markers from text."""
    if not text:
        return ""
    return re.sub(r"^#+\s*", "", text).strip()


def _extract_heading_text(section: Node) -> str:
    """Extract the heading text from a section node.
    Looks for 'docling_raw' metadata first, then child nodes with
    'markdown_rendering' set to 'heading.
    """
    raw = section.metadata.get("docling_raw") or {}
    heading = raw.get("text")
    if heading:
        return heading.strip()
    for child in section.children:
        if child.metadata.get("markdown_rendering") == "heading":
            return child.content
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
    return 0

# def _build_structured_text(
#     *,
#     # restaurant: str,
#     # chef: str,
#     # dish: str,
#     # subsections: dict[str, list[str]],
# ) -> str:
#     parts = [
#         f"Ristorante: {restaurant or 'n/d'}",
#         f"Chef: {chef or 'n/d'}",
#         f"Piatto: {dish}",
#     ]
#     if subsections["ingredients"]:
#         parts.append("Ingredienti: " + " ".join(subsections["ingredients"]))
#     if subsections["techniques"]:
#         parts.append("Tecniche: " + " ".join(subsections["techniques"]))
#     if subsections["notes"]:
#         parts.append("Note: " + " ".join(subsections["notes"]))
#     return "\n".join(parts)


class RestaurantInfo(BaseModel):
    restaurant_name: str = Field(default="", description="The name of the restaurant")
    chef_name: str = Field(default="", description="The name of the chef")
    
    @field_validator("restaurant_name", "chef_name", mode="before")
    @classmethod
    def _ensure_text(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

class MenuInfo(BaseModel):
    dish_name: str = Field(default="", description="The name of the dish described")
    ingredients: list[str] = Field(default_factory=list, description="List of ingredients used in the dish")
    techniques: list[str] = Field(default_factory=list, description="List of techniques or preparation methods used in the dish")
    notes: list[str] = Field(default_factory=list, description="Additional notes about the dish")

    @field_validator("dish_name", mode="before")
    @classmethod
    def _ensure_text(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("ingredients", "techniques", "notes", mode="before")
    @classmethod
    def _normalize_collection(cls, value: Any) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if isinstance(value, list):
            iterable = value
        elif isinstance(value, (set, tuple)):
            iterable = list(value)
        else:
            iterable = [value]

        cleaned: list[str] = []
        for item in iterable:
            if not item:
                continue
            cleaned.append(str(item).strip())
        return cleaned

# class ChunkMetadata(BaseModel):
#     num_pag: list[int] = Field(default_factory=list)
#     restaurant_info: RestaurantInfo = Field(default_factory=RestaurantInfo)
#     dish_info: list[MenuInfo] = Field(
#         default_factory=list,
#         description="List of dishes described in the document",
#     )
#     document_summary: str = Field(
#         default="",
#         description="Short summary of the restaurant and its menu extracted from the document",
#     )


class DishInfo(BaseModel):
    """Structured information about a dish extracted from the document.
    """ 

    heading: str = Field(
        default="",
        description="Heading of the chunk",
    )

    dish_name: str = Field(
        default="",
        description="Name of the dish",
    )
    ingredients: list[str] = Field(
        default_factory=list,
        description="Ingredients used for the dish",
    )
    techniques: list[str] = Field(
        default_factory=list,
        description="Preparation or cooking techniques used for the dish",
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Additional notes or remarks about the dish",
    )


class DocInfo(BaseModel):
    document_summary: str = Field(
        default="",
        description="Overall summary of the document content",
    )
    chef_name: str = Field(
        default="",
        description="Name of the chef mentioned in the document",
    )
    restaurant_name: str = Field(
        default="",
        description="Name of the restaurant mentioned in the document",
    )
    dish_info: list[DishInfo] = Field(
        default_factory=list,
        description=(
            "List of dishes described in the document with their details"
        ),
    )

class ChunkInfo(BaseModel):
    heading: str
    body: str
    pag_number: List[int]
    metadata: dict = {}



class MenuSectionSplitter(PipelineComponent):
    """Custom splitter that extracts restaurant, chef and dishes as chunks."""

    def __init__(self, llm: OpenAILikeClient | None = None):
        self.llm = llm
        # self.metadata: ChunkMetadata = ChunkMetadata()


    def _extract_document_info(
        self, *, document: str
    ) -> DocInfo:
        
        """Extract structured document info using LLM."""
        assert self.llm is not None, "LLM client is required for document info extraction"

        prompt = (
            "Sei un assistente che aiuta a estrarre dati strutturati dal testo di un chunk.\n"
            "Ogni testo descrive un ristorante e il suo menu.\n"
            "I menu descrivono in linguaggio naturale il ristorante, riportando il nome dello Chef, il nome del ristorante, (laddove presente) il pianeta su cui c'è il ristorante e le licenze culinarie che ha lo chef\n"
            "Ogni menu contiene 10 piatti\n"
            "Ogni piatto contiene gli ingredienti usati e le tecniche di preparazione\n"
            "Alcuni menu possiedono anche una descrizione in linguaggio naturale della preparazione\n"
            "Laddove vi siano certi ordini professionali, i menu lo citano\n"
            f"Testo del chunk: {document}\n"

            "# COME ESTRARE INFORMAZIONI DA UN TESTO\n"
            "Di seguito ti spiego come estrarre le info di un ristorante da un testo."
            "chef_name: str estrarre il nome dello chef dal testo.\n"
            "restaurant_name: str estrarre il nome del ristorante dal testo.\n"
            "document_summary: str Genera un breve riassunto in italiano con massimo 5 frasi del ristorante descritto nel testo seguente.\n"
            "dish_info: list[DishInfo] Estrai le informazioni sui piatti presenti nel menu e la pagina in cui sono menzionati.\n"
            "Includi il nome del ristorante, dello chef e una descrizione generale.\n"
            

            
        )
        try:
            response = self.llm.structured_response(
                input=prompt,
                output_cls=DocInfo,
            )
            return response.structured_data[0] 
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM generation failed: %s", exc)
        return ""       

    def _run(self, document: Node) -> list[Chunk]:
        """Split the document into chunks based on menu sections.
        1. Identify restaurant and chef from headings or body.
        2. For each section, extract dish name, ingredients, techniques, and notes if possible.
        3. Generate a summary of the document using the extracted information and the whole document body.
        4. Create chunks for each dish section with structured metadata.
        """

        """comment to be removed later
        # for idx, (section, heading, body) in enumerate(section_data):

        #     if idx in skip_indices:
        #         continue

        #     if not body:
        #         continue

        #     if (
        #         re.search(
        #             r"(?i)chef[:\-\s]+([A-Za-zÀ-ÖØ-öø-ÿ\s\"'\\]+)", # es: chef: Mario Rossi
        #             heading,    
        #         )
        #         or re.search(
        #                 r"(?i)ristorante[:\-\s]+([A-Za-zÀ-ÖØ-öø-ÿ\s\"'\\]+)", # es: ristorante: Datapizza
        #                 heading,
        #             )
        #         or re.search(
        #             r"(?i)\bmenu\b", # es: "Menu del giorno" / "Menu"
        #             heading,
        #         )
        #     ):
        #         continue

        #     ingredients: list[str] = []
        #     techniques: list[str] = []
        #     notes: list[str] = []

        #     lookahead = idx + 1
        #     while lookahead < len(section_data):
        #         _, next_heading, next_body = section_data[lookahead]
        #         lowered_next = next_heading.lower()

        #         if "ingredient" in lowered_next:
        #             if next_body:
        #                 ingredients.extend(_split_list_lines(next_body))
        #                 combined = f"{next_heading}\n{next_body}".strip()
        #                 body = f"{body}\n\n{combined}" if body else combined
        #             skip_indices.add(lookahead)
        #             lookahead += 1
        #             continue

        #         if "tecnic" in lowered_next:
        #             if next_body:
        #                 techniques.extend(_split_list_lines(next_body))
        #                 combined = f"{next_heading}\n{next_body}".strip()
        #                 body = f"{body}\n\n{combined}" if body else combined
        #             skip_indices.add(lookahead)
        #             lookahead += 1
        #             continue

        #         if "note" in lowered_next:
        #             if next_body:
        #                 notes.extend(_split_list_lines(next_body))
        #                 combined = f"{next_heading}\n{next_body}".strip()
        #                 body = f"{body}\n\n{combined}" if body else combined
        #             skip_indices.add(lookahead)
        #             lookahead += 1
        #             continue

        #         break
        """
        sections = [
            child for child in document.children if child.node_type == NodeType.SECTION
        ]
        if not sections:
            raise ValueError("Parsed document does not contain any sections")

        section_data: list[tuple[Node, str, str]] = []
        chunks_data: list[ChunkInfo] = []
        document_txt = ""
        chunk_to_skip: set[int] = set()
        previous_page = -1
        for i, section in enumerate(sections):
            heading = _extract_heading_text(section)
            body = _section_body_text(section)
            pag = _extract_page_numbers(section.metadata)
            # if pag == 1 and previous_page == -1:
            #     document_txt += f"\n\n#pagina {pag}\n\n"
            # if pag != previous_page and previous_page != -1: 
            #     document_txt += f"\n\n#fine pagina {previous_page}\n\n"
            #     document_txt += f"\n\n#pagina {pag}\n\n"     

            
            document_txt += f"\n#HEADING:{heading}\nBODY:\n{body}\n".strip()
            previous_page = pag

            # per ogni sezione devo vedere se i chunk successivi sono ingredients, techniques, notes
            # in tal caso li aggiungo al body del chunk corrente e metterli in chunk_to_skip
            # inoltre se il body e' vuoto, non creo il chunk pero lo agiungo a document_txt

            if section.id in chunk_to_skip:
                continue    

            if not body:
                continue

            # aggiungo il testo di una sezione precedente al body corrente se e' ingredients, techniques, notes perche poi la skippo
            lookahead = i + 1
            while lookahead < len(sections):
                next_section = sections[lookahead]
                next_heading = _clean_heading_text(_extract_heading_text(next_section))
                next_body = _section_body_text(next_section)
                lowered_next = next_heading.lower()

                if "ingredient" in lowered_next:
                    if next_body:
                        combined = f"{next_heading}\n{next_body}".strip()
                        body = f"{body}\n\n{combined}" if body else combined
                    chunk_to_skip.add(next_section.id)
                    lookahead += 1
                    continue

                if "tecnic" in lowered_next:
                    if next_body:
                        combined = f"{next_heading}\n{next_body}".strip()
                        body = f"{body}\n\n{combined}" if body else combined
                    chunk_to_skip.add(next_section.id)
                    lookahead += 1
                    continue

                if "note" in lowered_next:
                    if next_body:
                        combined = f"{next_heading}\n{next_body}".strip()
                        body = f"{body}\n\n{combined}" if body else combined
                    chunk_to_skip.add(next_section.id)
                    lookahead += 1
                    continue

                break


            #concateno ogni heading e body per creare il testo completo del documento
            chunk_info = ChunkInfo(
                heading=heading,
                body=body,
                pag_number=pag,
            )
            chunks_data.append(chunk_info)
           

        doc_info: DocInfo = self._extract_document_info(
            document=document_txt
        )
        # ora ho la lista di chunk
        """
        - per ogni chunk, ricostruisco il testo completo
        - estraggo i numeri di pagina
        - cerco nella dish_list di doc_info se il piatto e' nel chunk tramite page number
        - creo il chunk con il testo e la metadata completa
        """
        final_chunks: list[Chunk] = []
        for chunk in chunks_data: 
            pag = chunk.pag_number
            txt = f"{chunk.heading}\n{chunk.body}".strip()
            # map dishes nella doc_info.dish_info
            # per ogni piatto lo associo al chunk se il heading coincide
            for dish in doc_info.dish_info:
                chunk.metadata["restaurant_name"] = doc_info.restaurant_name
                chunk.metadata["chef_name"] = doc_info.chef_name
                chunk.metadata["summary"] = doc_info.document_summary
                if _clean_heading_text(chunk.heading).lower().strip() == _clean_heading_text(dish.heading).lower().strip():
                    chunk.metadata["dish_info"] = dish.model_dump()

            final_chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    text=txt,
                    metadata=chunk.metadata,
                )
            )
        return final_chunks



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
            structured = chunk.metadata
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
        base_url=settings.base_url,
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
        modules=[parser, splitter, chunk_embedder, 
                #  contextual_embedder
                 ],
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
    logger.info("Provider: %s", settings.provider)
    target_dir = menu_dir or (settings.knowledge_base_dir / "menu")
    if not target_dir.exists():
        raise typer.BadParameter(f"Menu directory {target_dir} does not exist")

    pdfs = list(_iter_pdfs(target_dir))
    if not pdfs:
        raise typer.Exit("No PDF files found in menu directory")
    if limit > 0:
        pdfs = pdfs[:limit]

    docling_json_dir = docling_json_dir or settings.docling_json_output_dir

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
