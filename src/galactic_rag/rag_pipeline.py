"""RAG pipeline assembly utilities for Galactic menu questions."""

from __future__ import annotations

import logging
from pathlib import Path

from datapizza.clients.openai_like import OpenAILikeClient
from datapizza.core.models import PipelineComponent
from datapizza.embedders.openai import OpenAIEmbedder
from datapizza.embedders.google import GoogleEmbedder
from datapizza.memory import Memory
from datapizza.modules.prompt.prompt import ChatPromptTemplate
from datapizza.modules.rewriters.tool_rewriter import ToolRewriter
from datapizza.pipeline import DagPipeline
from datapizza.type import Chunk
from datapizza.vectorstores.qdrant import QdrantVectorstore
from pydantic import BaseModel, Field, field_validator
from galactic_rag.config import Settings
from galactic_rag.rag_postprocess import DishPostProcessor

logger = logging.getLogger(__name__)

REWRITER_SYSTEM_PROMPT = (
    "Sei un assistente intelligente che riformula le domande degli utenti cosi da aumentare la accuracy del sistema di retrieval. "
)

ANSWER_SYSTEM_PROMPT = (
    "Sei uno chef enciclopedico. Usa solo i documenti forniti per individuare i piatti "
    "galattici richiesti. Rispondi esclusivamente con JSON della forma "
    '{"piatti": ["nome piatto 1", "nome piatto 2"]}' 
    "Se non sei certo, scegli i piatti "
    "più pertinenti in base ai documenti."
)

USER_PROMPT_TEMPLATE = (
    "Domanda dell'utente:\n"
    "{{ user_prompt }}\n\n"
    "Produci un elenco di piatti coerente e completo, basandoti solo sui documenti."
)

RETRIEVAL_PROMPT_TEMPLATE = (
    "Contenuti recuperati dalla knowledge base:\n"
    "{% for chunk in chunks %}"
    "[CHUNK {{ loop.index }}]\n"
    "Ristorante: {{ chunk.metadata.get('restaurant', 'n/d') }}\n"
    "Chef: {{ chunk.metadata.get('chef', 'n/d') }}\n"
    "Piatto: {{ chunk.metadata.get('dish_name', chunk.metadata.get('section_heading', 'n/d')) }}\n"
    "Dettagli: {{ chunk.text }}\n\n"
    "{% endfor %}"
)


class FilterSpec(BaseModel):
    ingredients: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    restaurant: list[str] = Field(default_factory=list)
    chef: list[str] = Field(default_factory=list)

    @field_validator("*", mode="before")
    @classmethod
    def _normalize(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]
        if isinstance(value, (set, tuple)):
            value = list(value)
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [str(value).strip()]


class QueryEmbedder(PipelineComponent):
    """Embed the rewritten query so it can be used by the retriever."""

    def __init__(self, embedder: OpenAIEmbedder, model_name: str):
        self.embedder = embedder
        self.model_name = model_name

    def _run(self, text: str | None = None, fallback_prompt: str | None = None) -> list[float]:
        query_text = (text or fallback_prompt or "").strip()
        if not query_text:
            raise ValueError("QueryEmbedder received an empty query")
        return self.embedder.embed(query_text, self.model_name)


class PromptBuilder(PipelineComponent):
    """Create a Memory object that combines the user prompt and retrieved chunks."""

    def __init__(self, template: ChatPromptTemplate):
        self.template = template

    def _run(
        self,
        user_prompt: str,
        retrieval_query: str | None = None,
        chunks: list[Chunk] | None = None,
        memory: Memory | None = None,
    ) -> Memory:
        retrieval = retrieval_query or user_prompt
        return self.template.format(
            memory=memory,
            chunks=chunks,
            user_prompt=user_prompt,
            retrieval_query=retrieval,
        )


class FilterExtractor(PipelineComponent):
    """Use an LLM to derive metadata filters for the retriever."""

    def __init__(self, client: OpenAILikeClient | None = None):
        self.client = client
        self.prompt = (
        "Estrai eventuali ingredienti, tecniche, ristoranti o chef citati dalla seguente richiesta.\n"
        "Rispondi solo con JSON della forma {{\"ingredients\": [], \"techniques\": [], \"restaurant\": [], \"chef\": []}}.\n"
        "Usa esattamente questi nomi di chiave anche se gli array rimangono vuoti e non aggiungere testo libero.\n\n"
        "Esempi:\n"
        "Domanda: Quali sono i piatti che includono le Chocobo Wings come ingrediente?\n"
        'Risposta: {{"ingredients": ["Chocobo Wings"], "techniques": [], "restaurant": [], "chef": []}}\n\n'
        "Domanda: Quali piatti dovrei scegliere per un banchetto a tema magico che includa le celebri Cioccorane?\n"
        'Risposta: {{"ingredients": ["Cioccorane"], "techniques": [], "restaurant": [], "chef": []}}\n\n'
        "Domanda: Quali piatti usano la Sferificazione Filamentare a Molecole Vibrazionali, ma evitano la Decostruzione Magnetica Risonante?\n"
        'Risposta: {{"ingredients": [], "techniques": ["Sferificazione Filamentare a Molecole Vibrazionali"], "restaurant": [], "chef": []}}\n\n'
        "Domanda da cui devi estrarre i dati: {question}\n"
    )


    def _run(self, user_prompt: str | None = None) -> dict | None:
        if not self.client or not user_prompt:
            return None
        try:
            response = self.client.structured_response(
                input=self.prompt.format(question=user_prompt),
                output_cls=FilterSpec,
                temperature=0.0,
            )
        except KeyError as exc:  # noqa: BLE001
            logger.warning("Filter extractor missing field: %s", exc)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Filter extractor failed: %s", exc)
            return None
        if not response.structured_data:
            return None
        spec: FilterSpec = response.structured_data[0]
        return self._build_filter(spec)

    def _build_filter(self, spec: FilterSpec) -> dict | None:
        must: list[dict] = []
        for ingredient in spec.ingredients:
            must.append({"key": "text", "match": {"text": ingredient}})
        for technique in spec.techniques:
            must.append({"key": "text", "match": {"text": technique}})
        for restaurant in spec.restaurant:
            must.append({"key": "text", "match": {"text": restaurant}})
        for chef in spec.chef:
            must.append({"key": "text", "match": {"text": chef}})
        if not must:
            return None
        return {"must": must}


def build_rag_pipeline(
    *,
    settings: Settings,
    vectorstore: QdrantVectorstore,
    dish_mapping_path: Path | None = None,
) -> DagPipeline:
    """Assemble the Datapizza DAG pipeline used to answer questions."""

    dish_mapping = dish_mapping_path or settings.dish_mapping_json

    rewriter_client = OpenAILikeClient(
        api_key=settings.api_key,
        model=settings.llm_model,
        system_prompt=REWRITER_SYSTEM_PROMPT,
        temperature=0.0,
        base_url=settings.base_url,
    )
    rewriter = ToolRewriter(
        client=rewriter_client,
        system_prompt=REWRITER_SYSTEM_PROMPT,
        tool_choice="required",
        tool_output_name="query",
        temperature=0.0,
    )

    if settings.provider == "gemini":
        embedder_client = GoogleEmbedder(
            api_key=settings.api_key,
            model_name=settings.embedding_model,
            base_url=settings.base_url,
        )
    elif settings.provider == "openai":
        embedder_client = OpenAIEmbedder(
            api_key=settings.api_key,
            model_name=settings.embedding_model,
            base_url=settings.base_url,
        )
    query_embedder = QueryEmbedder(embedder=embedder_client, model_name=settings.embedding_model)

    prompt_template = ChatPromptTemplate(
        user_prompt_template=USER_PROMPT_TEMPLATE,
        retrieval_prompt_template=RETRIEVAL_PROMPT_TEMPLATE,
    )
    prompt_builder = PromptBuilder(prompt_template)

    generator_client = OpenAILikeClient(
        api_key=settings.api_key,
        model=settings.llm_model,
        system_prompt=ANSWER_SYSTEM_PROMPT,
        temperature=0.0,
        base_url=settings.base_url,
    )
    generator = generator_client.as_inference_module_component()

    postprocess = DishPostProcessor(mapping_path=dish_mapping)

    filter_extractor = FilterExtractor(client=rewriter_client)

    pipeline = DagPipeline()
    pipeline.add_module("rewriter", rewriter)
    pipeline.add_module("filter_builder", filter_extractor)
    pipeline.add_module("query_embedder", query_embedder)
    pipeline.add_module("retriever", vectorstore.as_retriever())
    pipeline.add_module("prompt_builder", prompt_builder)
    pipeline.add_module("generator", generator)
    pipeline.add_module("postprocess", postprocess)

    pipeline.connect("rewriter", "query_embedder", target_key="text")
    pipeline.connect("rewriter", "prompt_builder", target_key="retrieval_query")
    # pipeline.connect("filter_builder", "retriever", target_key="query_filter")
    pipeline.connect("query_embedder", "retriever", target_key="query_vector")
    pipeline.connect("retriever", "prompt_builder", target_key="chunks")
    pipeline.connect("prompt_builder", "generator", target_key="memory")
    pipeline.connect("retriever", "postprocess", target_key="chunks")
    pipeline.connect("generator", "postprocess", target_key="response")

    logger.info("RAG pipeline ready: provider=%s, collection=%s", settings.provider, settings.qdrant_collection)
    return pipeline
