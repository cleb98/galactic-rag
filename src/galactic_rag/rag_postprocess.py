"""Post-processing utilities for RAG answers."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Sequence

from datapizza.core.clients.models import ClientResponse
from datapizza.core.models import PipelineComponent
from datapizza.type import Chunk

logger = logging.getLogger(__name__)


class DishPostProcessor(PipelineComponent):
    """Convert model output into CSV-ready dish IDs."""

    def __init__(self, mapping_path: Path, fallback_chunks: int = 3):
        self.mapping_path = mapping_path
        self.fallback_chunks = fallback_chunks
        self._name_to_id = self._load_mapping(mapping_path)

    def _run(
        self,
        response: ClientResponse,
        chunks: Sequence[Chunk] | None = None,
        question: str = "",
        row_id: int | None = None,
        difficulty: str | None = None,
    ) -> dict[str, object]:
        dish_names = self._extract_dish_names(response)
        if not dish_names:
            dish_names = self._fallback_from_chunks(chunks)

        dish_ids = self._map_to_ids(dish_names)
        if not dish_ids:
            fallback_names = self._fallback_from_chunks(chunks)
            dish_ids = self._map_to_ids(fallback_names)
            if fallback_names:
                dish_names = fallback_names

        if not dish_ids and self._name_to_id:
            # deterministic final fallback: take the first available ID
            dish_ids = [next(iter(self._name_to_id.values()))]
            logger.warning(
                "Falling back to default dish ID for row %s (question=%s)",
                row_id,
                question,
            )

        result = ",".join(str(idx) for idx in dish_ids)
        return {
            "result": result,
            "dishes": dish_names,
            "row_id": row_id,
            "question": question,
            "difficulty": difficulty,
        }

    def _load_mapping(self, path: Path) -> dict[str, int]:
        if not path.exists():
            raise FileNotFoundError(f"Dish mapping file not found: {path}")
        with path.open("r", encoding="utf-8") as mapping_file:
            data = json.load(mapping_file)
        normalized = {self._normalize(name): int(idx) for name, idx in data.items()}
        return normalized

    def _extract_dish_names(self, response: ClientResponse) -> list[str]:
        text = response.text.strip()
        if not text:
            return []

        json_blob = self._extract_json_blob(text)
        if json_blob:
            try:
                parsed = json.loads(json_blob)
            except json.JSONDecodeError:
                logger.debug("Could not decode JSON blob: %s", json_blob)
            else:
                return self._names_from_parsed(parsed)

        return self._split_candidates(text)

    def _extract_json_blob(self, text: str) -> str | None:
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if fenced:
            return fenced[0].strip()

        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return brace_match.group(0)

        list_match = re.search(r"\[.*\]", text, re.DOTALL)
        if list_match:
            return list_match.group(0)
        return None

    def _names_from_parsed(self, data) -> list[str]:
        if isinstance(data, dict):
            for key in ("piatti", "dishes", "results", "risultati"):
                value = data.get(key)
                if isinstance(value, list):
                    return [str(item).strip() for item in value if str(item).strip()]
                if isinstance(value, str):
                    return self._split_candidates(value)
            # fallback to all string fields
            values = [str(v).strip() for v in data.values() if isinstance(v, str)]
            if values:
                return self._split_candidates(values[0])
        if isinstance(data, list):
            return [str(item).strip() for item in data if str(item).strip()]
        if isinstance(data, str):
            return self._split_candidates(data)
        return []

    def _split_candidates(self, value: str) -> list[str]:
        cleaned = [v.strip(" \n\t\"')") for v in value.split(",")]
        return [v for v in cleaned if v]

    def _fallback_from_chunks(self, chunks: Sequence[Chunk] | None) -> list[str]:
        if not chunks:
            return []
        candidates: list[str] = []
        for chunk in chunks[: self.fallback_chunks]:
            metadata = chunk.metadata or {}
            dish_name = metadata.get("dish_name")
            if not dish_name:
                menu_info = metadata.get("menu_info")
                if isinstance(menu_info, dict):
                    dish_name = menu_info.get("dish_name")
            if not dish_name:
                structured = metadata.get("structured_text") or ""
                match = re.search(r"Piatto:\s*(.+)", structured)
                if match:
                    dish_name = match.group(1).strip()
            if dish_name:
                candidates.append(dish_name)
        return candidates

    def _map_to_ids(self, names: Sequence[str]) -> list[int]:
        ids: list[int] = []
        for name in names:
            normalized = self._normalize(name)
            dish_id = self._name_to_id.get(normalized)
            if dish_id is not None:
                ids.append(dish_id)
            else:
                logger.debug("Dish '%s' not found in mapping", name)
        return ids

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", value.casefold())
