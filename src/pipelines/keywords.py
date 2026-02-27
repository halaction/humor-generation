import asyncio
import math
from collections.abc import Iterable
from itertools import batched
from pathlib import Path
from typing import Any, cast

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from openai import AsyncOpenAI
from pydantic import BaseModel
from sklearn.feature_extraction.text import CountVectorizer

from datasets import Dataset
from src.config import EmbeddingsConfig, KeywordsConfig, config
from src.paths import DATA_DIR
from src.settings import settings


class KeywordsItem(BaseModel):
    joke_id: str
    keywords: list[str]
    scores: list[float]


class KeywordsOutputs(BaseModel):
    results: list[KeywordsItem]
    data_path: Path


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def _select_top_indices_with_mmr(
    candidate_vectors: list[list[float]],
    relevance_scores: list[float],
    top_n: int,
    diversity: float,
) -> list[int]:
    if not candidate_vectors or top_n <= 0:
        return []

    max_keywords = min(top_n, len(candidate_vectors))
    selected_indices: list[int] = []
    available_indices = set(range(len(candidate_vectors)))

    first_index = max(available_indices, key=lambda idx: relevance_scores[idx])
    selected_indices.append(first_index)
    available_indices.remove(first_index)

    while len(selected_indices) < max_keywords and available_indices:
        best_index = None
        best_score = -math.inf
        for candidate_index in available_indices:
            redundancy = max(
                _cosine_similarity(candidate_vectors[candidate_index], candidate_vectors[chosen_index])
                for chosen_index in selected_indices
            )
            mmr_score = ((1.0 - diversity) * relevance_scores[candidate_index]) - (diversity * redundancy)
            if mmr_score > best_score:
                best_score = mmr_score
                best_index = candidate_index

        if best_index is None:
            break
        selected_indices.append(best_index)
        available_indices.remove(best_index)

    return selected_indices


class KeywordsPipeline:
    def __init__(
        self,
        *,
        keywords_config: KeywordsConfig | None = None,
        embeddings_config: EmbeddingsConfig | None = None,
        client: Any | None = None,
    ) -> None:
        self.config = keywords_config or config.keywords
        self.embedding_config = embeddings_config or config.embeddings
        if self.config.ngram_min > self.config.ngram_max:
            msg = (
                f"Invalid n-gram range: ngram_min={self.config.ngram_min} must be <= ngram_max={self.config.ngram_max}."
            )
            raise ValueError(msg)

        self.stop_words = "english" if self.config.stopwords else None
        self.client = client or AsyncOpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=self.config.timeout,
        )
        self.embedding_batch_size = max(1, min(self.config.batch_size, self.embedding_config.batch_size))

    def _extract_candidates(self, text: str) -> list[str]:
        cleaned_text = text.strip()
        if not cleaned_text:
            return []

        vectorizer = CountVectorizer(
            ngram_range=(self.config.ngram_min, self.config.ngram_max),
            stop_words=self.stop_words,
            max_features=self.config.max_candidates,
        )
        try:
            vectorizer.fit([cleaned_text])
        except ValueError:
            return []

        return [str(candidate).strip() for candidate in vectorizer.get_feature_names_out() if str(candidate).strip()]

    async def _embed_text_batch(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.embeddings.create(
                    model=self.embedding_config.model,
                    input=texts,
                    dimensions=self.embedding_config.dimensions,
                )
            except Exception:
                if attempt + 1 >= self.config.max_retries:
                    raise
                await asyncio.sleep(2**attempt)
            else:
                vectors = [item.embedding for item in response.data]
                if len(vectors) != len(texts):
                    msg = f"Embedding API returned mismatched outputs: expected={len(texts)} got={len(vectors)}"
                    raise ValueError(msg)
                return vectors

        msg = "Unexpected embedding retry flow."
        raise RuntimeError(msg)

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in batched(texts, self.embedding_batch_size):
            embeddings.extend(await self._embed_text_batch(list(batch)))
        return embeddings

    async def _process_joke(
        self,
        joke_id: str,
        joke: str,
        joke_embedding: list[float],
        semaphore: asyncio.Semaphore,
    ) -> KeywordsItem:
        async with semaphore:
            cleaned_joke = str(joke).strip()
            candidates = self._extract_candidates(cleaned_joke)
            if not candidates:
                return KeywordsItem(joke_id=joke_id, keywords=[], scores=[])

            candidate_vectors = await self._embed_texts(candidates)

            relevance_scores = [_cosine_similarity(joke_embedding, vector) for vector in candidate_vectors]
            selected_indices = _select_top_indices_with_mmr(
                candidate_vectors=candidate_vectors,
                relevance_scores=relevance_scores,
                top_n=self.config.top_n,
                diversity=self.config.mmr_diversity,
            )
            keywords = [candidates[index] for index in selected_indices]
            scores = [relevance_scores[index] for index in selected_indices]
            return KeywordsItem(joke_id=joke_id, keywords=keywords, scores=scores)

    @staticmethod
    def _load_existing_outputs(path: Path) -> dict[str, KeywordsItem]:
        if not path.exists():
            return {}

        table = pq.read_table(path)
        required_columns = {"joke_id", "keywords", "scores"}
        if not required_columns.issubset(set(table.schema.names)):
            return {}

        outputs: dict[str, KeywordsItem] = {}
        for row in table.select(["joke_id", "keywords", "scores"]).to_pylist():
            item = KeywordsItem.model_validate(row)
            outputs[item.joke_id] = item
        return outputs

    @staticmethod
    def _write_parquet(outputs: list[KeywordsItem], output_path: Path) -> None:
        table = pa.Table.from_pylist(
            [item.model_dump(mode="python") for item in outputs],
            schema=pa.schema(
                [
                    pa.field("joke_id", pa.string()),
                    pa.field("keywords", pa.list_(pa.string())),
                    pa.field("scores", pa.list_(pa.float32())),
                ]
            ),
        )
        pq.write_table(
            table,
            output_path,
            compression="zstd",
            use_content_defined_chunking=True,
            write_page_index=True,
        )

    @staticmethod
    def _sort_joke_id(value: str) -> tuple[int, Any]:
        return (0, int(value)) if value.isdigit() else (1, value)

    def _consume_completed(
        self,
        completed_tasks: Iterable[asyncio.Task[KeywordsItem]],
        seen_ids: set[str],
        outputs: list[KeywordsItem],
    ) -> None:
        for task in completed_tasks:
            result = task.result()
            if result.joke_id in seen_ids:
                continue
            seen_ids.add(result.joke_id)
            outputs.append(result)

    async def run(
        self,
        jokes: Dataset,
        embeddings: Dataset,
    ) -> KeywordsOutputs:
        if "id" not in jokes.column_names:
            msg = "Jokes dataset must contain an 'id' column."
            raise ValueError(msg)
        if "text" not in jokes.column_names:
            msg = "Jokes dataset must contain a 'text' column."
            raise ValueError(msg)
        if "id" not in embeddings.column_names:
            msg = "Embeddings dataset must contain an 'id' column."
            raise ValueError(msg)
        if "embedding" not in embeddings.column_names:
            msg = "Embeddings dataset must contain an 'embedding' column."
            raise ValueError(msg)

        jokes_frame = pl.from_arrow(cast(pa.Table, cast("Any", jokes).data.table).select(["id", "text"]))
        embeddings_frame = pl.from_arrow(cast(pa.Table, cast("Any", embeddings).data.table).select(["id", "embedding"]))
        joined_frame = jokes_frame.join(embeddings_frame, on="id", how="inner").select(["id", "text", "embedding"])

        output_path = DATA_DIR / self.config.data_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        existing_outputs = self._load_existing_outputs(output_path)
        seen_ids = set(existing_outputs)
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        outputs: list[KeywordsItem] = []
        pending_tasks: set[asyncio.Task[KeywordsItem]] = set()

        for joke_id, joke_text, joke_embedding in joined_frame.iter_rows():
            if joke_id in seen_ids:
                continue
            if len(joke_embedding) != self.embedding_config.dimensions:
                msg = (
                    f"Inconsistent embedding size for id={joke_id}: expected "
                    f"{self.embedding_config.dimensions}, got {len(joke_embedding)}"
                )
                raise ValueError(msg)

            task = asyncio.create_task(
                self._process_joke(
                    joke_id=joke_id,
                    joke=joke_text,
                    joke_embedding=joke_embedding,
                    semaphore=semaphore,
                )
            )
            pending_tasks.add(task)

            if len(pending_tasks) >= self.config.max_parallel_requests:
                done, _ = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                pending_tasks -= done
                self._consume_completed(completed_tasks=done, seen_ids=seen_ids, outputs=outputs)

        while pending_tasks:
            done, _ = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            pending_tasks -= done
            self._consume_completed(completed_tasks=done, seen_ids=seen_ids, outputs=outputs)

        merged_outputs = list(existing_outputs.values())
        merged_outputs.extend(outputs)
        merged_outputs.sort(key=lambda item: self._sort_joke_id(item.joke_id))
        self._write_parquet(outputs=merged_outputs, output_path=output_path)
        return KeywordsOutputs(results=outputs, data_path=output_path)
