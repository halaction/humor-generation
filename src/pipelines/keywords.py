import asyncio
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq
from openai import AsyncOpenAI
from pydantic import BaseModel

from datasets import Dataset
from src.config import EmbeddingsConfig, KeywordsConfig, config
from src.paths import DATA_DIR
from src.settings import settings

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")
KEYBERT_ENGLISH_STOPWORDS = frozenset(
    {
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "doing",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "has",
        "have",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "now",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "she",
        "should",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "with",
        "would",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }
)


class KeywordsItem(BaseModel):
    joke_id: str
    keywords: list[str]
    scores: list[float]


ExtractionInputs = Dataset
KeywordsOutputs = list[KeywordsItem]


def _extract_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _generate_ngram_candidates(
    text: str,
    *,
    ngram_min: int,
    ngram_max: int,
    max_candidates: int,
    stopwords_mode: Literal["none", "english"],
) -> list[str]:
    tokens = _extract_tokens(text)
    if not tokens:
        return []

    seen: set[str] = set()
    candidates: list[str] = []
    upper = min(ngram_max, len(tokens))
    for ngram_size in range(ngram_min, upper + 1):
        for index in range(len(tokens) - ngram_size + 1):
            candidate = " ".join(tokens[index : index + ngram_size])
            if candidate in seen:
                continue
            if stopwords_mode == "english":
                if any(token in KEYBERT_ENGLISH_STOPWORDS for token in candidate.split()):
                    continue
            seen.add(candidate)
            candidates.append(candidate)
            if len(candidates) >= max_candidates:
                return candidates

    return candidates


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def _apply_length_penalty(score: float, candidate: str, alpha: float) -> float:
    if alpha == 0.0:
        return score
    ngram_len = len(candidate.split())
    return score - (alpha * max(ngram_len - 1, 0))


def _select_top_indices_with_mmr(
    *,
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
        self.client = client or AsyncOpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=self.config.timeout,
        )

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
        for index in range(0, len(texts), self.config.batch_size):
            batch = texts[index : index + self.config.batch_size]
            embeddings.extend(await self._embed_text_batch(batch))
        return embeddings

    async def _process_joke(self, joke_id: str, joke: str, semaphore: asyncio.Semaphore) -> KeywordsItem:
        async with semaphore:
            cleaned_joke = str(joke).strip()
            candidates = _generate_ngram_candidates(
                cleaned_joke,
                ngram_min=self.config.ngram_min,
                ngram_max=self.config.ngram_max,
                max_candidates=self.config.max_candidates,
                stopwords_mode=self.config.stopwords,
            )
            if not candidates:
                return KeywordsItem(joke_id=joke_id, keywords=[], scores=[])

            texts = [cleaned_joke, *candidates]
            vectors = await self._embed_texts(texts)
            joke_embedding = vectors[0]
            candidate_vectors = vectors[1:]

            relevance_scores: list[float] = [
                _apply_length_penalty(
                    score=_cosine_similarity(joke_embedding, vector),
                    candidate=candidate,
                    alpha=self.config.length_penalty_alpha,
                )
                for candidate, vector in zip(candidates, candidate_vectors, strict=True)
            ]
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
        outputs: dict[str, KeywordsItem] = {}
        for joke_id, keywords, scores in zip(
            table.column("joke_id").to_pylist(),
            table.column("keywords").to_pylist(),
            table.column("scores").to_pylist(),
            strict=True,
        ):
            normalized_keywords = [str(item).strip() for item in (keywords or []) if str(item).strip()]
            normalized_scores = [float(item) for item in (scores or [])]
            count = min(len(normalized_keywords), len(normalized_scores))
            outputs[str(joke_id)] = KeywordsItem(
                joke_id=str(joke_id),
                keywords=normalized_keywords[:count],
                scores=normalized_scores[:count],
            )
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

    def _consume_completed(
        self,
        completed_tasks: Iterable[asyncio.Task[KeywordsItem]],
        seen_ids: set[str],
        outputs: KeywordsOutputs,
    ) -> None:
        for task in completed_tasks:
            result = task.result()
            if result.joke_id in seen_ids:
                continue
            seen_ids.add(result.joke_id)
            outputs.append(result)

    async def run(
        self,
        inputs: ExtractionInputs,
    ) -> tuple[KeywordsOutputs, Path]:
        if "text" not in inputs.column_names:
            msg = "Dataset must contain a 'text' column."
            raise ValueError(msg)
        if "id" not in inputs.column_names:
            msg = "Dataset must contain an 'id' column."
            raise ValueError(msg)

        output_path = DATA_DIR / self.config.data_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        existing_outputs = self._load_existing_outputs(output_path)
        seen_ids = set(existing_outputs)
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        outputs: KeywordsOutputs = []
        pending_tasks: set[asyncio.Task[KeywordsItem]] = set()
        for row in inputs:
            payload = cast("dict[str, Any]", row)
            joke_id = str(payload["id"])
            if joke_id in seen_ids:
                continue

            joke_text = str(payload["text"])
            task = asyncio.create_task(self._process_joke(joke_id=joke_id, joke=joke_text, semaphore=semaphore))
            pending_tasks.add(task)

            if len(pending_tasks) >= self.config.max_parallel_requests:
                done, _ = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                pending_tasks -= done
                self._consume_completed(
                    completed_tasks=done,
                    seen_ids=seen_ids,
                    outputs=outputs,
                )

        while pending_tasks:
            done, _ = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
            pending_tasks -= done
            self._consume_completed(
                completed_tasks=done,
                seen_ids=seen_ids,
                outputs=outputs,
            )

        merged_outputs = list(existing_outputs.values())
        merged_outputs.extend(outputs)
        merged_outputs.sort(key=lambda item: int(item.joke_id) if item.joke_id.isdigit() else item.joke_id)
        self._write_parquet(outputs=merged_outputs, output_path=output_path)
        return outputs, output_path
