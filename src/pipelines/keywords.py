import asyncio
from collections import Counter
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.parquet as pq
from openai import AsyncOpenAI
from sklearn.feature_extraction.text import CountVectorizer
from tqdm.auto import tqdm

from datasets import Dataset, load_dataset
from src.config import KeywordsConfig, config
from src.datasets.embeddings import build_embeddings_dataset
from src.datasets.jokes import build_jokes_dataset
from src.logging import get_logger
from src.models import KeywordsInputs, KeywordsOutputs
from src.paths import DATA_DIR
from src.pipelines.base import BasePipeline
from src.settings import settings

if TYPE_CHECKING:
    import polars as pl


logger = get_logger(__name__)


def _cosine_relevance_scores(
    joke_embedding: npt.NDArray[np.float32],
    candidate_embeddings: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    joke_norm = np.linalg.norm(joke_embedding)
    candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
    relevance_scores = np.zeros(candidate_embeddings.shape[0], dtype=np.float32)
    denominator = candidate_norms * joke_norm
    valid_mask = denominator > 0

    if np.any(valid_mask):
        relevance_scores[valid_mask] = (candidate_embeddings[valid_mask] @ joke_embedding) / denominator[valid_mask]
    return relevance_scores


def _select_top_indices_with_mmr(
    candidate_embeddings: npt.NDArray[np.float32],
    relevance_scores: npt.NDArray[np.float32],
    top_n: int,
    diversity: float,
) -> npt.NDArray[np.int64]:
    candidate_count = candidate_embeddings.shape[0]
    if candidate_count == 0 or top_n <= 0:
        return np.array([], dtype=np.int64)

    max_keywords = min(top_n, candidate_count)
    selected_indices: list[int] = []
    available_mask = np.ones(candidate_count, dtype=bool)

    first_index = int(np.argmax(relevance_scores))
    selected_indices.append(first_index)
    available_mask[first_index] = False

    candidate_norms = np.linalg.norm(candidate_embeddings, axis=1)
    normalized_vectors = np.zeros_like(candidate_embeddings)
    nonzero_mask = candidate_norms > 0
    normalized_vectors[nonzero_mask] = candidate_embeddings[nonzero_mask] / candidate_norms[nonzero_mask, np.newaxis]

    while len(selected_indices) < max_keywords and np.any(available_mask):
        available_indices = np.flatnonzero(available_mask)
        selected_matrix = normalized_vectors[np.array(selected_indices, dtype=np.int64)]
        redundancy_scores = normalized_vectors[available_indices] @ selected_matrix.T
        max_redundancy = np.max(redundancy_scores, axis=1)
        mmr_scores = ((1.0 - diversity) * relevance_scores[available_indices]) - (diversity * max_redundancy)
        best_local_index = int(np.argmax(mmr_scores))
        best_index = int(available_indices[best_local_index])
        selected_indices.append(best_index)
        available_mask[best_index] = False

    return np.array(selected_indices, dtype=np.int64)


class KeywordsPipeline(BasePipeline):
    def __init__(
        self,
        pipeline_config: KeywordsConfig | None = None,
        output_dir: Path | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.config = pipeline_config or config.keywords
        self.output_dir = output_dir or DATA_DIR / self.config.hf_config_name
        self.client = client or AsyncOpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=self.config.timeout,
        )
        self.next_part_index = 0

        self.schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("keywords", pa.list_(pa.string())),
                pa.field("scores", pa.list_(pa.float32())),
            ]
        )

        ngram_range = (self.config.ngram_min, self.config.ngram_max)
        stop_words = "english" if self.config.stopwords else None
        self._candidate_analyzer = CountVectorizer(
            ngram_range=ngram_range,
            stop_words=stop_words,
        ).build_analyzer()

    def _extract_candidates(self, text: str) -> list[str]:
        cleaned_text = text.strip()
        if not cleaned_text:
            return []

        candidate_counts = Counter(token.strip() for token in self._candidate_analyzer(cleaned_text))
        if not candidate_counts:
            return []

        return [candidate for candidate, _ in candidate_counts.most_common(self.config.max_candidates) if candidate]

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await self.client.embeddings.create(
                    model=self.config.model,
                    input=batch,
                    dimensions=self.config.dimensions,
                )
                embeddings = [item.embedding for item in response.data]
            except Exception:
                if attempt >= self.config.max_retries:
                    raise
                await asyncio.sleep(2 ** (attempt - 1))
            else:
                return embeddings

        msg = "Unexpected retry error."
        raise RuntimeError(msg)

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for batch in batched(texts, self.config.batch_size, strict=False):
            embeddings.extend(await self._embed_batch(list(batch)))
        return embeddings

    async def _extract_keywords(
        self,
        inputs: KeywordsInputs,
        semaphore: asyncio.Semaphore,
    ) -> KeywordsOutputs:
        async with semaphore:
            candidates = self._extract_candidates(inputs.text)
            if not candidates:
                return KeywordsOutputs(id=inputs.id, keywords=[], scores=[])

            candidate_embeddings = await self._embed_texts(candidates)

            candidate_embeddings = np.asarray(candidate_embeddings, dtype=np.float32)
            joke_embedding = np.asarray(inputs.embedding, dtype=np.float32)

            relevance_scores = _cosine_relevance_scores(
                joke_embedding=joke_embedding,
                candidate_embeddings=candidate_embeddings,
            )
            selected_indices = _select_top_indices_with_mmr(
                candidate_embeddings=candidate_embeddings,
                relevance_scores=relevance_scores,
                top_n=self.config.top_n,
                diversity=self.config.mmr_diversity,
            )

            keywords = []
            scores = []
            for index in selected_indices.tolist():
                keywords.append(candidates[index])
                scores.append(relevance_scores[index])

            return KeywordsOutputs(id=inputs.id, keywords=keywords, scores=scores)

    def _get_table(self, write_buffer: list[KeywordsOutputs]) -> pa.Table:
        outputs = [item.model_dump() for item in write_buffer]
        return pa.Table.from_pylist(outputs, schema=self.schema)

    def _check_buffer_size(self, write_buffer: list[KeywordsOutputs]) -> bool:
        return len(write_buffer) >= self.config.shard_size

    async def run(
        self,
        jokes: Dataset,
        embeddings: Dataset,
        resume: bool = False,
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.next_part_index = self._get_next_part_index()

        jokes_frame = cast("pl.DataFrame", jokes.to_polars())
        embeddings_frame = cast("pl.DataFrame", embeddings.to_polars())
        joined_frame = jokes_frame.join(embeddings_frame, on="id", how="inner").select(["id", "text", "embedding"])
        dataset = Dataset.from_polars(joined_frame)

        if resume:
            seen_ids = self._get_seen_ids()
            dataset = dataset.filter(lambda item: item["id"] not in seen_ids)
        elif self.next_part_index == 0:
            for file in self.output_dir.iterdir():
                file.unlink()

        write_buffer: list[KeywordsOutputs] = []
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        pending_tasks: set[asyncio.Task[KeywordsOutputs]] = set()

        for item in tqdm(dataset):
            item = cast("dict[str, Any]", item)
            inputs = KeywordsInputs(
                id=item["id"],
                text=item["text"],
                embedding=item["embedding"],
            )

            task = asyncio.create_task(self._extract_keywords(inputs=inputs, semaphore=semaphore))
            pending_tasks.add(task)

            if len(pending_tasks) >= self.config.max_parallel_requests:
                await self._wait_one(
                    pending_tasks=pending_tasks,
                    write_buffer=write_buffer,
                )

        while pending_tasks:
            await self._wait_one(
                pending_tasks=pending_tasks,
                write_buffer=write_buffer,
            )

        self._flush_buffer(
            write_buffer=write_buffer,
        )

        logger.info(
            "run.done",
            model=self.config.model,
            output_dir=str(self.output_dir),
        )

        return self.output_dir


async def main() -> None:
    jokes_path = DATA_DIR / config.jokes.data_filename
    if not jokes_path.exists():
        jokes_path = build_jokes_dataset()

    embeddings_dir = DATA_DIR / config.embeddings.hf_config_name
    if not embeddings_dir.exists():
        embeddings_dir = build_embeddings_dataset().data_path

    jokes = load_dataset("parquet", data_files=str(jokes_path), split="train[:50]")
    embeddings = load_dataset("parquet", data_dir=str(embeddings_dir), split="train")

    pipeline = KeywordsPipeline()
    output_dir = await pipeline.run(jokes, embeddings, resume=True)
    print(
        {
            "jokes_path": str(jokes_path),
            "keywords_dir": str(output_dir),
        }
    )

    keywords = load_dataset("parquet", data_dir=str(output_dir), split="train[:]")
    print(keywords[:])


if __name__ == "__main__":
    asyncio.run(main())
