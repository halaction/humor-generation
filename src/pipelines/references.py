import asyncio
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

from datasets import Dataset, load_dataset
from src.config import ReferencesConfig, config
from src.datasets.embeddings import build_embeddings_dataset
from src.datasets.jokes import build_jokes_dataset
from src.datasets.keywords import build_keywords_dataset
from src.logging import get_logger
from src.paths import DATA_DIR
from src.pipelines.base import BasePipeline
from src.settings import settings
from src.templates import environment

logger = get_logger(__name__)


@dataclass
class _IndexArtifacts:
    index: Any
    ids: list[str]


class _ReferencesBatchOutputs(BaseModel):
    id: list[str]
    prompt: list[str]
    references: list[list[str]]
    scores: list[list[float]]


class ReferencesPipeline(BasePipeline):
    def __init__(
        self,
        pipeline_config: ReferencesConfig | None = None,
        output_dir: Path | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.config = pipeline_config or config.references
        self.output_dir = output_dir or DATA_DIR / self.config.hf_config_name
        self.client = client or AsyncOpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=self.config.timeout,
        )
        self.next_part_index = 0

        self.index_dir = DATA_DIR / self.config.index_dirname
        self.index_path = self.index_dir / "index.faiss"
        self.ids_path = self.index_dir / "ids.parquet"
        self.meta_path = self.index_dir / "meta.json"

        self.prompt_template = environment.get_template("reference_query.j2")

        self.schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("prompt", pa.string()),
                pa.field("references", pa.list_(pa.string())),
                pa.field("scores", pa.list_(pa.float32())),
            ]
        )

    @staticmethod
    def _normalize_vectors(vectors: np.ndarray) -> None:
        if vectors.size == 0:
            return

        norms = np.linalg.norm(vectors, axis=1)
        valid_mask = norms > 0
        vectors[valid_mask] = vectors[valid_mask] / norms[valid_mask, np.newaxis]

    def _render_prompt(self, keywords: list[str]) -> str:
        return self.prompt_template.render(keywords=keywords).strip()

    def _format_query(self, prompt: str) -> str:
        return f"Instruct: {self.config.query_instruction}\nQuery: {prompt}"

    async def _embed_query_batch(self, prompts: list[str]) -> list[list[float]]:
        formatted_queries = [self._format_query(prompt) for prompt in prompts]
        for attempt in range(1, self.config.max_retries + 1):
            try:
                response = await self.client.embeddings.create(
                    model=self.config.model,
                    input=formatted_queries,
                    dimensions=self.config.dimensions,
                )
                return [item.embedding for item in response.data]
            except Exception:
                if attempt >= self.config.max_retries:
                    raise
                await asyncio.sleep(2 ** (attempt - 1))

        msg = "Unexpected retry error."
        raise RuntimeError(msg)

    def _get_table(self, write_buffer: list[_ReferencesBatchOutputs]) -> pa.Table:
        outputs = defaultdict(list)
        for batch in write_buffer:
            for key, value in batch.model_dump().items():
                outputs[key].extend(value)
        return pa.Table.from_pydict(outputs, schema=self.schema)

    def _check_buffer_size(self, write_buffer: list[_ReferencesBatchOutputs]) -> bool:
        return len(write_buffer) * self.config.batch_size >= self.config.shard_size

    def _count_parquet_rows(self, path: Path) -> int:
        return pq.read_metadata(path).num_rows

    def _load_ids(self) -> list[str]:
        table = pq.read_table(self.ids_path, columns=["id"])
        return cast("list[str]", table.column("id").to_pylist())

    def _load_cached_index(self, expected_rows: int) -> _IndexArtifacts | None:
        if not (self.index_path.exists() and self.ids_path.exists() and self.meta_path.exists()):
            return None

        metadata = json.loads(self.meta_path.read_text(encoding="utf-8"))
        effective_nlist = min(self.config.faiss_nlist, max(1, expected_rows))
        if metadata.get("model") != self.config.model:
            return None
        if metadata.get("dimensions") != self.config.dimensions:
            return None
        if metadata.get("rows") != expected_rows:
            return None
        if metadata.get("faiss_nlist_config") != self.config.faiss_nlist:
            return None
        if metadata.get("faiss_nlist_effective") != effective_nlist:
            return None

        ids_row_count = self._count_parquet_rows(self.ids_path)
        if ids_row_count != expected_rows:
            return None

        index = faiss.read_index(str(self.index_path))
        if int(index.ntotal) != expected_rows:
            return None

        if hasattr(index, "nprobe"):
            index.nprobe = min(self.config.faiss_nprobe, effective_nlist)

        logger.info(
            "index.load.done",
            path=str(self.index_path),
            rows=expected_rows,
            nlist=effective_nlist,
        )
        return _IndexArtifacts(index=index, ids=self._load_ids())

    def _sample_training_vectors(self, embeddings: Dataset, sample_size: int) -> np.ndarray:
        reservoir = np.empty((sample_size, self.config.dimensions), dtype=np.float32)
        random_generator = np.random.default_rng(42)

        seen = 0
        for batch in embeddings.iter(batch_size=self.config.faiss_add_batch_size):
            batch = cast("dict[str, list[Any]]", batch)
            vectors = np.asarray(batch["embedding"], dtype=np.float32)
            if vectors.size == 0:
                continue

            self._normalize_vectors(vectors)
            for vector in vectors:
                if seen < sample_size:
                    reservoir[seen] = vector
                else:
                    replacement = int(random_generator.integers(0, seen + 1))
                    if replacement < sample_size:
                        reservoir[replacement] = vector
                seen += 1

        return reservoir[: min(sample_size, seen)]

    def _build_index(self, embeddings: Dataset, expected_rows: int) -> _IndexArtifacts:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        effective_nlist = min(self.config.faiss_nlist, max(1, expected_rows))
        logger.info(
            "index.build.start",
            rows=expected_rows,
            nlist=effective_nlist,
            dimensions=self.config.dimensions,
        )

        quantizer = faiss.IndexFlatIP(self.config.dimensions)
        index = faiss.IndexIVFFlat(
            quantizer,
            self.config.dimensions,
            effective_nlist,
            faiss.METRIC_INNER_PRODUCT,
        )

        if expected_rows > 0:
            training_size = min(self.config.faiss_train_size, expected_rows)
            train_vectors = self._sample_training_vectors(embeddings, training_size)
            if train_vectors.size == 0:
                msg = "Cannot train Faiss index with empty training vectors."
                raise RuntimeError(msg)
            index.train(train_vectors)

        ids: list[str] = []
        total_batches = math.ceil(expected_rows / self.config.faiss_add_batch_size) if expected_rows else 0
        for batch in tqdm(
            embeddings.iter(batch_size=self.config.faiss_add_batch_size),
            total=total_batches,
            desc="Building reference index",
        ):
            batch = cast("dict[str, list[Any]]", batch)
            batch_ids = cast("list[str]", batch["id"])
            vectors = np.asarray(batch["embedding"], dtype=np.float32)
            if vectors.size == 0:
                continue

            self._normalize_vectors(vectors)
            index.add(vectors)
            ids.extend(batch_ids)

        if len(ids) != expected_rows:
            msg = f"Indexed id count mismatch. expected={expected_rows}, got={len(ids)}"
            raise RuntimeError(msg)

        faiss.write_index(index, str(self.index_path))
        ids_table = pa.Table.from_pydict({"id": ids}, schema=pa.schema([pa.field("id", pa.string())]))
        pq.write_table(
            ids_table,
            where=str(self.ids_path),
            compression="zstd",
            use_content_defined_chunking=True,
            write_page_index=True,
        )
        self.meta_path.write_text(
            json.dumps(
                {
                    "model": self.config.model,
                    "dimensions": self.config.dimensions,
                    "rows": expected_rows,
                    "faiss_nlist_config": self.config.faiss_nlist,
                    "faiss_nlist_effective": effective_nlist,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if hasattr(index, "nprobe"):
            index.nprobe = min(self.config.faiss_nprobe, effective_nlist)

        logger.info(
            "index.build.done",
            path=str(self.index_path),
            rows=expected_rows,
            nlist=effective_nlist,
        )
        return _IndexArtifacts(index=index, ids=ids)

    def _load_or_build_index(self, embeddings: Dataset) -> _IndexArtifacts:
        expected_rows = len(embeddings)
        cached = self._load_cached_index(expected_rows=expected_rows)
        if cached is not None:
            return cached
        return self._build_index(embeddings=embeddings, expected_rows=expected_rows)

    def _search_single(
        self,
        *,
        query_vector: np.ndarray,
        source_id: str,
        index: Any,
        ids: list[str],
    ) -> tuple[list[str], list[float]]:
        if int(index.ntotal) == 0:
            return [], []

        search_k = self.config.top_k + self.config.oversample
        if self.config.exclude_self:
            search_k += 1
        search_k = min(search_k, int(index.ntotal))

        scores_array, indices_array = index.search(query_vector[np.newaxis, :], search_k)
        scores_row = cast("np.ndarray", scores_array[0])
        indices_row = cast("np.ndarray", indices_array[0])

        reference_text_ids: list[str] = []
        reference_scores: list[float] = []

        for score, index_value in zip(scores_row.tolist(), indices_row.tolist(), strict=True):
            if index_value < 0:
                continue

            candidate_id = ids[index_value]
            if self.config.exclude_self and candidate_id == source_id:
                continue
            if score < self.config.min_similarity:
                continue

            reference_text_ids.append(candidate_id)
            reference_scores.append(float(score))
            if len(reference_text_ids) >= self.config.top_k:
                break

        return reference_text_ids, reference_scores

    async def _process_batch(
        self,
        batch: dict[str, list[Any]],
        semaphore: asyncio.Semaphore,
        index: Any,
        ids: list[str],
        jokes_by_id: dict[str, str],
    ) -> _ReferencesBatchOutputs:
        async with semaphore:
            row_ids = cast("list[str]", batch["id"])
            keyword_lists = cast("list[list[str]]", batch["keywords"])

            prompts = [self._render_prompt(keywords) for keywords in keyword_lists]
            references_by_row: list[list[str]] = [[] for _ in row_ids]
            scores_by_row: list[list[float]] = [[] for _ in row_ids]

            active_indices: list[int] = []
            active_prompts: list[str] = []
            for index_value, prompt in enumerate(prompts):
                if prompt:
                    active_indices.append(index_value)
                    active_prompts.append(prompt)

            if active_prompts:
                query_embeddings = await self._embed_query_batch(active_prompts)
                query_vectors = np.asarray(query_embeddings, dtype=np.float32)
                self._normalize_vectors(query_vectors)

                for local_index, query_vector in enumerate(query_vectors):
                    row_index = active_indices[local_index]
                    candidate_ids, reference_scores = self._search_single(
                        query_vector=query_vector,
                        source_id=row_ids[row_index],
                        index=index,
                        ids=ids,
                    )
                    references_by_row[row_index] = [jokes_by_id[candidate_id] for candidate_id in candidate_ids]
                    scores_by_row[row_index] = reference_scores

            return _ReferencesBatchOutputs(
                id=row_ids,
                prompt=prompts,
                references=references_by_row,
                scores=scores_by_row,
            )

    def _build_jokes_lookup(self, jokes: Dataset) -> dict[str, str]:
        jokes_by_id: dict[str, str] = {}
        for batch in jokes.iter(batch_size=self.config.faiss_add_batch_size):
            batch = cast("dict[str, list[Any]]", batch)
            ids = cast("list[str]", batch["id"])
            texts = cast("list[str]", batch["text"])
            jokes_by_id.update(dict(zip(ids, texts, strict=True)))
        return jokes_by_id

    async def run(
        self,
        keywords: Dataset,
        embeddings: Dataset,
        jokes: Dataset,
        resume: bool = False,
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.next_part_index = self._get_next_part_index()

        dataset = keywords
        if resume:
            seen_ids = self._get_seen_ids()
            dataset = dataset.filter(lambda item: item["id"] not in seen_ids)
        elif self.next_part_index == 0:
            for file in self.output_dir.iterdir():
                file.unlink()

        index_artifacts = self._load_or_build_index(embeddings)
        jokes_by_id = self._build_jokes_lookup(jokes)

        batched_dataset = dataset.batch(self.config.batch_size)
        total_batches = math.ceil(len(dataset) / self.config.batch_size) if len(dataset) else 0

        write_buffer: list[_ReferencesBatchOutputs] = []
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        pending_tasks: set[asyncio.Task[_ReferencesBatchOutputs]] = set()

        for batch in tqdm(batched_dataset, total=total_batches):
            batch = cast("dict[str, list[Any]]", batch)
            task = asyncio.create_task(
                self._process_batch(
                    batch=batch,
                    semaphore=semaphore,
                    index=index_artifacts.index,
                    ids=index_artifacts.ids,
                    jokes_by_id=jokes_by_id,
                )
            )
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

        self._flush_buffer(write_buffer)

        logger.info(
            "run.done",
            model=self.config.model,
            output_dir=str(self.output_dir),
            top_k=self.config.top_k,
        )
        return self.output_dir


async def main() -> None:
    jokes_path = DATA_DIR / config.jokes.data_filename
    if not jokes_path.exists():
        jokes_path = build_jokes_dataset()

    embeddings_dir = DATA_DIR / config.embeddings.hf_config_name
    if not embeddings_dir.exists():
        embeddings_dir = build_embeddings_dataset()

    keywords_dir = DATA_DIR / config.keywords.hf_config_name
    if not keywords_dir.exists():
        keywords_dir = build_keywords_dataset()

    jokes = load_dataset("parquet", data_files=str(jokes_path), split="train")
    embeddings = load_dataset("parquet", data_dir=str(embeddings_dir), split="train")
    keywords = load_dataset("parquet", data_dir=str(keywords_dir), split="train[:1000]")

    pipeline = ReferencesPipeline()
    output_dir = await pipeline.run(keywords=keywords, embeddings=embeddings, jokes=jokes, resume=True)
    print(
        {
            "jokes_path": str(jokes_path),
            "embeddings_dir": str(embeddings_dir),
            "keywords_dir": str(keywords_dir),
            "references_dir": str(output_dir),
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
