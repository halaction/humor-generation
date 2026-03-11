import asyncio
import json
import math
from collections import defaultdict
from itertools import batched, combinations
from pathlib import Path
from typing import Any, cast

import faiss
import numpy as np
import numpy.typing as npt
import polars as pl
import pyarrow as pa
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from datasets import Dataset, load_dataset
from src.config import ReferencesConfig, config
from src.datasets.embeddings import build_embeddings_dataset
from src.datasets.jokes import build_jokes_dataset
from src.datasets.keywords import build_keywords_dataset
from src.logging import get_logger
from src.models import ReferencesInputs, ReferencesOutputs
from src.paths import DATA_DIR
from src.pipelines.base import BasePipeline
from src.settings import settings
from src.templates import environment

logger = get_logger(__name__)

_INDEX_ID_STORAGE = "faiss_add_with_ids_int64_v1"


class ReferencesPipeline(BasePipeline):
    def __init__(
        self,
        pipeline_config: ReferencesConfig | None = None,
        output_dir: Path | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.config = pipeline_config or config.references
        if self.config.min_keywords > self.config.max_keywords:
            msg = "`min_keywords` must be less than or equal to `max_keywords`."
            raise ValueError(msg)
        self.output_dir = output_dir or DATA_DIR / self.config.hf_config_name
        self.client = client or AsyncOpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=self.config.timeout,
        )
        self.next_part_index = 0

        self.index_dir = DATA_DIR / "index"
        self.index_path = self.index_dir / "index.faiss"
        self.meta_path = self.index_dir / "meta.json"

        self.prompt_template = environment.get_template("reference_prompt.j2")
        self.query_template = environment.get_template("reference_query.j2")

        self.schema = pa.schema(
            [
                pa.field("id", pa.int64()),
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

    def _get_table(self, write_buffer: list[ReferencesOutputs]) -> pa.Table:
        outputs = defaultdict(list)
        for batch in write_buffer:
            for key, value in batch.model_dump().items():
                outputs[key].extend(value)
        return pa.Table.from_pydict(outputs, schema=self.schema)

    def _check_buffer_size(self, write_buffer: list[ReferencesOutputs]) -> bool:
        return sum(len(batch.id) for batch in write_buffer) >= self.config.shard_size

    def _build_keyword_groups(self, keywords: list[str]) -> list[list[str]]:
        cleaned_keywords = []
        for keyword in keywords:
            cleaned_keyword = keyword.strip()
            if cleaned_keyword:
                cleaned_keywords.append(cleaned_keyword)

        cleaned_keywords = list(dict.fromkeys(cleaned_keywords))
        if not cleaned_keywords:
            return []

        max_keywords = min(self.config.max_keywords, len(cleaned_keywords))
        min_keywords = min(self.config.min_keywords, max_keywords)

        groups: list[list[str]] = []
        for keyword_count in range(min_keywords, max_keywords + 1):
            for group in combinations(cleaned_keywords, keyword_count):
                groups.append(list(group))

        return groups

    async def _embed_batch(self, prompts: list[str]) -> npt.NDArray[np.float32]:
        if not prompts:
            return np.empty((0, self.config.dimensions), dtype=np.float32)

        embeddings: list[list[float]] = []
        for prompt_batch in batched(prompts, self.config.output_batch_size, strict=False):
            formatted_queries = [self.query_template.render(prompt=prompt).strip() for prompt in prompt_batch]
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    response = await self.client.embeddings.create(
                        model=self.config.model,
                        input=formatted_queries,
                        dimensions=self.config.dimensions,
                    )
                    embeddings.extend([item.embedding for item in response.data])
                except Exception:
                    if attempt >= self.config.max_retries:
                        raise
                    await asyncio.sleep(2 ** (attempt - 1))
                else:
                    break

        return np.asarray(embeddings, dtype=np.float32)

    def _sample_training_vectors(self, embeddings: Dataset, sample_size: int) -> np.ndarray:
        reservoir = np.empty((sample_size, self.config.dimensions), dtype=np.float32)
        random_generator = np.random.default_rng(42)

        seen = 0
        for batch in embeddings.iter(batch_size=self.config.faiss_batch_size):
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

    def _load_faiss_index(self, expected_rows: int) -> faiss.IndexIVFFlat | None:
        if not (self.index_path.exists() and self.meta_path.exists()):
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
        if metadata.get("id_storage") != _INDEX_ID_STORAGE:
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
        return index

    def _build_faiss_index(self, embeddings: Dataset) -> faiss.IndexIVFFlat:
        expected_rows = len(embeddings)
        cached_index = self._load_faiss_index(expected_rows=expected_rows)
        if cached_index is not None:
            return cached_index

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

        indexed_count = 0
        total_batches = math.ceil(expected_rows / self.config.faiss_batch_size) if expected_rows else 0
        for batch in tqdm(
            embeddings.iter(batch_size=self.config.faiss_batch_size),
            total=total_batches,
            desc="Building reference index",
        ):
            batch = cast("dict[str, list[Any]]", batch)
            batch_ids = np.asarray(batch["id"], dtype=np.int64)
            vectors = np.asarray(batch["embedding"], dtype=np.float32)
            if vectors.size == 0:
                continue
            if vectors.shape[0] != batch_ids.shape[0]:
                msg = (
                    "Embedding/id batch length mismatch while indexing. "
                    f"ids={batch_ids.shape[0]}, vectors={vectors.shape[0]}"
                )
                raise RuntimeError(msg)

            self._normalize_vectors(vectors)
            index.add_with_ids(vectors, batch_ids)
            indexed_count += int(batch_ids.shape[0])

        if indexed_count != expected_rows:
            msg = f"Indexed row count mismatch. expected={expected_rows}, got={indexed_count}"
            raise RuntimeError(msg)

        faiss.write_index(index, str(self.index_path))
        self.meta_path.write_text(
            json.dumps(
                {
                    "model": self.config.model,
                    "dimensions": self.config.dimensions,
                    "rows": expected_rows,
                    "faiss_nlist_config": self.config.faiss_nlist,
                    "faiss_nlist_effective": effective_nlist,
                    "id_storage": _INDEX_ID_STORAGE,
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
        return index

    def _search_batch(
        self,
        query_vectors: npt.NDArray[np.float32],
        source_ids: npt.NDArray[np.int64],
        faiss_index: faiss.IndexIVFFlat,
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
        search_k = self.config.top_k + self.config.oversample
        if self.config.exclude_self:
            search_k += 1
        search_k = min(search_k, int(faiss_index.ntotal))

        candidate_scores, candidate_ids = faiss_index.search(query_vectors, search_k)  # type: ignore
        candidate_scores = np.asarray(candidate_scores, dtype=np.float32)
        candidate_ids = np.asarray(candidate_ids, dtype=np.int64)

        keep_mask = candidate_ids >= 0
        keep_mask &= candidate_scores >= self.config.min_similarity
        if self.config.exclude_self:
            keep_mask &= candidate_ids != np.expand_dims(source_ids, axis=1)

        rank_mask = np.cumsum(keep_mask, axis=1, dtype=np.int32) <= self.config.top_k
        keep_mask &= rank_mask

        return candidate_ids, candidate_scores, keep_mask

    def _build_jokes_lookup(self, jokes: Dataset) -> dict[int, str]:
        jokes_mapping: dict[int, str] = {}
        for batch in tqdm(jokes.batch(self.config.faiss_batch_size)):
            batch = cast("dict[str, list[Any]]", batch)
            ids = cast("list[int]", batch["id"])
            texts = cast("list[str]", batch["text"])
            jokes_mapping.update(dict(zip(ids, texts, strict=True)))
        return jokes_mapping

    async def _retrieve_references(
        self,
        inputs: ReferencesInputs,
        semaphore: asyncio.Semaphore,
        faiss_index: faiss.IndexIVFFlat,
        jokes_mapping: dict[int, str],
    ) -> ReferencesOutputs | None:
        async with semaphore:
            expanded_ids: list[int] = []
            expanded_prompts: list[str] = []

            logger.debug("retrieve.expand")
            for row_id, keywords in zip(inputs.id, inputs.keywords, strict=True):
                keyword_groups = self._build_keyword_groups(keywords)
                prompts = [self.prompt_template.render(keywords=group).strip() for group in keyword_groups]
                for prompt in prompts:
                    expanded_ids.append(row_id)
                    expanded_prompts.append(prompt)

            if not expanded_prompts:
                return None

            logger.debug("retrieve.embed")
            query_vectors = await self._embed_batch(expanded_prompts)
            self._normalize_vectors(query_vectors)

            logger.debug("retrieve.search")
            source_ids = np.asarray(expanded_ids, dtype=np.int64)
            candidate_ids_batch, candidate_scores_batch, candidate_mask_batch = self._search_batch(
                query_vectors=query_vectors,
                source_ids=source_ids,
                faiss_index=faiss_index,
            )

            output_ids: list[int] = []
            output_prompts: list[str] = []
            output_references: list[list[str]] = []
            output_scores: list[list[float]] = []

            logger.debug("retrieve.zip")
            for source_id, prompt, candidate_ids, candidate_scores, candidate_mask in zip(
                expanded_ids,
                expanded_prompts,
                candidate_ids_batch,
                candidate_scores_batch,
                candidate_mask_batch,
                strict=True,
            ):
                if source_id not in jokes_mapping:
                    continue
                references = [jokes_mapping[source_id]]
                scores = [1.0]
                masked_ids = cast("list[int]", candidate_ids[candidate_mask].tolist())
                masked_scores = cast("list[float]", candidate_scores[candidate_mask].tolist())

                # TODO: Can this be sped up? Think about join with jokes as a frame.
                for candidate_id, candidate_score in zip(masked_ids, masked_scores, strict=True):
                    if candidate_id not in jokes_mapping:
                        continue
                    references.append(jokes_mapping[candidate_id])
                    scores.append(candidate_score)

                output_ids.append(source_id)
                output_prompts.append(prompt)
                output_references.append(references)
                output_scores.append(scores)

            return ReferencesOutputs(
                id=output_ids,
                prompt=output_prompts,
                references=output_references,
                scores=output_scores,
            )

    async def run(
        self,
        keywords: Dataset,
        embeddings: Dataset,
        jokes: Dataset,
        resume: bool = False,
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.next_part_index = self._get_next_part_index()

        keywords_frame = cast("pl.DataFrame", keywords.to_polars())
        jokes_frame = cast("pl.DataFrame", jokes.to_polars())
        joined_frame = jokes_frame.join(keywords_frame, on="id", how="inner").select(["id", "text", "keywords"])
        dataset = Dataset.from_polars(joined_frame)

        if resume:
            seen_ids = self._get_seen_ids()
            dataset = dataset.filter(lambda item: item["id"] not in seen_ids)
        elif self.next_part_index > 0:
            for file in self.output_dir.glob("part-*.parquet"):
                file.unlink()

        dataset = dataset.batch(self.config.input_batch_size)

        faiss_index = self._build_faiss_index(embeddings)
        jokes_mapping = self._build_jokes_lookup(jokes)

        write_buffer: list[ReferencesOutputs] = []
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        pending_tasks: set[asyncio.Task[ReferencesOutputs | None]] = set()

        for batch in tqdm(dataset):
            batch = cast("dict[str, list[Any]]", batch)
            inputs = ReferencesInputs(
                id=cast("list[int]", batch["id"]),
                keywords=cast("list[list[str]]", batch["keywords"]),
            )

            task = asyncio.create_task(
                self._retrieve_references(
                    inputs=inputs,
                    semaphore=semaphore,
                    faiss_index=faiss_index,
                    jokes_mapping=jokes_mapping,
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
    output_dir = await pipeline.run(
        keywords=keywords,
        embeddings=embeddings,
        jokes=jokes,
        resume=False,
    )
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
