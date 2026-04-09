import asyncio
import pickle
from pathlib import Path
from typing import cast

import numpy as np
import pyarrow.parquet as pq

from datasets import Dataset
from src.config import ReferencesConfig
from src.pipelines import references as references_module
from src.pipelines.references import ReferencesPipeline


def _render_prompt(keywords: list[str]) -> str:
    return f"Write a joke using the following keyword(s): {', '.join(keywords)}"


class _MockEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _MockEmbeddingResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.data = [_MockEmbeddingItem(embedding) for embedding in embeddings]


class _MockEmbeddingsAPI:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.mapping = mapping
        self.calls = 0

    async def create(
        self,
        *,
        model: str,
        input: list[str],
        dimensions: int,
    ) -> _MockEmbeddingResponse:
        del model
        self.calls += 1
        embeddings: list[list[float]] = []
        for query in input:
            prompt = query.split("\nQuery: ", maxsplit=1)[1]
            embedding = self.mapping[prompt]
            assert len(embedding) == dimensions
            embeddings.append(embedding)
        return _MockEmbeddingResponse(embeddings)


class _MockAsyncClient:
    def __init__(self, mapping: dict[str, list[float]]) -> None:
        self.embeddings = _MockEmbeddingsAPI(mapping)


class _FakeIndexFlatIP:
    def __init__(self, dimensions: int) -> None:
        self.dimensions = dimensions


class _FakeIndexIVFFlat:
    def __init__(self, quantizer: _FakeIndexFlatIP, dimensions: int, nlist: int, metric: int) -> None:
        del quantizer
        del nlist
        del metric
        self.dimensions = dimensions
        self.nprobe = 1
        self._vectors = np.empty((0, dimensions), dtype=np.float32)
        self._ids = np.empty((0,), dtype=np.int64)

    @property
    def ntotal(self) -> int:
        return int(self._ids.shape[0])

    def train(self, vectors: np.ndarray) -> None:
        if vectors.shape[1] != self.dimensions:
            msg = "Training vectors dimension mismatch."
            raise ValueError(msg)

    def add_with_ids(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        if vectors.size == 0:
            return
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32, copy=False)])
        self._ids = np.concatenate([self._ids, ids.astype(np.int64, copy=False)])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0:
            scores = np.full((queries.shape[0], k), -np.inf, dtype=np.float32)
            indices = np.full((queries.shape[0], k), -1, dtype=np.int64)
            return scores, indices
        similarities = queries @ self._vectors.T
        order = np.argsort(-similarities, axis=1)
        scores = np.full((queries.shape[0], k), -np.inf, dtype=np.float32)
        labels = np.full((queries.shape[0], k), -1, dtype=np.int64)
        max_width = min(k, self.ntotal)
        scores[:, :max_width] = np.take_along_axis(similarities, order[:, :max_width], axis=1)
        labels[:, :max_width] = self._ids[order[:, :max_width]]
        return scores, labels


class _FakeFaiss:
    METRIC_INNER_PRODUCT = 0

    @staticmethod
    def IndexFlatIP(dimensions: int) -> _FakeIndexFlatIP:
        return _FakeIndexFlatIP(dimensions)

    @staticmethod
    def IndexIVFFlat(
        quantizer: _FakeIndexFlatIP,
        dimensions: int,
        nlist: int,
        metric: int,
    ) -> _FakeIndexIVFFlat:
        return _FakeIndexIVFFlat(quantizer, dimensions, nlist, metric)

    @staticmethod
    def write_index(index: _FakeIndexIVFFlat, path: str) -> None:
        payload = {
            "dimensions": index.dimensions,
            "nprobe": index.nprobe,
            "vectors": index._vectors,
            "ids": index._ids,
        }
        with Path(path).open("wb") as output:
            pickle.dump(payload, output)

    @staticmethod
    def read_index(path: str) -> _FakeIndexIVFFlat:
        with Path(path).open("rb") as input_file:
            payload = pickle.load(input_file)  # noqa: S301
        index = _FakeIndexIVFFlat(
            quantizer=_FakeIndexFlatIP(payload["dimensions"]),
            dimensions=payload["dimensions"],
            nlist=1,
            metric=_FakeFaiss.METRIC_INNER_PRODUCT,
        )
        index.nprobe = payload["nprobe"]
        index.add_with_ids(payload["vectors"], payload["ids"])
        return index


def _load_split_rows(output_dir: Path, split: str) -> list[dict[str, object]]:
    split_files = sorted((output_dir / split).glob("part-*.parquet"))
    if not split_files:
        return []
    table = pq.read_table(split_files)
    return table.to_pylist()


def _load_all_split_rows(output_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ("train", "validation", "test"):
        rows.extend(_load_split_rows(output_dir, split))
    return rows


def test_references_pipeline_deduplicates_keyword_groups_across_jokes(tmp_path: Path) -> None:
    references_module.faiss = _FakeFaiss

    embeddings = Dataset.from_dict(
        {
            "id": [0, 1, 2],
            "embedding": [[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]],
        }
    )
    keywords = Dataset.from_dict(
        {
            "id": [0, 1, 2],
            "keywords": [["cat", "bar"], ["dog", "park"], ["cat", "park"]],
        }
    )
    jokes = Dataset.from_dict(
        {
            "id": [0, 1, 2],
            "text": ["cat joke", "dog joke", "cat park joke"],
        }
    )

    client = _MockAsyncClient(
        {
            _render_prompt(["cat"]): [1.0, 0.0],
            _render_prompt(["bar"]): [1.0, 0.0],
            _render_prompt(["cat", "bar"]): [1.0, 0.0],
            _render_prompt(["dog"]): [0.0, 1.0],
            _render_prompt(["park"]): [0.8, 0.2],
            _render_prompt(["dog", "park"]): [0.0, 1.0],
            _render_prompt(["cat", "park"]): [0.8, 0.2],
        }
    )
    pipeline = ReferencesPipeline(
        pipeline_config=ReferencesConfig(
            model="mock-model",
            dimensions=2,
            top_k=1,
            input_batch_size=2,
            output_batch_size=2,
            shard_size=50,
            max_parallel_requests=2,
            timeout=10,
            max_retries=1,
            faiss_nlist=1,
            faiss_nprobe=1,
            faiss_train_size=3,
            faiss_batch_size=2,
            index_dirname=str(tmp_path / "index"),
            oversample=2,
            min_similarity=0.0,
            validation_fraction=0.2,
            test_fraction=0.2,
        ),
        output_dir=tmp_path / "references",
        client=client,
    )

    asyncio.run(pipeline.run(keywords=keywords, embeddings=embeddings, jokes=jokes, resume=False))
    rows = _load_all_split_rows(tmp_path / "references")
    by_keywords = {tuple(cast("list[str]", row["keywords"])): row for row in rows}

    assert len(rows) == 7
    assert tuple(["cat"]) in by_keywords
    assert tuple(["park"]) in by_keywords
    assert by_keywords[("cat",)]["references"] == ["cat joke"]
    assert by_keywords[("dog",)]["references"] == ["dog joke"]
    assert by_keywords[("park",)]["references"] == ["cat park joke"]


def test_references_pipeline_writes_train_validation_test_splits(tmp_path: Path) -> None:
    references_module.faiss = _FakeFaiss
    embeddings = Dataset.from_dict(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "embedding": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        }
    )
    keywords = Dataset.from_dict(
        {"id": [0, 1, 2, 3, 4, 5], "keywords": [["first"], ["second"], ["third"], ["fourth"], ["fifth"], ["sixth"]]}
    )
    jokes = Dataset.from_dict(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "text": ["first joke", "second joke", "third joke", "fourth joke", "fifth joke", "sixth joke"],
        }
    )
    client = _MockAsyncClient(
        {
            _render_prompt(["first"]): [1.0, 0.0],
            _render_prompt(["second"]): [0.0, 1.0],
            _render_prompt(["third"]): [1.0, 0.0],
            _render_prompt(["fourth"]): [0.0, 1.0],
            _render_prompt(["fifth"]): [1.0, 0.0],
            _render_prompt(["sixth"]): [0.0, 1.0],
        }
    )
    pipeline = ReferencesPipeline(
        pipeline_config=ReferencesConfig(
            model="mock-model",
            dimensions=2,
            top_k=1,
            input_batch_size=1,
            output_batch_size=1,
            shard_size=50,
            max_parallel_requests=1,
            timeout=10,
            max_retries=1,
            faiss_nlist=1,
            faiss_nprobe=1,
            faiss_train_size=2,
            faiss_batch_size=2,
            index_dirname=str(tmp_path / "index"),
            oversample=1,
            min_similarity=-1.0,
            validation_fraction=0.2,
            test_fraction=0.2,
            random_seed=7,
        ),
        output_dir=tmp_path / "references",
        client=client,
    )

    asyncio.run(pipeline.run(keywords=keywords, embeddings=embeddings, jokes=jokes, resume=False))
    train_rows = _load_split_rows(tmp_path / "references", "train")
    validation_rows = _load_split_rows(tmp_path / "references", "validation")
    test_rows = _load_split_rows(tmp_path / "references", "test")

    assert len(train_rows) > 0
    assert len(validation_rows) > 0
    assert len(test_rows) > 0
    assert len(train_rows) + len(validation_rows) + len(test_rows) == 6
    assert {int(row["id"]) for row in validation_rows}.isdisjoint({int(row["id"]) for row in test_rows})


def test_references_pipeline_resume_uses_existing_splits(tmp_path: Path) -> None:
    references_module.faiss = _FakeFaiss
    embeddings = Dataset.from_dict(
        {
            "id": [0, 1, 2, 3, 4],
            "embedding": [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        }
    )
    keywords = Dataset.from_dict(
        {"id": [0, 1, 2, 3, 4], "keywords": [["first"], ["second"], ["third"], ["fourth"], ["fifth"]]}
    )
    jokes = Dataset.from_dict(
        {"id": [0, 1, 2, 3, 4], "text": ["first joke", "second joke", "third joke", "fourth joke", "fifth joke"]}
    )
    client = _MockAsyncClient(
        {
            _render_prompt(["first"]): [1.0, 0.0],
            _render_prompt(["second"]): [0.0, 1.0],
            _render_prompt(["third"]): [1.0, 0.0],
            _render_prompt(["fourth"]): [0.0, 1.0],
            _render_prompt(["fifth"]): [1.0, 0.0],
        }
    )

    pipeline = ReferencesPipeline(
        pipeline_config=ReferencesConfig(
            model="mock-model",
            dimensions=2,
            top_k=1,
            input_batch_size=1,
            output_batch_size=1,
            shard_size=10,
            max_parallel_requests=1,
            timeout=10,
            max_retries=1,
            faiss_nlist=1,
            faiss_nprobe=1,
            faiss_train_size=1,
            faiss_batch_size=1,
            index_dirname=str(tmp_path / "index"),
            validation_fraction=0.2,
            test_fraction=0.2,
        ),
        output_dir=tmp_path / "references",
        client=client,
    )

    asyncio.run(pipeline.run(keywords=keywords, embeddings=embeddings, jokes=jokes, resume=False))
    first_calls = client.embeddings.calls
    asyncio.run(pipeline.run(keywords=keywords, embeddings=embeddings, jokes=jokes, resume=True))
    assert client.embeddings.calls == first_calls
