import asyncio
import pickle
from pathlib import Path

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
        self.batch_sizes: list[int] = []

    async def create(
        self,
        *,
        model: str,
        input: list[str],
        dimensions: int,
    ) -> _MockEmbeddingResponse:
        del model
        self.calls += 1
        self.batch_sizes.append(len(input))
        embeddings: list[list[float]] = []
        for query in input:
            assert query.startswith("Instruct: ")
            assert "\nQuery: " in query
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

    @property
    def ntotal(self) -> int:
        return int(self._vectors.shape[0])

    def train(self, vectors: np.ndarray) -> None:
        if vectors.shape[1] != self.dimensions:
            msg = "Training vectors dimension mismatch."
            raise ValueError(msg)

    def add(self, vectors: np.ndarray) -> None:
        if vectors.size == 0:
            return
        self._vectors = np.vstack([self._vectors, vectors.astype(np.float32, copy=False)])

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        if self.ntotal == 0:
            scores = np.full((queries.shape[0], k), -np.inf, dtype=np.float32)
            indices = np.full((queries.shape[0], k), -1, dtype=np.int64)
            return scores, indices

        similarities = queries @ self._vectors.T
        order = np.argsort(-similarities, axis=1)
        scores = np.full((queries.shape[0], k), -np.inf, dtype=np.float32)
        indices = np.full((queries.shape[0], k), -1, dtype=np.int64)

        max_width = min(k, self.ntotal)
        scores[:, :max_width] = np.take_along_axis(similarities, order[:, :max_width], axis=1)
        indices[:, :max_width] = order[:, :max_width].astype(np.int64)
        return scores, indices


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
        payload = {"dimensions": index.dimensions, "nprobe": index.nprobe, "vectors": index._vectors}
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
        index.add(payload["vectors"])
        return index


def _load_output_rows(output_dir: Path) -> list[dict[str, object]]:
    output_files = sorted(output_dir.glob("part-*.parquet"))
    assert output_files
    table = pq.read_table(output_files)
    return table.to_pylist()


def test_references_pipeline_retrieves_neighbors_and_excludes_self(tmp_path: Path) -> None:
    references_module.faiss = _FakeFaiss

    embeddings = Dataset.from_dict(
        {
            "id": ["0", "1", "2"],
            "embedding": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.8, 0.2, 0.0],
            ],
        }
    )
    keywords = Dataset.from_dict(
        {
            "id": ["0", "1", "2"],
            "keywords": [["cat", "bar"], ["dog", "park"], ["cat", "park"]],
        }
    )
    jokes = Dataset.from_dict(
        {
            "id": ["0", "1", "2"],
            "text": ["cat joke", "dog joke", "cat park joke"],
        }
    )

    query_mapping = {
        _render_prompt(["cat"]): [1.0, 0.0, 0.0],
        _render_prompt(["bar"]): [1.0, 0.0, 0.0],
        _render_prompt(["cat", "bar"]): [1.0, 0.0, 0.0],
        _render_prompt(["dog"]): [0.0, 1.0, 0.0],
        _render_prompt(["park"]): [0.8, 0.2, 0.0],
        _render_prompt(["dog", "park"]): [0.0, 1.0, 0.0],
        _render_prompt(["cat", "park"]): [0.8, 0.2, 0.0],
    }
    client = _MockAsyncClient(query_mapping)
    pipeline_config = ReferencesConfig(
        model="mock-model",
        dimensions=3,
        top_k=1,
        input_batch_size=2,
        output_batch_size=2,
        shard_size=2,
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
    )
    output_dir = tmp_path / "references"
    pipeline = ReferencesPipeline(
        pipeline_config=pipeline_config,
        output_dir=output_dir,
        client=client,
    )

    asyncio.run(pipeline.run(keywords=keywords, embeddings=embeddings, jokes=jokes, resume=False))

    rows = _load_output_rows(output_dir=output_dir)
    by_id_and_prompt = {(row["id"], row["prompt"]): row for row in rows}

    assert len(rows) == 9
    assert by_id_and_prompt[("0", _render_prompt(["cat"]))]["references"] == ["cat joke", "cat park joke"]
    assert by_id_and_prompt[("0", _render_prompt(["bar"]))]["references"] == ["cat joke", "cat park joke"]
    assert by_id_and_prompt[("0", _render_prompt(["cat", "bar"]))]["references"] == ["cat joke", "cat park joke"]
    assert by_id_and_prompt[("1", _render_prompt(["dog"]))]["references"] == ["dog joke", "cat park joke"]
    assert by_id_and_prompt[("1", _render_prompt(["park"]))]["references"] == ["dog joke", "cat park joke"]
    assert by_id_and_prompt[("1", _render_prompt(["dog", "park"]))]["references"] == ["dog joke", "cat park joke"]
    assert by_id_and_prompt[("2", _render_prompt(["cat"]))]["references"] == ["cat park joke", "cat joke"]
    assert by_id_and_prompt[("2", _render_prompt(["park"]))]["references"] == ["cat park joke", "cat joke"]
    assert by_id_and_prompt[("2", _render_prompt(["cat", "park"]))]["references"] == ["cat park joke", "cat joke"]

    for row in rows:
        assert len(row["scores"]) == 2
        assert row["scores"][0] == 1.0
        assert row["scores"][1] > 0.0
    assert client.embeddings.calls == 5
    assert max(client.embeddings.batch_sizes) <= 2


def test_references_pipeline_resume_skips_seen_ids(tmp_path: Path) -> None:
    references_module.faiss = _FakeFaiss

    embeddings = Dataset.from_dict(
        {
            "id": ["0", "1"],
            "embedding": [
                [1.0, 0.0],
                [0.0, 1.0],
            ],
        }
    )
    keywords = Dataset.from_dict(
        {
            "id": ["0", "1"],
            "keywords": [["first"], ["second"]],
        }
    )
    jokes = Dataset.from_dict(
        {
            "id": ["0", "1"],
            "text": ["first joke", "second joke"],
        }
    )

    query_mapping = {
        _render_prompt(["first"]): [1.0, 0.0],
        _render_prompt(["second"]): [0.0, 1.0],
    }
    client = _MockAsyncClient(query_mapping)
    pipeline_config = ReferencesConfig(
        model="mock-model",
        dimensions=2,
        top_k=1,
        input_batch_size=1,
        output_batch_size=1,
        shard_size=1,
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
    )
    output_dir = tmp_path / "references"
    pipeline = ReferencesPipeline(
        pipeline_config=pipeline_config,
        output_dir=output_dir,
        client=client,
    )

    asyncio.run(pipeline.run(keywords=keywords, embeddings=embeddings, jokes=jokes, resume=True))
    first_call_count = client.embeddings.calls
    asyncio.run(pipeline.run(keywords=keywords, embeddings=embeddings, jokes=jokes, resume=True))

    rows = _load_output_rows(output_dir=output_dir)

    assert len(rows) == 2
    assert rows[0]["prompt"] == _render_prompt(["first"])
    assert rows[1]["prompt"] == _render_prompt(["second"])
    assert rows[0]["scores"][0] == 1.0
    assert rows[1]["scores"][0] == 1.0
    assert first_call_count == 2
    assert client.embeddings.calls == first_call_count
