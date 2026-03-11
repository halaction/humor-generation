import asyncio
from pathlib import Path

import pyarrow.parquet as pq

from datasets import Dataset
from src.config import EmbeddingsConfig
from src.pipelines.embeddings import EmbeddingsPipeline


class _MockEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _MockEmbeddingResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.data = [_MockEmbeddingItem(embedding) for embedding in embeddings]


class _MockEmbeddingsAPI:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    async def create(
        self,
        *,
        model: str,
        input: list[str],
        dimensions: int,
    ) -> _MockEmbeddingResponse:
        del model
        self.batch_sizes.append(len(input))
        rows: list[list[float]] = []
        for text in input:
            base = float(len(text))
            rows.append([base + float(i) for i in range(dimensions)])
        return _MockEmbeddingResponse(rows)


class _MockAsyncClient:
    def __init__(self) -> None:
        self.embeddings = _MockEmbeddingsAPI()


def _load_rows(output_dir: Path) -> list[dict[str, object]]:
    table = pq.read_table(sorted(output_dir.glob("part-*.parquet")))
    return table.to_pylist()


def test_embeddings_pipeline_skips_empty_text_rows(tmp_path: Path) -> None:
    pipeline = EmbeddingsPipeline(
        pipeline_config=EmbeddingsConfig(
            model="mock-model",
            dimensions=4,
            batch_size=2,
            shard_size=2,
            max_parallel_requests=2,
            timeout=10,
            max_retries=1,
        ),
        output_dir=tmp_path / "embeddings",
        client=_MockAsyncClient(),
    )

    jokes = Dataset.from_dict(
        {
            "id": [0, 1, 2, 3],
            "text": ["joke one", " ", "joke two", ""],
        }
    )

    asyncio.run(pipeline.run(jokes, resume=False))
    rows = _load_rows(tmp_path / "embeddings")

    assert {row["id"] for row in rows} == {0, 2}
    assert all(len(row["embedding"]) == 4 for row in rows)
    assert pipeline.client.embeddings.batch_sizes == [1, 1]
