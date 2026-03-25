import asyncio
from pathlib import Path

import pyarrow.parquet as pq

from datasets import Dataset
from src.config import KeywordsConfig
from src.pipelines.keywords import KeywordsPipeline


class _MockEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _MockEmbeddingResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.data = [_MockEmbeddingItem(embedding) for embedding in embeddings]


class _MockEmbeddingsAPI:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []
        self.queries: list[str] = []

    async def create(
        self,
        *,
        model: str,
        input: list[str],
        dimensions: int,
    ) -> _MockEmbeddingResponse:
        del model
        self.batch_sizes.append(len(input))
        self.queries.extend(input)
        rows: list[list[float]] = []
        for query in input:
            keyword = query.split("\nQuery: ", maxsplit=1)[1]
            if "cat" in keyword:
                rows.append([1.0, 0.0, 0.0][:dimensions])
            elif "dog" in keyword:
                rows.append([0.0, 1.0, 0.0][:dimensions])
            else:
                rows.append([0.1, 0.1, 0.1][:dimensions])
        return _MockEmbeddingResponse(rows)


class _MockAsyncClient:
    def __init__(self) -> None:
        self.embeddings = _MockEmbeddingsAPI()


def _load_rows(output_dir: Path) -> list[dict[str, object]]:
    table = pq.read_table(sorted(output_dir.glob("part-*.parquet")))
    return table.to_pylist()


def test_keywords_pipeline_uses_template_instruction_and_skips_empty_rows(tmp_path: Path) -> None:
    client = _MockAsyncClient()
    pipeline = KeywordsPipeline(
        pipeline_config=KeywordsConfig(
            model="mock-model",
            dimensions=3,
            ngram_min=1,
            ngram_max=1,
            top_n=2,
            stopwords=False,
            max_candidates=8,
            batch_size=2,
            shard_size=2,
            max_parallel_requests=1,
            timeout=10,
            max_retries=1,
        ),
        output_dir=tmp_path / "keywords",
        client=client,
    )

    jokes = Dataset.from_dict({"id": [0, 1], "text": ["cat cat joke", " "]})
    embeddings = Dataset.from_dict({"id": [0, 1], "embedding": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]})

    asyncio.run(pipeline.run(jokes=jokes, embeddings=embeddings, resume=False))
    rows = _load_rows(tmp_path / "keywords")

    assert [row["id"] for row in rows] == [0]
    assert rows[0]["keywords"]
    assert all(
        query.startswith("Instruct: Given a short keyword or phrase, retrieve jokes")
        for query in client.embeddings.queries
    )
    assert max(client.embeddings.batch_sizes) <= 2
