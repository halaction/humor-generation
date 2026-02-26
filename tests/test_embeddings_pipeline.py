import asyncio

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
        self.active_calls = 0
        self.max_active_calls = 0

    async def create(
        self,
        *,
        model: str,
        input: list[str],
        dimensions: int,
    ) -> _MockEmbeddingResponse:
        del model
        self.batch_sizes.append(len(input))
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        await asyncio.sleep(0.02)
        embeddings: list[list[float]] = []
        for text in input:
            base = float(len(text))
            vector = [base + float(index) for index in range(dimensions)]
            embeddings.append(vector)
        self.active_calls -= 1
        return _MockEmbeddingResponse(embeddings)


class _MockAsyncClient:
    def __init__(self) -> None:
        self.embeddings = _MockEmbeddingsAPI()


def test_embeddings_pipeline() -> None:
    data_filename = "embeddings-test.parquet"
    embeddings_config = EmbeddingsConfig(
        data_filename=data_filename,
        model="mock-embedding-model",
        dimensions=4,
        batch_size=2,
        shard_size=3,
        max_parallel_requests=2,
        timeout=10,
        max_retries=1,
    )
    mock_client = _MockAsyncClient()
    pipeline = EmbeddingsPipeline(
        pipeline_config=embeddings_config,
        client=mock_client,
    )

    dataset = Dataset.from_list(
        [
            {"id": "0", "text": "joke about cats"},
            {"id": "1", "text": "joke about dogs"},
            {"id": "2", "text": "joke about birds"},
            {"id": "3", "text": "joke about fish"},
            {"id": "4", "text": "joke about snakes"},
            {"id": "5", "text": "joke about cows"},
        ]
    )

    outputs = asyncio.run(pipeline.run(dataset))

    assert outputs.data_path.exists()
    table = pq.read_table(outputs.data_path)
    assert table.num_rows == 6
    assert set(table.column("id").to_pylist()) == {"0", "1", "2", "3", "4", "5"}
    for embedding in table.column("embedding").to_pylist():
        assert len(embedding) == 4

    assert mock_client.embeddings.batch_sizes == [2, 2, 2]
    assert mock_client.embeddings.max_active_calls >= 2
