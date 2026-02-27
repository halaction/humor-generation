import asyncio
import re

import pyarrow.parquet as pq

from src.config import EmbeddingsConfig, KeywordsConfig
from src.paths import DATA_DIR
from src.pipelines.keywords import KeywordsPipeline

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


class _InMemoryDataset:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = rows
        self.column_names = list(rows[0].keys()) if rows else []

    def __iter__(self):
        return iter(self._rows)


class _MockEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _MockEmbeddingResponse:
    def __init__(self, embeddings: list[list[float]]) -> None:
        self.data = [_MockEmbeddingItem(embedding) for embedding in embeddings]


class _MockEmbeddingsAPI:
    def __init__(self, forbidden_texts: set[str]) -> None:
        self.batch_sizes: list[int] = []
        self.active_calls = 0
        self.max_active_calls = 0
        self.forbidden_texts = forbidden_texts

    async def create(
        self,
        *,
        model: str,
        input: list[str],
        dimensions: int,
    ) -> _MockEmbeddingResponse:
        del model
        if any(text in self.forbidden_texts for text in input):
            raise AssertionError("Keywords pipeline re-embedded full joke texts instead of using input embeddings.")

        self.batch_sizes.append(len(input))
        self.active_calls += 1
        self.max_active_calls = max(self.max_active_calls, self.active_calls)
        await asyncio.sleep(0.02)
        embeddings: list[list[float]] = []
        for text in input:
            tokens = TOKEN_PATTERN.findall(text.lower())
            token_count = float(len(tokens))
            character_count = float(len(text))
            unique_count = float(len(set(tokens)))
            vector = [token_count, character_count, unique_count]
            if dimensions > 3:
                vector.extend([0.0] * (dimensions - 3))
            embeddings.append(vector[:dimensions])
        self.active_calls -= 1
        return _MockEmbeddingResponse(embeddings)


class _MockAsyncClient:
    def __init__(self, forbidden_texts: set[str]) -> None:
        self.embeddings = _MockEmbeddingsAPI(forbidden_texts=forbidden_texts)


def _build_embedding_vector(text: str, dimensions: int) -> list[float]:
    tokens = TOKEN_PATTERN.findall(text.lower())
    token_count = float(len(tokens))
    character_count = float(len(text))
    unique_count = float(len(set(tokens)))
    vector = [token_count, character_count, unique_count]
    if dimensions > 3:
        vector.extend([0.0] * (dimensions - 3))
    return vector[:dimensions]


def test_keywords_pipeline() -> None:
    keywords_config = KeywordsConfig(
        data_filename="keywords-test.parquet",
        ngram_min=1,
        ngram_max=3,
        top_n=3,
        stopwords=True,
        max_candidates=9,
        batch_size=2,
        max_parallel_requests=2,
        timeout=10,
        max_retries=1,
    )

    embeddings_config = EmbeddingsConfig(
        model="mock-embedding-model",
        dimensions=3,
        batch_size=2,
        shard_size=4,
        max_parallel_requests=2,
        timeout=10,
        max_retries=1,
    )

    output_path = DATA_DIR / keywords_config.data_filename
    if output_path.exists():
        output_path.unlink()

    rows = [
        {"id": "0", "text": "A cat walks into a bar and orders milk"},
        {"id": "1", "text": "A dog walks into a bar and orders water"},
        {"id": "2", "text": "A bird flies into a bar and orders seeds"},
        {"id": "3", "text": "A cow runs into a bar and orders grass"},
    ]
    rows_with_embeddings = [
        {
            "id": row["id"],
            "text": row["text"],
            "embedding": _build_embedding_vector(row["text"], dimensions=embeddings_config.dimensions),
        }
        for row in rows
    ]

    mock_client = _MockAsyncClient(forbidden_texts={row["text"] for row in rows})
    pipeline = KeywordsPipeline(
        keywords_config=keywords_config,
        embeddings_config=embeddings_config,
        client=mock_client,
    )
    dataset = _InMemoryDataset(rows_with_embeddings)

    try:
        first_results, first_output_path = asyncio.run(pipeline.run(dataset))
        second_results, second_output_path = asyncio.run(pipeline.run(dataset))

        assert first_output_path == second_output_path
        assert first_output_path.exists()
        assert len(first_results) == 4
        assert len(second_results) == 0

        table = pq.read_table(first_output_path)
        assert table.num_rows == 4
        assert set(table.column("joke_id").to_pylist()) == {"0", "1", "2", "3"}
        for keywords, scores in zip(
            table.column("keywords").to_pylist(),
            table.column("scores").to_pylist(),
            strict=True,
        ):
            assert isinstance(keywords, list)
            assert len(keywords) == 3
            for keyword in keywords:
                assert isinstance(keyword, str)
                assert keyword
            assert isinstance(scores, list)
            assert len(scores) == 3

        assert mock_client.embeddings.max_active_calls >= 2
        assert max(mock_client.embeddings.batch_sizes) <= 2
        assert len(mock_client.embeddings.batch_sizes) >= 4
    finally:
        if output_path.exists():
            output_path.unlink()
