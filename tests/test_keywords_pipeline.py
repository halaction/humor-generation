import asyncio
import json
import re
from pathlib import Path

from datasets import Dataset
from src.config import EmbeddingsConfig, KeywordsConfig
from src.pipelines.keywords import KeywordsPipeline

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


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
    def __init__(self) -> None:
        self.embeddings = _MockEmbeddingsAPI()


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def test_keywords_pipeline() -> None:
    results_filename = "keywords-test.jsonl"
    keywords_config = KeywordsConfig(
        results_filename=results_filename,
        ngram_min=1,
        ngram_max=3,
        top_n=3,
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

    mock_client = _MockAsyncClient()
    pipeline = KeywordsPipeline(
        keywords_config=keywords_config,
        embeddings_config=embeddings_config,
        client=mock_client,
    )

    dataset = Dataset.from_list(
        [
            {"id": "0", "text": "A cat walks into a bar and orders milk"},
            {"id": "1", "text": "A dog walks into a bar and orders water"},
            {"id": "2", "text": "A bird flies into a bar and orders seeds"},
            {"id": "3", "text": "A cow runs into a bar and orders grass"},
        ]
    )

    first_results, output_path = asyncio.run(pipeline.run(dataset))
    second_results, second_output_path = asyncio.run(pipeline.run(dataset))

    assert output_path == second_output_path
    assert output_path.exists()
    assert len(first_results) == 4
    assert len(second_results) == 0

    rows = _read_jsonl(output_path)
    assert len(rows) == 4
    assert {row["joke_id"] for row in rows} == {"0", "1", "2", "3"}
    for row in rows:
        keywords = row["keywords"]
        assert isinstance(keywords, list)
        assert len(keywords) == 3
        for keyword in keywords:
            assert isinstance(keyword, str)
            assert keyword

    assert mock_client.embeddings.max_active_calls >= 2
    assert max(mock_client.embeddings.batch_sizes) <= 2
    assert len(mock_client.embeddings.batch_sizes) >= 4
