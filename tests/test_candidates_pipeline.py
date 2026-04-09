import asyncio
from pathlib import Path

import pyarrow.parquet as pq
from datasets import Dataset

from src.config import CandidatesConfig
from src.pipelines.candidates import CandidatesPipeline


class _MockMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _MockChoice:
    def __init__(self, content: str) -> None:
        self.message = _MockMessage(content=content)


class _MockCompletion:
    def __init__(self, content: str) -> None:
        self.choices = [_MockChoice(content=content)]


class _MockChatCompletions:
    def __init__(self) -> None:
        self.calls = 0
        self.prompts: list[str] = []

    async def create(
        self,
        *,
        model: str,
        temperature: float,
        max_completion_tokens: int,
        messages: list[dict[str, str]],
    ) -> _MockCompletion:
        del model
        del temperature
        del max_completion_tokens
        self.calls += 1
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        return _MockCompletion(content=f"generated: {prompt}")


class _MockChat:
    def __init__(self) -> None:
        self.completions = _MockChatCompletions()


class _MockClient:
    def __init__(self) -> None:
        self.chat = _MockChat()


def _load_rows(output_dir: Path) -> list[dict[str, object]]:
    table = pq.read_table(sorted(output_dir.glob("part-*.parquet")))
    return table.to_pylist()


def test_candidates_pipeline_generates_from_keywords_only(tmp_path: Path) -> None:
    client = _MockClient()
    pipeline = CandidatesPipeline(
        pipeline_config=CandidatesConfig(
            model="mock-model",
            shard_size=10,
            max_parallel_requests=2,
            timeout=10,
            max_retries=1,
            temperature=1.0,
            max_completion_tokens=64,
        ),
        output_dir=tmp_path / "candidates",
        client=client,
    )

    references = Dataset.from_dict(
        {
            "id": [1, 2],
            "keywords": [["cat"], ["dog", "park"]],
            "references": [["r1"], ["r2"]],
            "scores": [[0.9], [0.8]],
        }
    )
    asyncio.run(
        pipeline.run(
            references=references,
            model="mock-model",
            model_id="base-v1",
            resume=False,
        )
    )

    rows = _load_rows(tmp_path / "candidates")
    assert len(rows) == 2
    assert sorted(str(row["model_id"]) for row in rows) == ["base-v1", "base-v1"]
    assert sorted(str(row["model"]) for row in rows) == ["mock-model", "mock-model"]
    assert all("Write a joke using the following keyword(s):" in prompt for prompt in client.chat.completions.prompts)
    assert client.chat.completions.calls == 2


def test_candidates_pipeline_resume_skips_seen_ids(tmp_path: Path) -> None:
    client = _MockClient()
    pipeline = CandidatesPipeline(
        pipeline_config=CandidatesConfig(
            model="mock-model",
            shard_size=1,
            max_parallel_requests=1,
            timeout=10,
            max_retries=1,
        ),
        output_dir=tmp_path / "candidates",
        client=client,
    )
    references = Dataset.from_dict(
        {
            "id": [1, 2],
            "keywords": [["cat"], ["dog"]],
            "references": [["r1"], ["r2"]],
            "scores": [[0.9], [0.8]],
        }
    )
    asyncio.run(
        pipeline.run(
            references=references,
            model="mock-model",
            model_id="base-v1",
            resume=True,
        )
    )
    first_calls = client.chat.completions.calls
    asyncio.run(
        pipeline.run(
            references=references,
            model="mock-model",
            model_id="base-v1",
            resume=True,
        )
    )
    assert client.chat.completions.calls == first_calls
