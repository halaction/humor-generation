import asyncio
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datasets import Dataset

from src.config import EvaluationConfig
from src.models import EvaluationJudgeDecision
from src.pipelines.evaluation import EvaluationPipeline


class _MockMessage:
    def __init__(self, parsed: EvaluationJudgeDecision | None) -> None:
        self.parsed = parsed


class _MockChoice:
    def __init__(self, parsed: EvaluationJudgeDecision | None) -> None:
        self.message = _MockMessage(parsed=parsed)


class _MockCompletion:
    def __init__(self, parsed: EvaluationJudgeDecision | None) -> None:
        self.choices = [_MockChoice(parsed=parsed)]


class _MockChatCompletions:
    def __init__(self, decisions: dict[tuple[str, str], str], invalid_before_success: int = 0) -> None:
        self.decisions = decisions
        self.invalid_before_success = invalid_before_success
        self.calls = 0

    async def parse(
        self,
        *,
        model: str,
        temperature: float,
        messages: list[dict[str, str]],
        response_format: type[EvaluationJudgeDecision],
    ) -> _MockCompletion:
        del model
        del temperature
        del response_format
        self.calls += 1
        if self.calls <= self.invalid_before_success:
            return _MockCompletion(parsed=None)

        user_content = messages[1]["content"]
        left = user_content.split("Left:\n", maxsplit=1)[1].split("\n\nRight:\n", maxsplit=1)[0]
        right = user_content.split("\n\nRight:\n", maxsplit=1)[1]
        winner = self.decisions[(left, right)]
        return _MockCompletion(parsed=EvaluationJudgeDecision(winner=winner))


class _MockChat:
    def __init__(self, decisions: dict[tuple[str, str], str], invalid_before_success: int = 0) -> None:
        self.completions = _MockChatCompletions(
            decisions=decisions,
            invalid_before_success=invalid_before_success,
        )


class _MockClient:
    def __init__(self, decisions: dict[tuple[str, str], str], invalid_before_success: int = 0) -> None:
        self.chat = _MockChat(decisions=decisions, invalid_before_success=invalid_before_success)


def _read_part_rows(directory: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("part-*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    return rows


def test_evaluation_keyword_schema_and_leaderboard(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "id": [1, 1, 1, 2, 2],
            "keywords": [
                ["prompt", "one"],
                ["prompt", "one"],
                ["prompt", "one"],
                ["prompt", "two"],
                ["prompt", "two"],
            ],
            "model": ["base-v1", "instruct-v1", "thinking-v1", "base-v1", "instruct-v1"],
            "text": ["base p1", "instruct p1", "thinking p1", "base p2", "instruct p2"],
        }
    )
    decisions = {
        ("base p1", "instruct p1"): "left",
        ("instruct p1", "base p1"): "right",
        ("base p1", "thinking p1"): "right",
        ("thinking p1", "base p1"): "left",
        ("instruct p1", "thinking p1"): "right",
        ("thinking p1", "instruct p1"): "left",
        ("base p2", "instruct p2"): "right",
        ("instruct p2", "base p2"): "left",
    }
    client = _MockClient(decisions=decisions)
    pipeline = EvaluationPipeline(
        pipeline_config=EvaluationConfig(
            model="mock-judge",
            max_parallel_requests=2,
            max_retries=1,
            random_seed=42,
            shard_size=2,
        ),
        output_dir=tmp_path / "bundle",
        client=client,
    )

    asyncio.run(pipeline.run(candidates=candidates, resume=False))

    evaluations = _read_part_rows(tmp_path / "bundle" / "evaluation")
    assert len(evaluations) == 4
    assert client.chat.completions.calls == 4
    assert sorted(evaluations[0].keys()) == [
        "id",
        "left_model",
        "left_text",
        "prompt",
        "reference_id",
        "right_model",
        "right_text",
        "winner",
    ]
    assert sorted(int(row["id"]) for row in evaluations) == [0, 1, 2, 3]

    leaderboard = _read_part_rows(tmp_path / "bundle" / "leaderboard")
    by_model = {str(row["model"]): row for row in leaderboard}
    assert by_model["thinking-v1"]["bt_score"] > by_model["base-v1"]["bt_score"]


def test_evaluation_requires_keywords(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "id": [1, 1],
            "model": ["base-v1", "instruct-v1"],
            "text": ["a", "b"],
        }
    )
    pipeline = EvaluationPipeline(
        pipeline_config=EvaluationConfig(model="mock-judge"),
        output_dir=tmp_path / "bundle",
        client=_MockClient(decisions={}),
    )
    with pytest.raises(ValueError):
        asyncio.run(pipeline.run(candidates=candidates, resume=False))


def test_evaluation_rejects_inconsistent_keywords_for_same_id(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "id": [1, 1],
            "keywords": [["cat"], ["dog"]],
            "model": ["base-v1", "instruct-v1"],
            "text": ["a", "b"],
        }
    )
    pipeline = EvaluationPipeline(
        pipeline_config=EvaluationConfig(model="mock-judge"),
        output_dir=tmp_path / "bundle",
        client=_MockClient(decisions={}),
    )
    with pytest.raises(ValueError):
        asyncio.run(pipeline.run(candidates=candidates, resume=False))


def test_evaluation_resume_keeps_existing_pairs(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "id": [1, 1],
            "keywords": [["prompt", "one"], ["prompt", "one"]],
            "model": ["base-v1", "instruct-v1"],
            "text": ["base p1", "instruct p1"],
        }
    )
    decisions = {
        ("base p1", "instruct p1"): "left",
        ("instruct p1", "base p1"): "right",
    }
    client = _MockClient(decisions=decisions)
    pipeline = EvaluationPipeline(
        pipeline_config=EvaluationConfig(model="mock-judge", random_seed=42, max_retries=1, shard_size=1),
        output_dir=tmp_path / "bundle",
        client=client,
    )

    asyncio.run(pipeline.run(candidates=candidates, resume=True))
    first_calls = client.chat.completions.calls
    asyncio.run(pipeline.run(candidates=candidates, resume=True))
    assert client.chat.completions.calls == first_calls


def test_evaluation_resume_rebuilds_when_rows_are_incomplete(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "id": [1, 1, 2, 2],
            "keywords": [["prompt", "one"], ["prompt", "one"], ["prompt", "two"], ["prompt", "two"]],
            "model": ["base-v1", "instruct-v1", "base-v1", "instruct-v1"],
            "text": ["base p1", "instruct p1", "base p2", "instruct p2"],
        }
    )
    decisions = {
        ("base p1", "instruct p1"): "left",
        ("instruct p1", "base p1"): "right",
        ("base p2", "instruct p2"): "left",
        ("instruct p2", "base p2"): "right",
    }
    client = _MockClient(decisions=decisions)
    pipeline = EvaluationPipeline(
        pipeline_config=EvaluationConfig(model="mock-judge", random_seed=42, max_retries=1, shard_size=10),
        output_dir=tmp_path / "bundle",
        client=client,
    )

    asyncio.run(pipeline.run(candidates=candidates, resume=False))
    first_calls = client.chat.completions.calls
    evaluation_dir = tmp_path / "bundle" / "evaluation"
    rows = _read_part_rows(evaluation_dir)
    for path in evaluation_dir.glob("part-*.parquet"):
        path.unlink()
    pq.write_table(pa.Table.from_pylist(rows[:1]), evaluation_dir / "part-0000.parquet")

    asyncio.run(pipeline.run(candidates=candidates, resume=True))
    assert client.chat.completions.calls == first_calls + 1
