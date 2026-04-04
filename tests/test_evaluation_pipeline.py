import asyncio
from pathlib import Path

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


def _read_rows(path: Path) -> list[dict[str, object]]:
    return pq.read_table(path).to_pylist()


def test_evaluation_minimal_schema_and_leaderboard(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "prompt_id": ["p1", "p1", "p1", "p2", "p2"],
            "prompt": ["prompt one", "prompt one", "prompt one", "prompt two", "prompt two"],
            "model": ["base", "instruct", "thinking", "base", "instruct"],
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
        output_dir=tmp_path / "evaluation",
        client=client,
    )

    asyncio.run(pipeline.run(candidates=candidates, resume=False))

    evaluations = _read_rows(tmp_path / "evaluation" / "evaluation.parquet")
    assert len(evaluations) == 4
    assert client.chat.completions.calls == 4
    assert sorted(evaluations[0].keys()) == [
        "left_model",
        "left_text",
        "prompt",
        "prompt_id",
        "right_model",
        "right_text",
        "winner",
    ]
    assert set(row["winner"] for row in evaluations) <= {"left", "right"}

    leaderboard = _read_rows(tmp_path / "evaluation" / "leaderboard.parquet")
    by_model = {row["model"]: row for row in leaderboard}
    assert by_model["thinking"]["bt_score"] > by_model["base"]["bt_score"]
    assert by_model["thinking"]["wins"] >= by_model["base"]["wins"]


def test_evaluation_rejects_duplicate_prompt_model(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "prompt_id": ["p1", "p1", "p1"],
            "prompt": ["prompt one", "prompt one", "prompt one"],
            "model": ["base", "base", "instruct"],
            "text": ["a", "b", "c"],
        }
    )
    pipeline = EvaluationPipeline(
        pipeline_config=EvaluationConfig(model="mock-judge"),
        output_dir=tmp_path / "evaluation",
        client=_MockClient(decisions={}),
    )

    with pytest.raises(ValueError):
        asyncio.run(pipeline.run(candidates=candidates, resume=False))


def test_evaluation_resume_skips_existing_rows(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "prompt_id": ["p1", "p1"],
            "prompt": ["prompt one", "prompt one"],
            "model": ["base", "instruct"],
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
        output_dir=tmp_path / "evaluation",
        client=client,
    )

    asyncio.run(pipeline.run(candidates=candidates, resume=True))
    first_calls = client.chat.completions.calls
    asyncio.run(pipeline.run(candidates=candidates, resume=True))
    assert client.chat.completions.calls == first_calls


def test_evaluation_retries_invalid_output(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "prompt_id": ["p1", "p1"],
            "prompt": ["prompt one", "prompt one"],
            "model": ["base", "instruct"],
            "text": ["base p1", "instruct p1"],
        }
    )
    decisions = {
        ("base p1", "instruct p1"): "left",
        ("instruct p1", "base p1"): "right",
    }
    client = _MockClient(decisions=decisions, invalid_before_success=1)
    pipeline = EvaluationPipeline(
        pipeline_config=EvaluationConfig(model="mock-judge", max_retries=2, random_seed=42),
        output_dir=tmp_path / "evaluation",
        client=client,
    )

    asyncio.run(pipeline.run(candidates=candidates, resume=False))
    rows = _read_rows(tmp_path / "evaluation" / "evaluation.parquet")
    assert len(rows) == 1
    assert rows[0]["winner"] in {"left", "right"}
    assert client.chat.completions.calls == 2


def test_evaluation_accepts_system_id_and_response_text_aliases(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "prompt_id": ["p1", "p1"],
            "prompt": ["prompt one", "prompt one"],
            "system_id": ["base", "instruct"],
            "response_text": ["base p1", "instruct p1"],
        }
    )
    decisions = {
        ("base p1", "instruct p1"): "left",
        ("instruct p1", "base p1"): "right",
    }
    pipeline = EvaluationPipeline(
        pipeline_config=EvaluationConfig(model="mock-judge", random_seed=42),
        output_dir=tmp_path / "evaluation",
        client=_MockClient(decisions=decisions),
    )

    asyncio.run(pipeline.run(candidates=candidates, resume=False))
    rows = _read_rows(tmp_path / "evaluation" / "evaluation.parquet")
    assert len(rows) == 1
