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


def test_evaluation_minimal_schema_and_leaderboard(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "id": [1, 2, 3, 4, 5],
            "prompt_id": ["p1", "p1", "p1", "p2", "p2"],
            "prompt": ["prompt one", "prompt one", "prompt one", "prompt two", "prompt two"],
            "model_id": ["base-v1", "instruct-v1", "thinking-v1", "base-v1", "instruct-v1"],
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
        output_dir=tmp_path / "bundle",
        client=client,
    )

    asyncio.run(pipeline.run(candidates=candidates, resume=False))

    evaluations = _read_part_rows(tmp_path / "bundle" / "evaluation")
    assert len(evaluations) == 4
    assert client.chat.completions.calls == 4
    assert sorted(evaluations[0].keys()) == [
        "id",
        "left_candidate_id",
        "left_model",
        "left_model_id",
        "left_text",
        "prompt",
        "prompt_id",
        "right_candidate_id",
        "right_model",
        "right_model_id",
        "right_text",
        "winner",
    ]
    assert set(row["winner"] for row in evaluations) <= {"left", "right"}
    assert sorted(int(row["id"]) for row in evaluations) == [0, 1, 2, 3]

    leaderboard = _read_part_rows(tmp_path / "bundle" / "leaderboard")
    by_model_id = {str(row["model_id"]): row for row in leaderboard}
    assert by_model_id["thinking-v1"]["bt_score"] > by_model_id["base-v1"]["bt_score"]
    assert by_model_id["thinking-v1"]["wins"] >= by_model_id["base-v1"]["wins"]
    assert sorted(int(row["id"]) for row in leaderboard) == [0, 1, 2]


def test_evaluation_rejects_duplicate_candidate_id(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "id": [1, 1, 2],
            "prompt_id": ["p1", "p1", "p1"],
            "prompt": ["prompt one", "prompt one", "prompt one"],
            "model_id": ["base-v1", "instruct-v1", "instruct-v1"],
            "model": ["base", "instruct", "instruct"],
            "text": ["a", "b", "c"],
        }
    )
    pipeline = EvaluationPipeline(
        pipeline_config=EvaluationConfig(model="mock-judge"),
        output_dir=tmp_path / "bundle",
        client=_MockClient(decisions={}),
    )

    with pytest.raises(ValueError):
        asyncio.run(pipeline.run(candidates=candidates, resume=False))


def test_evaluation_requires_model_id(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "id": [1, 2],
            "prompt_id": ["p1", "p1"],
            "prompt": ["prompt one", "prompt one"],
            "model": ["base", "instruct"],
            "text": ["base p1", "instruct p1"],
        }
    )
    pipeline = EvaluationPipeline(
        pipeline_config=EvaluationConfig(model="mock-judge"),
        output_dir=tmp_path / "bundle",
        client=_MockClient(decisions={}),
    )

    with pytest.raises(ValueError):
        asyncio.run(pipeline.run(candidates=candidates, resume=False))


def test_evaluation_resume_skips_existing_rows(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "id": [1, 2],
            "prompt_id": ["p1", "p1"],
            "prompt": ["prompt one", "prompt one"],
            "model_id": ["base-v1", "instruct-v1"],
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
        output_dir=tmp_path / "bundle",
        client=client,
    )

    asyncio.run(pipeline.run(candidates=candidates, resume=True))
    first_calls = client.chat.completions.calls
    asyncio.run(pipeline.run(candidates=candidates, resume=True))
    assert client.chat.completions.calls == first_calls


def test_evaluation_retries_invalid_output(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "id": [1, 2],
            "prompt_id": ["p1", "p1"],
            "prompt": ["prompt one", "prompt one"],
            "model_id": ["base-v1", "instruct-v1"],
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
        output_dir=tmp_path / "bundle",
        client=client,
    )

    asyncio.run(pipeline.run(candidates=candidates, resume=False))
    rows = _read_part_rows(tmp_path / "bundle" / "evaluation")
    assert len(rows) == 1
    assert rows[0]["winner"] in {"left", "right"}
    assert client.chat.completions.calls == 2


def test_evaluation_resume_keeps_only_candidate_complete_pairs(tmp_path: Path) -> None:
    candidates = Dataset.from_dict(
        {
            "id": [1, 2, 3, 4],
            "prompt_id": ["p1", "p1", "p1", "p1"],
            "prompt": ["prompt one", "prompt one", "prompt one", "prompt one"],
            "model_id": ["base-v1", "base-v1", "instruct-v1", "instruct-v1"],
            "model": ["base", "base", "instruct", "instruct"],
            "text": ["base a", "base b", "instr a", "instr b"],
        }
    )
    decisions = {
        ("base a", "instr a"): "left",
        ("instr a", "base a"): "right",
        ("base a", "instr b"): "left",
        ("instr b", "base a"): "right",
        ("base b", "instr a"): "left",
        ("instr a", "base b"): "right",
        ("base b", "instr b"): "left",
        ("instr b", "base b"): "right",
    }
    client = _MockClient(decisions=decisions)
    pipeline = EvaluationPipeline(
        pipeline_config=EvaluationConfig(model="mock-judge", random_seed=42, max_retries=1, shard_size=10),
        output_dir=tmp_path / "bundle",
        client=client,
    )

    asyncio.run(pipeline.run(candidates=candidates, resume=False))
    first_calls = client.chat.completions.calls
    assert first_calls == 4

    evaluation_dir = tmp_path / "bundle" / "evaluation"
    existing_rows = _read_part_rows(evaluation_dir)
    assert len(existing_rows) == 4
    for path in evaluation_dir.glob("part-*.parquet"):
        path.unlink()
    pq.write_table(pa.Table.from_pylist(existing_rows[:3]), evaluation_dir / "part-0000.parquet")

    asyncio.run(pipeline.run(candidates=candidates, resume=True))
    assert client.chat.completions.calls == first_calls + 3

    rows = _read_part_rows(evaluation_dir)
    assert len(rows) == 4
