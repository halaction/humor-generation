from pathlib import Path

import pyarrow as pa

from src.config import JokesConfig, JokesDeduplicationConfig
from src.pipelines.jokes import JokesPipeline


def _build_table(rows: list[dict[str, object]]) -> pa.Table:
    return pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("text", pa.string()),
                pa.field("source_name", pa.string()),
                pa.field("source_filename", pa.string()),
                pa.field("source_id", pa.int64()),
            ]
        ),
    )


def test_jokes_pipeline_deduplicates_exact_and_near_verbatim_rows(tmp_path: Path) -> None:
    pipeline = JokesPipeline(
        pipeline_config=JokesConfig(
            deduplication=JokesDeduplicationConfig(
                enabled=True,
                min_tokens_for_near_match=4,
                token_jaccard_threshold=0.92,
                char_jaccard_threshold=0.90,
                edit_ratio_threshold=0.94,
            )
        ),
        output_dir=tmp_path / "jokes",
    )

    table = _build_table(
        [
            {
                "id": 0,
                "text": "What do you call a cheap circumcision? A rip-off.",
                "source_name": "short-jokes",
                "source_filename": "shortjokes.csv",
                "source_id": 1,
            },
            {
                "id": 1,
                "text": " What do you call a cheap circumcision? A rip off! ",
                "source_name": "r-jokes",
                "source_filename": "train.tsv",
                "source_id": 2,
            },
            {
                "id": 2,
                "text": "What do you call a cheap circumcision? A rip-off.",
                "source_name": "short-jokes",
                "source_filename": "shortjokes.csv",
                "source_id": 3,
            },
            {
                "id": 3,
                "text": "How do you make holy water? Boil the hell out of it.",
                "source_name": "short-jokes",
                "source_filename": "shortjokes.csv",
                "source_id": 4,
            },
        ]
    )

    deduplicated, stats = pipeline._deduplicate_table(table)
    rows = deduplicated.to_pylist()

    assert len(rows) == 2
    assert [str(row["text"]) for row in rows] == [
        "What do you call a cheap circumcision? A rip-off.",
        "How do you make holy water? Boil the hell out of it.",
    ]
    assert stats == {"raw_rows": 4, "kept_rows": 2, "exact_drops": 2, "token_set_drops": 0, "near_drops": 0}


def test_jokes_pipeline_deduplicates_token_set_variants(tmp_path: Path) -> None:
    pipeline = JokesPipeline(
        pipeline_config=JokesConfig(
            deduplication=JokesDeduplicationConfig(
                enabled=True,
                token_set_min_unique_tokens=5,
                min_tokens_for_near_match=4,
                token_jaccard_threshold=0.92,
                char_jaccard_threshold=0.90,
                edit_ratio_threshold=0.94,
            )
        ),
        output_dir=tmp_path / "jokes",
    )

    table = _build_table(
        [
            {
                "id": 0,
                "text": "What do you call an alligator in a vest? Investigator!",
                "source_name": "short-jokes",
                "source_filename": "shortjokes.csv",
                "source_id": 10,
            },
            {
                "id": 1,
                "text": "What do you call an alligator in a vest? An investigator!",
                "source_name": "r-jokes",
                "source_filename": "train.tsv",
                "source_id": 11,
            },
        ]
    )

    deduplicated, stats = pipeline._deduplicate_table(table)
    rows = deduplicated.to_pylist()

    assert len(rows) == 1
    assert str(rows[0]["text"]) == "What do you call an alligator in a vest? Investigator!"
    assert stats == {"raw_rows": 2, "kept_rows": 1, "exact_drops": 0, "token_set_drops": 1, "near_drops": 0}


def test_jokes_pipeline_deduplicates_near_rows_without_exact_match(tmp_path: Path) -> None:
    pipeline = JokesPipeline(
        pipeline_config=JokesConfig(
            deduplication=JokesDeduplicationConfig(
                enabled=True,
                min_tokens_for_near_match=4,
                token_jaccard_threshold=0.50,
                char_jaccard_threshold=0.40,
                edit_ratio_threshold=0.60,
            )
        ),
        output_dir=tmp_path / "jokes",
    )

    table = _build_table(
        [
            {
                "id": 0,
                "text": "What do you call a cheap circumcision? A rip-off.",
                "source_name": "short-jokes",
                "source_filename": "shortjokes.csv",
                "source_id": 20,
            },
            {
                "id": 1,
                "text": "What do you call a cheap cicumsicion? A rip off.",
                "source_name": "r-jokes",
                "source_filename": "train.tsv",
                "source_id": 21,
            },
        ]
    )

    deduplicated, stats = pipeline._deduplicate_table(table)
    rows = deduplicated.to_pylist()

    assert len(rows) == 1
    assert stats == {"raw_rows": 2, "kept_rows": 1, "exact_drops": 0, "token_set_drops": 0, "near_drops": 1}


def test_jokes_pipeline_keeps_distinct_punchlines(tmp_path: Path) -> None:
    pipeline = JokesPipeline(
        pipeline_config=JokesConfig(
            deduplication=JokesDeduplicationConfig(
                enabled=True,
                min_tokens_for_near_match=4,
                token_jaccard_threshold=0.95,
                char_jaccard_threshold=0.93,
                edit_ratio_threshold=0.96,
            )
        ),
        output_dir=tmp_path / "jokes",
    )

    table = _build_table(
        [
            {
                "id": 0,
                "text": "Why did the chicken cross the road? To get to the other side.",
                "source_name": "short-jokes",
                "source_filename": "shortjokes.csv",
                "source_id": 10,
            },
            {
                "id": 1,
                "text": "Why did the chicken cross the road? To get to the gay guy's house.",
                "source_name": "short-jokes",
                "source_filename": "shortjokes.csv",
                "source_id": 11,
            },
        ]
    )

    deduplicated, stats = pipeline._deduplicate_table(table)
    rows = deduplicated.to_pylist()

    assert len(rows) == 2
    assert stats == {"raw_rows": 2, "kept_rows": 2, "exact_drops": 0, "token_set_drops": 0, "near_drops": 0}
