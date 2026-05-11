from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset
from src.templates import environment


@dataclass
class MRVFRow:
    id: int
    keywords: list[str]
    prompt: str
    references: list[str]
    scores: list[float]


def build_prompt(keywords: list[str]) -> str:
    cleaned = [item.strip() for item in keywords if item and item.strip()]
    template = environment.get_template("reference_prompt.j2")
    return template.render(keywords=cleaned).strip()


def normalize_row(row: dict[str, Any], max_reference_samples: int) -> MRVFRow | None:
    keywords = [item.strip() for item in row["keywords"] if item and item.strip()]
    if not keywords:
        return None

    raw_references = row.get("references", [])
    raw_scores = row.get("scores", [])
    pairs: list[tuple[str, float]] = []
    for idx, reference in enumerate(raw_references):
        if not reference:
            continue
        cleaned_reference = str(reference).strip()
        if not cleaned_reference:
            continue
        score = float(raw_scores[idx]) if idx < len(raw_scores) else 0.0
        pairs.append((cleaned_reference, score))
    if not pairs:
        return None

    dedup_pairs: list[tuple[str, float]] = []
    seen: set[str] = set()
    for reference, score in pairs:
        if reference in seen:
            continue
        seen.add(reference)
        dedup_pairs.append((reference, score))

    dedup_references = [item[0] for item in dedup_pairs]
    dedup_scores = [item[1] for item in dedup_pairs]
    limit = max_reference_samples if max_reference_samples > 0 else len(dedup_references)
    dedup_references = dedup_references[:limit]
    scores = dedup_scores[:limit]
    if not dedup_references:
        return None

    return MRVFRow(
        id=int(row["id"]),
        keywords=keywords,
        prompt=build_prompt(keywords),
        references=dedup_references,
        scores=scores,
    )


def prepare_mrvf_dataset(dataset: Dataset, max_reference_samples: int) -> Dataset:
    rows: list[dict[str, Any]] = []
    for row in dataset:
        normalized = normalize_row(dict(row), max_reference_samples=max_reference_samples)
        if normalized is None:
            continue
        rows.append(
            {
                "id": normalized.id,
                "keywords": normalized.keywords,
                "prompt": normalized.prompt,
                "references": normalized.references,
                "scores": normalized.scores,
            }
        )
    return Dataset.from_list(rows)


def load_reference_splits(dataset_name: str, dataset_config_name: str, train_split: str, eval_split: str) -> tuple[Dataset, Dataset]:
    dataset = load_dataset(dataset_name, dataset_config_name)
    train_dataset = dataset[train_split]
    eval_dataset = dataset[eval_split]
    return train_dataset, eval_dataset
