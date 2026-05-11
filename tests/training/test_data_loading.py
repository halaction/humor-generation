import pytest

datasets = pytest.importorskip("datasets")
Dataset = datasets.Dataset
from src.training.data import prepare_mrvf_dataset


def test_prepare_mrvf_dataset_filters_and_deduplicates() -> None:
    dataset = Dataset.from_list(
        [
            {
                "id": 1,
                "keywords": ["cat", "dog"],
                "references": ["j1", "j1", "  ", "j2"],
                "scores": [0.9, 0.8, 0.1, 0.5],
            },
            {
                "id": 2,
                "keywords": [" "],
                "references": ["j3"],
                "scores": [0.2],
            },
        ]
    )
    prepared = prepare_mrvf_dataset(dataset, max_reference_samples=2)
    assert len(prepared) == 1
    row = prepared[0]
    assert row["id"] == 1
    assert row["references"] == ["j1", "j2"]
    assert row["scores"] == [0.9, 0.5]
    assert row["prompt"] == "Write a joke using the following keyword(s): cat, dog"
