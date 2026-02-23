import asyncio

from datasets import load_dataset

from src.dataset.extraction import ExtractionPipeline
from src.dataset.jokes import build_jokes_dataset, publish_jokes_dataset
from src.settings import settings

if __name__ == "__main__":
    parquet_path = build_jokes_dataset()
    repo_id, config_name = publish_jokes_dataset(parquet_path=parquet_path)
    dataset = load_dataset(
        path=repo_id,
        name=config_name,
        split="train[:10]",
        token=settings.HF_TOKEN,
    )

    pipeline = ExtractionPipeline()
    results = asyncio.run(pipeline.run(dataset))

    for row, result in zip(dataset, results, strict=True):
        print(row["text"])
        print(result.model_dump())
        print()
