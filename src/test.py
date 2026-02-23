import asyncio

from datasets import load_dataset

from src.dataset.extraction import ExtractionPipeline
from src.dataset.jokes import HF_DATASET_REPO_ID, JOKES_CONFIG_NAME
from src.dataset.keywords import build_keywords_dataset, publish_keywords_dataset
from src.settings import settings

if __name__ == "__main__":
    dataset = load_dataset(
        path=HF_DATASET_REPO_ID,
        name=JOKES_CONFIG_NAME,
        split="train[:20]",
        token=settings.HF_TOKEN,
    )

    pipeline = ExtractionPipeline()
    results, extraction_results_path = asyncio.run(pipeline.run(dataset))
    keywords_parquet_path = build_keywords_dataset(extraction_results_path=extraction_results_path)
    publish_keywords_dataset(parquet_path=keywords_parquet_path)

    for row, result in zip(dataset, results, strict=True):
        print(row["text"])
        print({"joke_id": result.joke_id, "keywords": result.keywords})
        print()
