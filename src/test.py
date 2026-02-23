import asyncio

from datasets import load_dataset

from src.dataset.extraction import ExtractionPipeline
from src.dataset.jokes import build_jokes_dataset

if __name__ == "__main__":
    parquet_path = build_jokes_dataset()
    dataset = load_dataset("parquet", data_files=str(parquet_path), split="train[:10]")

    pipeline = ExtractionPipeline()
    results = asyncio.run(pipeline.run(dataset))

    for row, result in zip(dataset, results, strict=True):
        print(row["text"])
        print(result.model_dump())
        print()
