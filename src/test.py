import asyncio

from datasets import load_dataset

from src.config import JOKES_CONFIG_NAME
from src.datasets.embeddings import publish_embeddings_dataset
from src.datasets.keywords import build_keywords_dataset, publish_keywords_dataset
from src.pipelines.embedding import EmbeddingPipeline
from src.pipelines.extraction import ExtractionPipeline
from src.settings import settings

if __name__ == "__main__":
    dataset = load_dataset(
        path=settings.HF_DATASET_REPO_ID,
        name=JOKES_CONFIG_NAME,
        split="train[:20]",
        token=settings.HF_TOKEN,
    )

    pipeline = ExtractionPipeline()
    results, extraction_results_path = asyncio.run(pipeline.run(dataset))
    keywords_parquet_path = build_keywords_dataset(extraction_results_path=extraction_results_path)
    publish_keywords_dataset(parquet_path=keywords_parquet_path)
    embeddings_pipeline = EmbeddingPipeline()
    embeddings_outputs = asyncio.run(embeddings_pipeline.run(dataset))
    publish_embeddings_dataset(data_path=embeddings_outputs.data_path)

    for row, result in zip(dataset, results, strict=True):
        print(row["text"])
        print({"joke_id": result.joke_id, "keywords": result.keywords})
        print()

    print({"embeddings_path": str(embeddings_outputs.data_path)})
