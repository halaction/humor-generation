import asyncio

from datasets import load_dataset
from src.config import config
from src.datasets.embeddings import build_embeddings_dataset
from src.datasets.jokes import build_jokes_dataset
from src.paths import DATA_DIR
from src.pipelines.keywords import KeywordsPipeline


if __name__ == "__main__":
    jokes_path = DATA_DIR / config.jokes.data_filename
    if not jokes_path.exists():
        jokes_path = build_jokes_dataset()
    embeddings_path = DATA_DIR / config.embeddings.data_filename
    if not embeddings_path.exists():
        embeddings_path = build_embeddings_dataset().data_path

    jokes_dataset = load_dataset("parquet", data_files=str(jokes_path), split="train[:30]")
    embeddings_dataset = load_dataset("parquet", data_files=str(embeddings_path), split="train")
    keywords_pipeline = KeywordsPipeline()
    keyword_outputs = asyncio.run(keywords_pipeline.run(jokes_dataset, embeddings_dataset))
    print(
        {
            "jokes_path": str(jokes_path),
            "results_path": str(keyword_outputs.data_path),
            "processed_count": len(keyword_outputs.results),
        }
    )

    print(keyword_outputs.results)
