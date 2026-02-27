import asyncio

from datasets import load_dataset

from src.config import config
from src.datasets.jokes import build_jokes_dataset
from src.paths import DATA_DIR
from src.pipelines.keywords import KeywordsPipeline

if __name__ == "__main__":
    jokes_path = DATA_DIR / config.jokes.data_filename
    if not jokes_path.exists():
        jokes_path = build_jokes_dataset()

    dataset = load_dataset("parquet", data_files=str(jokes_path), split="train[:30]")
    keywords_pipeline = KeywordsPipeline()
    keyword_outputs, results_path = asyncio.run(keywords_pipeline.run(dataset))
    print(
        {
            "jokes_path": str(jokes_path),
            "results_path": str(results_path),
            "processed_count": len(keyword_outputs),
        }
    )

    print(keyword_outputs)
