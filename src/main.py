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
    embedding_map = {str(item["id"]): [float(value) for value in item["embedding"]] for item in embeddings_dataset}

    def attach_embedding(row: dict[str, object]) -> dict[str, list[float]]:
        joke_id = str(row["id"])
        embedding = embedding_map.get(joke_id)
        if embedding is None:
            raise ValueError(f"Missing embedding for joke id={joke_id}.")
        return {"embedding": embedding}

    dataset = jokes_dataset.map(attach_embedding, desc="Attach embeddings")
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
