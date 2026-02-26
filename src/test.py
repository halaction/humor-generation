import asyncio

from datasets import load_dataset

from src.config import config
from src.datasets.embeddings import publish_embeddings_dataset
from src.pipelines.embedding import EmbeddingPipeline
from src.settings import settings

if __name__ == "__main__":
    dataset = load_dataset(
        path=settings.HF_DATASET_REPO_ID,
        name=config.jokes.hf_config_name,
        split="train[:30]",
        token=settings.HF_TOKEN,
    )

    embeddings_pipeline = EmbeddingPipeline()
    embeddings_outputs = asyncio.run(embeddings_pipeline.run(dataset))
    publish_outputs = publish_embeddings_dataset()

    print(
        {
            "embeddings_path": str(embeddings_outputs.data_path),
            "repo_id": publish_outputs.repo_id,
            "config_name": publish_outputs.config_name,
        }
    )
