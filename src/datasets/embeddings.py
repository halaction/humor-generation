import asyncio
from pathlib import Path

from huggingface_hub import HfApi
from pydantic import BaseModel

from datasets import load_dataset
from src.config import config
from src.datasets.jokes import build_jokes_dataset
from src.logging import get_logger
from src.paths import DATA_DIR
from src.pipelines.embeddings import EmbeddingsPipeline
from src.settings import settings

logger = get_logger(__name__)


def build_embeddings_dataset() -> Path:
    jokes_path = DATA_DIR / config.jokes.data_filename
    if not jokes_path.exists():
        jokes_path = build_jokes_dataset()

    jokes = load_dataset("parquet", data_files=str(jokes_path), split="train")
    pipeline = EmbeddingsPipeline()
    output_dir = asyncio.run(pipeline.run(jokes, resume=True))
    logger.info(
        "build.done",
        jokes_data_path=str(jokes_path),
        output_dir=str(output_dir),
    )
    return output_dir


def publish_embeddings_dataset(
    repo_id: str = settings.HF_DATASET_REPO_ID,
    config_name: str = config.embeddings.hf_config_name,
    split: str = "train",
    private: bool = False,
) -> Path:
    output_dir = DATA_DIR / config.embeddings.hf_config_name
    if not output_dir.exists():
        output_dir = build_embeddings_dataset()

    dataset = load_dataset("parquet", data_dir=str(output_dir), split=split)
    api = HfApi(token=settings.HF_TOKEN)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    dataset.push_to_hub(
        repo_id=repo_id,
        config_name=config_name,
        token=settings.HF_TOKEN,
        private=private,
    )
    logger.info(
        "publish.done",
        repo_id=repo_id,
        config_name=config_name,
        output_dir=str(output_dir),
        split=split,
    )
    return output_dir


if __name__ == "__main__":
    publish_embeddings_dataset()
