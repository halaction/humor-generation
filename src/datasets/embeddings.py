import asyncio
from pathlib import Path

from huggingface_hub import HfApi
from pydantic import BaseModel

from datasets import load_dataset
from src.datasets.jokes import build_jokes_dataset
from src.logging import get_logger
from src.paths import JOKES_DATA_PATH
from src.pipelines.embedding import EmbeddingPipeline
from src.settings import settings

logger = get_logger(__name__)


EMBEDDINGS_CONFIG_NAME = "embeddings"


class EmbeddingsDatasetOutputs(BaseModel):
    data_path: Path
    repo_id: str | None = None
    config_name: str | None = None


def build_embeddings_dataset(jokes_data_path: Path = JOKES_DATA_PATH) -> EmbeddingsDatasetOutputs:
    target_jokes_path = jokes_data_path
    if not target_jokes_path.exists():
        target_jokes_path = build_jokes_dataset()

    inputs = load_dataset("parquet", data_files=str(target_jokes_path), split="train")
    pipeline = EmbeddingPipeline()
    outputs = asyncio.run(pipeline.run(inputs))
    logger.info(
        "build.done",
        jokes_data_path=str(target_jokes_path),
        output_path=str(outputs.data_path),
    )
    return EmbeddingsDatasetOutputs(data_path=outputs.data_path)


def publish_embeddings_dataset(
    data_path: Path | None = None,
    repo_id: str = settings.HF_DATASET_REPO_ID,
    config_name: str | None = None,
    split: str = "train",
    private: bool = False,
) -> EmbeddingsDatasetOutputs:
    if data_path is None:
        build_outputs = build_embeddings_dataset()
        target_path = build_outputs.data_path
    else:
        target_path = data_path

    dataset = load_dataset("parquet", data_files=str(target_path), split=split)
    api = HfApi(token=settings.HF_TOKEN)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    dataset.push_to_hub(
        repo_id=repo_id,
        config_name=EMBEDDINGS_CONFIG_NAME,
        token=settings.HF_TOKEN,
        private=private,
    )
    logger.info(
        "publish.done",
        repo_id=repo_id,
        config_name=EMBEDDINGS_CONFIG_NAME,
        parquet_path=str(target_path),
        split=split,
    )
    return EmbeddingsDatasetOutputs(
        data_path=target_path,
        repo_id=repo_id,
        config_name=EMBEDDINGS_CONFIG_NAME,
    )
