from datasets import load_dataset
from huggingface_hub import HfApi

from src.config import config
from src.logging import get_logger
from src.paths import DATA_DIR
from src.settings import settings

logger = get_logger(__name__)


def build_keywords_dataset() -> str:
    output_path = DATA_DIR / config.keywords.data_filename
    if not output_path.exists():
        raise FileNotFoundError(f"Keywords parquet was not found: {output_path}. Run KeywordsPipeline.run first.")
    logger.info("build.done", output_path=str(output_path))
    return str(output_path)


def publish_keywords_dataset(
    repo_id: str = settings.HF_DATASET_REPO_ID,
    config_name: str = config.keywords.hf_config_name,
    split: str = "train",
    private: bool = False,
) -> tuple[str, str]:
    target_path = build_keywords_dataset()
    dataset = load_dataset("parquet", data_files=target_path, split=split)
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
        parquet_path=target_path,
        config_name=config_name,
        split=split,
    )
    return repo_id, config_name
