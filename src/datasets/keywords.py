import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import HfApi
import pyarrow as pa
import pyarrow.parquet as pq

from src.config import config
from src.logging import get_logger
from src.paths import DATA_DIR
from src.settings import settings

logger = get_logger(__name__)


def _read_extraction_results(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            joke_id = str(payload["joke_id"])
            keywords = [str(item).strip() for item in payload["keywords"] if str(item).strip()]
            for keyword in keywords:
                records.append({"id": 0, "keyword": keyword, "joke_id": joke_id})

    for index, record in enumerate(records, start=1):
        record["id"] = index

    return records


def _write_parquet(records: list[dict[str, object]], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        records,
        schema=pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("keyword", pa.string()),
                pa.field("joke_id", pa.string()),
            ]
        ),
    )
    pq.write_table(
        table,
        destination,
        compression="zstd",
        use_content_defined_chunking=True,
        write_page_index=True,
    )
    return destination


def build_keywords_dataset() -> Path:
    keywords_results_path = DATA_DIR / config.keywords.results_filename
    if not keywords_results_path.exists():
        raise FileNotFoundError(
            "Keywords pipeline results were not found: "
            f"{keywords_results_path}. Run KeywordsPipeline.run first."
        )

    records = _read_extraction_results(keywords_results_path)
    output_path = DATA_DIR / config.keywords.data_filename
    _write_parquet(records=records, destination=output_path)
    logger.info(
        "build.done",
        rows=len(records),
        keywords_results_path=str(keywords_results_path),
        output_path=str(output_path),
    )
    return output_path


def publish_keywords_dataset(
    repo_id: str = settings.HF_DATASET_REPO_ID,
    config_name: str = config.keywords.hf_config_name,
    split: str = "train",
    private: bool = False,
) -> tuple[str, str]:
    target_path = DATA_DIR / config.keywords.data_filename
    if not target_path.exists():
        target_path = build_keywords_dataset()
    dataset = load_dataset("parquet", data_files=str(target_path), split=split)
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
        parquet_path=str(target_path),
        config_name=config_name,
        split=split,
    )
    return repo_id, config_name
