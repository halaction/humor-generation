from pathlib import Path

from pydantic import BaseModel, Field
import yaml

from src.paths import BASE_DIR
from src.settings import settings


class JokesConfig(BaseModel):
    hf_config_name: str = "jokes"
    data_filename: str = "jokes.parquet"


class KeywordsConfig(BaseModel):
    hf_config_name: str = "keywords"
    data_filename: str = "keywords.parquet"


class EmbeddingsConfig(BaseModel):
    hf_config_name: str = "embeddings"
    data_filename: str = "embeddings.parquet"
    model: str
    dimensions: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    shard_size: int = Field(gt=0)
    max_parallel_requests: int = Field(gt=0)
    timeout: int = Field(gt=0)
    max_retries: int = Field(gt=0)


class ExtractionConfig(BaseModel):
    results_filename: str = "extraction.jsonl"
    model: str
    temperature: float
    max_completion_tokens: int = Field(gt=0)
    max_parallel_requests: int = Field(gt=0)


class Config(BaseModel):
    jokes: JokesConfig
    keywords: KeywordsConfig
    embeddings: EmbeddingsConfig
    extraction: ExtractionConfig


config_path = BASE_DIR / settings.CONFIG_FILENAME
config = Config.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
