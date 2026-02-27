import yaml
from pydantic import BaseModel, Field

from src.paths import CONFIGS_DIR
from src.settings import settings


class JokesConfig(BaseModel):
    hf_config_name: str = "jokes"
    data_filename: str = "jokes.parquet"


class KeywordsConfig(BaseModel):
    hf_config_name: str = "keywords"
    data_filename: str = "keywords.parquet"
    ngram_min: int = Field(default=1, ge=1)
    ngram_max: int = Field(default=3, ge=1)
    top_n: int = Field(default=3, ge=1)
    mmr_diversity: float = Field(default=0.7, ge=0.0, le=1.0)
    stopwords: bool = False
    max_candidates: int = Field(default=256, ge=1)
    batch_size: int = Field(default=128, gt=0)
    max_parallel_requests: int = Field(gt=0)
    timeout: int = Field(default=60, gt=0)
    max_retries: int = Field(default=5, gt=0)


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


class Config(BaseModel):
    jokes: JokesConfig
    keywords: KeywordsConfig
    embeddings: EmbeddingsConfig


config_path = CONFIGS_DIR / settings.CONFIG_FILENAME
config = Config.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
