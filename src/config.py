import yaml
from pydantic import BaseModel, Field

from src.paths import CONFIGS_DIR
from src.settings import settings


class JokesConfig(BaseModel):
    hf_config_name: str = "jokes"
    data_filename: str = "jokes.parquet"


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


class KeywordsConfig(BaseModel):
    hf_config_name: str = "keywords"
    data_filename: str = "keywords.parquet"
    ngram_min: int = Field(default=1, ge=1)
    ngram_max: int = Field(default=3, ge=1)
    top_n: int = Field(default=3, ge=1)
    mmr_diversity: float = Field(default=0.7, ge=0.0, le=1.0)
    stopwords: bool = False
    max_candidates: int = Field(default=256, ge=1)
    model: str
    dimensions: int = Field(gt=0)
    batch_size: int = Field(default=128, gt=0)
    shard_size: int = Field(gt=0)
    max_parallel_requests: int = Field(gt=0)
    timeout: int = Field(default=60, gt=0)
    max_retries: int = Field(default=5, gt=0)


class ReferencesConfig(BaseModel):
    hf_config_name: str = "references"
    data_filename: str = "references.parquet"
    model: str
    dimensions: int = Field(gt=0)
    top_k: int = Field(default=20, gt=0)
    min_similarity: float = 0.0
    batch_size: int = Field(default=64, gt=0)
    shard_size: int = Field(default=10000, gt=0)
    max_parallel_requests: int = Field(default=5, gt=0)
    timeout: int = Field(default=60, gt=0)
    max_retries: int = Field(default=5, gt=0)
    faiss_nlist: int = Field(default=4096, gt=0)
    faiss_nprobe: int = Field(default=32, gt=0)
    faiss_train_size: int = Field(default=200000, gt=0)
    faiss_add_batch_size: int = Field(default=10000, gt=0)
    index_dirname: str = "references-index"
    query_instruction: str = "Given a prompt asking for a joke, retrieve jokes that would answer the prompt."
    exclude_self: bool = True
    oversample: int = Field(default=10, ge=0)


class Config(BaseModel):
    jokes: JokesConfig
    keywords: KeywordsConfig
    embeddings: EmbeddingsConfig
    references: ReferencesConfig


config_path = CONFIGS_DIR / settings.CONFIG_FILENAME
config = Config.model_validate(yaml.safe_load(config_path.read_text(encoding="utf-8")))
