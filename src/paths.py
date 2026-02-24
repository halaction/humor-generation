from pathlib import Path

SOURCE_DIR = Path(__file__).absolute().parent
BASE_DIR = SOURCE_DIR.parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATES_DIR = BASE_DIR / "templates"
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS_DIR = BASE_DIR / "configs"
CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

EXTRACTION_CONFIG_PATH = CONFIGS_DIR / "extraction.yaml"
EMBEDDING_CONFIG_PATH = CONFIGS_DIR / "embedding.yaml"

JOKES_DATA_PATH = DATA_DIR / "jokes.parquet"
KEYWORDS_DATA_PATH = DATA_DIR / "keywords.parquet"
EXTRACTION_RESULTS_PATH = DATA_DIR / "extraction.jsonl"
EMBEDDINGS_DATA_PATH = DATA_DIR / "embeddings.parquet"
