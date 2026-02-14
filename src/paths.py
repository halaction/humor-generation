from pathlib import Path

SOURCE_DIR = Path(__file__).absolute().parent
BASE_DIR = SOURCE_DIR.parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
