import gzip
from pathlib import Path
from urllib.parse import urlparse

import requests

from src.logging import get_logger
from src.paths import DATA_DIR

logger = get_logger(__name__)


class Dataset:
    def __init__(self) -> None: ...

    @staticmethod
    def _download_file(url: str, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            logger.debug("download.skip", url=url, path=str(path), reason="exists")
            return path

        logger.info("download.start", url=url, path=str(path))
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with path.open("wb") as output:
            output.write(response.content)
        logger.info("download.done", url=url, path=str(path), bytes_written=len(response.content))

        return path

    @staticmethod
    def _get_raw_github_url(url: str) -> str:
        return url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")

    @staticmethod
    def _get_filename_from_permalink(permalink: str) -> str:
        return Path(urlparse(permalink).path).name

    @staticmethod
    def _unzip_gz_file(path: Path) -> Path:
        extracted_path = path.with_suffix("")
        if extracted_path.exists():
            logger.debug("unzip.skip", source=str(path), destination=str(extracted_path), reason="exists")
            return extracted_path

        logger.info("unzip.start", source=str(path), destination=str(extracted_path))
        with gzip.open(path, "rb") as source, extracted_path.open("wb") as destination:
            destination.write(source.read())
        logger.info("unzip.done", source=str(path), destination=str(extracted_path))

        return extracted_path

    def _download_short_jokes(self) -> Path:
        # https://github.com/amoudgl/short-jokes-dataset.git

        destination_dir = DATA_DIR / "short-jokes"
        destination_dir.mkdir(parents=True, exist_ok=True)

        permalink = "https://github.com/amoudgl/short-jokes-dataset/blob/79c59bf8392929da3c560a3fa207be44e15b65db/shortjokes.csv"
        filename = self._get_filename_from_permalink(permalink)
        url = self._get_raw_github_url(permalink)
        self._download_file(url, destination_dir / filename)

        return destination_dir

    def _download_r_jokes(self) -> Path:
        # https://github.com/orionw/rJokesData.git

        destination_dir = DATA_DIR / "r-jokes"
        destination_dir.mkdir(parents=True, exist_ok=True)

        permalinks = [
            "https://github.com/orionw/rJokesData/blob/d48bedd71bdacc7557b84ad697bf556e7aad7c21/data/train.tsv.gz",
            "https://github.com/orionw/rJokesData/blob/d48bedd71bdacc7557b84ad697bf556e7aad7c21/data/dev.tsv.gz",
            "https://github.com/orionw/rJokesData/blob/d48bedd71bdacc7557b84ad697bf556e7aad7c21/data/test.tsv.gz",
        ]

        for permalink in permalinks:
            filename = self._get_filename_from_permalink(permalink)
            url = self._get_raw_github_url(permalink)
            gz_path = self._download_file(url, destination_dir / filename)
            self._unzip_gz_file(gz_path)

        return destination_dir
