import csv
import gzip
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi

from datasets import load_dataset
from src.config import config
from src.logging import get_logger
from src.paths import DATA_DIR
from src.settings import settings

logger = get_logger(__name__)


def _parse_source_id(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return -1


class JokesPipeline:
    def _download_file(self, url: str, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists() or path.with_suffix("").exists():
            logger.debug("download.skip", url=url, path=str(path), reason="exists")
            return path

        logger.info("download.start", url=url, path=str(path))
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with path.open("wb") as output:
            output.write(response.content)
        logger.info("download.done", url=url, path=str(path), bytes_written=len(response.content))
        return path

    def _get_raw_github_url(self, url: str) -> str:
        return url.replace("https://github.com/", "https://raw.githubusercontent.com/").replace("/blob/", "/")

    def _get_filename_from_permalink(self, permalink: str) -> str:
        return Path(urlparse(permalink).path).name

    def _unzip_gz_file(self, path: Path) -> Path:
        extracted_path = path.with_suffix("")
        if extracted_path.exists():
            path.unlink(missing_ok=True)
            logger.debug("unzip.skip", source=str(path), destination=str(extracted_path), reason="exists")
            return extracted_path

        logger.info("unzip.start", source=str(path), destination=str(extracted_path))
        with gzip.open(path, "rb") as source, extracted_path.open("wb") as destination:
            destination.write(source.read())
        path.unlink(missing_ok=True)
        logger.info("unzip.done", source=str(path), destination=str(extracted_path))
        return extracted_path

    def _download_short_jokes(self) -> Path:
        destination_dir = DATA_DIR / "short-jokes"
        destination_dir.mkdir(parents=True, exist_ok=True)

        permalink = "https://github.com/amoudgl/short-jokes-dataset/blob/79c59bf8392929da3c560a3fa207be44e15b65db/shortjokes.csv"
        filename = self._get_filename_from_permalink(permalink)
        url = self._get_raw_github_url(permalink)
        self._download_file(url, destination_dir / filename)
        return destination_dir

    def _download_r_jokes(self) -> Path:
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

    def _write_parquet(self, records: list[dict[str, Any]], destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(
            records,
            schema=pa.schema(
                [
                    pa.field("id", pa.int64()),
                    pa.field("text", pa.string()),
                    pa.field("source_name", pa.string()),
                    pa.field("source_filename", pa.string()),
                    pa.field("source_id", pa.int64()),
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

    def _preprocess_short_jokes(self, destination_dir: Path) -> Path:
        input_path = destination_dir / "shortjokes.csv"
        output_path = destination_dir / "data.parquet"
        logger.info(
            "preprocess.start", dataset="short-jokes", source_path=str(input_path), output_path=str(output_path)
        )

        records: list[dict[str, Any]] = []
        with input_path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                original_id = (row.get("ID") or "").strip()
                text = (row.get("Joke") or "").strip()
                if not text:
                    continue

                records.append(
                    {
                        "id": 0,
                        "text": text,
                        "source_name": "short-jokes",
                        "source_filename": input_path.name,
                        "source_id": _parse_source_id(original_id),
                    }
                )

        self._write_parquet(records=records, destination=output_path)
        logger.info("preprocess.done", dataset="short-jokes", rows=len(records), output_path=str(output_path))
        return output_path

    def _preprocess_r_jokes(self, destination_dir: Path) -> Path:
        output_path = destination_dir / "data.parquet"
        logger.info(
            "preprocess.start", dataset="r-jokes", source_path=str(destination_dir), output_path=str(output_path)
        )

        records: list[dict[str, Any]] = []
        for split in ("train", "dev", "test"):
            input_path = destination_dir / f"{split}.tsv"
            with input_path.open(encoding="utf-8", newline="") as input_file:
                reader = csv.reader(input_file, delimiter="\t")
                for original_id, row in enumerate(reader, start=1):
                    if not row:
                        continue

                    text = "\t".join(row[1:]).strip()
                    if not text:
                        continue

                    records.append(
                        {
                            "id": 0,
                            "text": text,
                            "source_name": "r-jokes",
                            "source_filename": input_path.name,
                            "source_id": original_id,
                        }
                    )

        self._write_parquet(records=records, destination=output_path)
        logger.info("preprocess.done", dataset="r-jokes", rows=len(records), output_path=str(output_path))
        return output_path

    def build(self) -> Path:
        short_jokes_dir = self._download_short_jokes()
        short_jokes_path = self._preprocess_short_jokes(short_jokes_dir)

        r_jokes_dir = self._download_r_jokes()
        r_jokes_path = self._preprocess_r_jokes(r_jokes_dir)

        short_jokes_table = pq.read_table(short_jokes_path)
        r_jokes_table = pq.read_table(r_jokes_path)
        combined_table = pa.concat_tables([short_jokes_table, r_jokes_table])

        global_ids = pa.array(range(combined_table.num_rows), type=pa.int64())
        combined_table = combined_table.set_column(combined_table.schema.get_field_index("id"), "id", global_ids)

        output_path = DATA_DIR / config.jokes.data_filename
        pq.write_table(
            combined_table,
            output_path,
            compression="zstd",
            use_content_defined_chunking=True,
            write_page_index=True,
        )
        logger.info("build.done", rows=combined_table.num_rows, output_path=str(output_path))
        return output_path

    def publish(
        self,
        repo_id: str = settings.HF_DATASET_REPO_ID,
        config_name: str = config.jokes.hf_config_name,
        split: str = "train",
        private: bool = False,
    ) -> tuple[str, str]:
        target_path = DATA_DIR / config.jokes.data_filename
        if not target_path.exists():
            target_path = self.build()
        dataset = load_dataset("parquet", data_files=str(target_path), split=split)
        api = HfApi(token=settings.HF_TOKEN)
        api.create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True,
        )
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


def main() -> None:
    pipeline = JokesPipeline()
    jokes_path = pipeline.build()
    repo_id, config_name = pipeline.publish()
    print(
        {
            "jokes_path": str(jokes_path),
            "repo_id": repo_id,
            "config_name": config_name,
        }
    )


if __name__ == "__main__":
    main()
