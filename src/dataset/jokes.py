import csv
import gzip
from pathlib import Path
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import requests

from src.logging import get_logger
from src.paths import DATA_DIR

logger = get_logger(__name__)


class JokesDataset:
    def __init__(self) -> None: ...

    @staticmethod
    def _download_file(url: str, path: Path) -> Path:
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

    @staticmethod
    def _write_parquet(records: list[dict[str, str]], destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pylist(
            records,
            schema=pa.schema(
                [
                    pa.field("id", pa.string()),
                    pa.field("text", pa.string()),
                    pa.field("source_name", pa.string()),
                    pa.field("source_filename", pa.string()),
                    pa.field("source_id", pa.string()),
                ]
            ),
        )
        pq.write_table(table, destination)
        return destination

    def _preprocess_short_jokes(self, destination_dir: Path) -> Path:
        input_path = destination_dir / "shortjokes.csv"
        output_path = destination_dir / "data.parquet"

        logger.info(
            "preprocess.start",
            dataset="short-jokes",
            source_path=str(input_path),
            output_path=str(output_path),
        )

        records: list[dict[str, str]] = []
        with input_path.open(encoding="utf-8", newline="") as input_file:
            reader = csv.DictReader(input_file)
            for row in reader:
                original_id = (row.get("ID") or "").strip()
                text = (row.get("Joke") or "").strip()

                if not text:
                    continue

                source_name = "short-jokes"
                source_filename = input_path.name
                source_id = original_id
                records.append(
                    {
                        "id": "",
                        "text": text,
                        "source_name": source_name,
                        "source_filename": source_filename,
                        "source_id": source_id,
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

        records: list[dict[str, str]] = []
        for split in ("train", "dev", "test"):
            input_path = destination_dir / f"{split}.tsv"
            with input_path.open(encoding="utf-8", newline="") as input_file:
                reader = csv.reader(input_file, delimiter="\t")
                for original_id, row in enumerate(reader, start=1):
                    if not row:
                        continue

                    _score = row[0].strip()
                    text = "\t".join(row[1:]).strip()
                    if not text:
                        continue

                    source_name = "r-jokes"
                    source_filename = input_path.name
                    source_id = str(original_id)
                    records.append(
                        {
                            "id": "",
                            "text": text,
                            "source_name": source_name,
                            "source_filename": source_filename,
                            "source_id": source_id,
                        }
                    )

        self._write_parquet(records=records, destination=output_path)
        logger.info("preprocess.done", dataset="r-jokes", rows=len(records), output_path=str(output_path))
        return output_path

    def _collect_data(self) -> Path:
        short_jokes_dir = self._download_short_jokes()
        short_jokes_path = self._preprocess_short_jokes(short_jokes_dir)

        r_jokes_dir = self._download_r_jokes()
        r_jokes_path = self._preprocess_r_jokes(r_jokes_dir)

        short_jokes_table = pq.read_table(short_jokes_path)
        r_jokes_table = pq.read_table(r_jokes_path)
        combined_table = pa.concat_tables([short_jokes_table, r_jokes_table])

        global_ids = pc.cast(pa.array(range(combined_table.num_rows), type=pa.int64()), pa.string())
        combined_table = combined_table.set_column(
            combined_table.schema.get_field_index("id"),
            "id",
            global_ids,
        )

        output_path = DATA_DIR / "jokes.parquet"
        pq.write_table(combined_table, output_path)

        logger.info("collect.done", rows=combined_table.num_rows, output_path=str(output_path))
        return output_path
