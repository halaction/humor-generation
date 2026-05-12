import csv
import difflib
import gzip
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from huggingface_hub import HfApi

from datasets import load_dataset
from src.config import JokesConfig, config
from src.logging import get_logger
from src.paths import DATA_DIR
from src.settings import settings
from src.pipelines.base import BasePipeline

logger = get_logger(__name__)
_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")


def _parse_source_id(value: str) -> int:
    try:
        return int(value)
    except ValueError:
        return -1


class JokesPipeline(BasePipeline):
    def __init__(
        self,
        pipeline_config: JokesConfig | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.config = pipeline_config or config.jokes
        self.output_dir = output_dir or DATA_DIR / self.config.hf_config_name

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

    @staticmethod
    def _normalize_exact(text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        normalized = _NON_ALNUM_PATTERN.sub(" ", normalized)
        return " ".join(normalized.split())

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(text)

    @staticmethod
    def _token_fingerprint(tokens: list[str]) -> tuple[str, ...]:
        return tuple(sorted(set(tokens)))

    @staticmethod
    def _char_shingles(text: str, width: int = 3) -> set[str]:
        compact = text.replace(" ", "")
        if len(compact) < width:
            return {compact} if compact else set()
        return {compact[index : index + width] for index in range(0, len(compact) - width + 1)}

    @staticmethod
    def _jaccard_similarity(left: set[str], right: set[str]) -> float:
        if not left and not right:
            return 1.0
        union = left | right
        if not union:
            return 1.0
        return len(left & right) / len(union)

    def _is_near_duplicate(
        self,
        incoming_normalized: str,
        incoming_tokens: list[str],
        candidate_normalized: str,
        candidate_tokens: list[str],
    ) -> bool:
        min_tokens = self.config.deduplication.min_tokens_for_near_match
        if len(incoming_tokens) < min_tokens or len(candidate_tokens) < min_tokens:
            return False

        incoming_token_set = set(incoming_tokens)
        candidate_token_set = set(candidate_tokens)
        token_jaccard = self._jaccard_similarity(incoming_token_set, candidate_token_set)
        if token_jaccard < self.config.deduplication.token_jaccard_threshold:
            return False

        incoming_shingles = self._char_shingles(incoming_normalized)
        candidate_shingles = self._char_shingles(candidate_normalized)
        char_jaccard = self._jaccard_similarity(incoming_shingles, candidate_shingles)
        if char_jaccard < self.config.deduplication.char_jaccard_threshold:
            return False

        edit_ratio = difflib.SequenceMatcher(
            None,
            incoming_normalized,
            candidate_normalized,
            autojunk=False,
        ).ratio()
        return edit_ratio >= self.config.deduplication.edit_ratio_threshold

    def _deduplicate_table(self, table: pa.Table) -> tuple[pa.Table, dict[str, int]]:
        if not self.config.deduplication.enabled or table.num_rows == 0:
            return table, {
                "raw_rows": table.num_rows,
                "kept_rows": table.num_rows,
                "exact_drops": 0,
                "token_set_drops": 0,
                "near_drops": 0,
            }

        rows = table.to_pylist()

        seen_exact: set[str] = set()
        seen_token_sets: set[tuple[str, ...]] = set()
        kept_rows: list[dict[str, Any]] = []
        normalized_texts: list[str] = []
        token_lists: list[list[str]] = []
        token_index: dict[tuple[int, str, str], list[int]] = {}
        exact_drops = 0
        token_set_drops = 0
        near_drops = 0

        for row in rows:
            text = str(row["text"])
            normalized = self._normalize_exact(text)
            if not normalized:
                continue

            if normalized in seen_exact:
                exact_drops += 1
                continue

            tokens = self._tokenize(normalized)
            if not tokens:
                continue

            token_fingerprint = self._token_fingerprint(tokens)
            if len(token_fingerprint) >= self.config.deduplication.token_set_min_unique_tokens:
                if token_fingerprint in seen_token_sets:
                    token_set_drops += 1
                    continue

            near_match = False
            first_token = tokens[0]
            last_token = tokens[-1]
            token_count_bucket = len(tokens) // 4
            bucket_key = (token_count_bucket, first_token, last_token)
            candidate_indices = token_index.get(bucket_key, [])
            for candidate_index in candidate_indices:
                if self._is_near_duplicate(
                    incoming_normalized=normalized,
                    incoming_tokens=tokens,
                    candidate_normalized=normalized_texts[candidate_index],
                    candidate_tokens=token_lists[candidate_index],
                ):
                    near_match = True
                    near_drops += 1
                    break
            if near_match:
                continue

            seen_exact.add(normalized)
            if len(token_fingerprint) >= self.config.deduplication.token_set_min_unique_tokens:
                seen_token_sets.add(token_fingerprint)
            kept_rows.append(row)
            normalized_texts.append(normalized)
            token_lists.append(tokens)
            token_index.setdefault(bucket_key, []).append(len(kept_rows) - 1)

        deduplicated_table = pa.Table.from_pylist(kept_rows, schema=table.schema)
        stats = {
            "raw_rows": table.num_rows,
            "kept_rows": deduplicated_table.num_rows,
            "exact_drops": exact_drops,
            "token_set_drops": token_set_drops,
            "near_drops": near_drops,
        }
        return deduplicated_table, stats

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

    def build(self) -> None:
        short_jokes_dir = self._download_short_jokes()
        short_jokes_path = self._preprocess_short_jokes(short_jokes_dir)

        r_jokes_dir = self._download_r_jokes()
        r_jokes_path = self._preprocess_r_jokes(r_jokes_dir)

        short_jokes_table = pq.read_table(short_jokes_path)
        r_jokes_table = pq.read_table(r_jokes_path)
        combined_table = pa.concat_tables([short_jokes_table, r_jokes_table])
        combined_table, dedup_stats = self._deduplicate_table(combined_table)

        global_ids = pa.array(range(combined_table.num_rows), type=pa.int64())
        combined_table = combined_table.set_column(combined_table.schema.get_field_index("id"), "id", global_ids)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "part-0000.parquet"
        pq.write_table(
            combined_table,
            output_path,
            compression="zstd",
            use_content_defined_chunking=True,
            write_page_index=True,
        )
        logger.info(
            "build.done",
            rows=combined_table.num_rows,
            raw_rows=dedup_stats["raw_rows"],
            dedup_enabled=self.config.deduplication.enabled,
            dedup_exact_drops=dedup_stats["exact_drops"],
            dedup_token_set_drops=dedup_stats["token_set_drops"],
            dedup_near_drops=dedup_stats["near_drops"],
            output_path=str(output_path),
        )

    def publish(
        self,
        repo_id: str = settings.HF_DATASET_REPO_ID,
        config_name: str = config.jokes.hf_config_name,
        split: str = "train",
        private: bool = False,
    ) -> None:
        output_dir = self.output_dir
        if not output_dir.exists():
            output_dir = self.build()

        dataset = load_dataset("parquet", data_dir=str(output_dir), split=split)
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
            output_dir=str(output_dir),
            config_name=config_name,
            split=split,
        )


def main() -> None:
    pipeline = JokesPipeline()
    pipeline.build()
    pipeline.publish()

    logger.info(
        "main.done",
        jokes_path=str(pipeline.output_dir),
        repo_id=settings.HF_DATASET_REPO_ID,
        config_name=pipeline.config.hf_config_name,
    )


if __name__ == "__main__":
    main()
