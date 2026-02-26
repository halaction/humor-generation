import asyncio
import json
import math
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any, TextIO, cast

from datasets import Dataset
from openai import AsyncOpenAI
from pydantic import BaseModel

from src.config import config
from src.paths import DATA_DIR
from src.settings import settings

TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


class KeywordsRecord(BaseModel):
    joke_id: str
    keywords: list[str]


ExtractionInputs = Dataset
KeywordsOutputs = list[KeywordsRecord]


def _extract_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.lower())


def _generate_ngram_candidates(
    text: str,
    *,
    ngram_min: int,
    ngram_max: int,
    max_candidates: int,
) -> list[str]:
    tokens = _extract_tokens(text)
    if not tokens:
        return []

    seen: set[str] = set()
    candidates: list[str] = []
    upper = min(ngram_max, len(tokens))
    for ngram_size in range(ngram_min, upper + 1):
        for index in range(len(tokens) - ngram_size + 1):
            candidate = " ".join(tokens[index : index + ngram_size])
            if candidate in seen:
                continue
            seen.add(candidate)
            candidates.append(candidate)
            if len(candidates) >= max_candidates:
                return candidates

    return candidates


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


class KeywordsPipeline:
    def __init__(self) -> None:
        self.config = config.keywords
        self.embedding_config = config.embeddings
        if self.config.ngram_min > self.config.ngram_max:
            msg = (
                "Invalid n-gram range: "
                f"ngram_min={self.config.ngram_min} must be <= ngram_max={self.config.ngram_max}."
            )
            raise ValueError(msg)
        self.client = AsyncOpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=self.config.timeout,
        )

    async def _embed_text_batch(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.embeddings.create(
                    model=self.embedding_config.model,
                    input=texts,
                    dimensions=self.embedding_config.dimensions,
                )
            except Exception:
                if attempt + 1 >= self.config.max_retries:
                    raise
                await asyncio.sleep(2**attempt)
            else:
                vectors = [item.embedding for item in response.data]
                if len(vectors) != len(texts):
                    msg = (
                        "Embedding API returned mismatched outputs: "
                        f"expected={len(texts)} got={len(vectors)}"
                    )
                    raise ValueError(msg)
                return vectors

        msg = "Unexpected embedding retry flow."
        raise RuntimeError(msg)

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for index in range(0, len(texts), self.config.batch_size):
            batch = texts[index : index + self.config.batch_size]
            embeddings.extend(await self._embed_text_batch(batch))
        return embeddings

    async def _process_joke(self, joke_id: str, joke: str, semaphore: asyncio.Semaphore) -> KeywordsRecord:
        async with semaphore:
            cleaned_joke = str(joke).strip()
            candidates = _generate_ngram_candidates(
                cleaned_joke,
                ngram_min=self.config.ngram_min,
                ngram_max=self.config.ngram_max,
                max_candidates=self.config.max_candidates,
            )
            if not candidates:
                return KeywordsRecord(joke_id=joke_id, keywords=[])

            texts = [cleaned_joke, *candidates]
            vectors = await self._embed_texts(texts)
            joke_embedding = vectors[0]
            candidate_vectors = vectors[1:]

            scored: list[tuple[str, float]] = [
                (candidate, _cosine_similarity(joke_embedding, vector))
                for candidate, vector in zip(candidates, candidate_vectors, strict=True)
            ]
            scored.sort(key=lambda item: item[1], reverse=True)
            keywords = [keyword for keyword, _ in scored[: self.config.top_n]]
            return KeywordsRecord(joke_id=joke_id, keywords=keywords)

    @staticmethod
    def _load_existing_joke_ids(path: Path) -> set[str]:
        if not path.exists():
            return set()

        joke_ids: set[str] = set()
        with path.open(encoding="utf-8") as input_file:
            for raw_line in input_file:
                line = raw_line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                joke_ids.add(str(payload["joke_id"]))
        return joke_ids

    @staticmethod
    def _write_record(record: KeywordsRecord, output_file: TextIO) -> None:
        payload = {"joke_id": record.joke_id, "keywords": record.keywords}
        output_file.write(f"{json.dumps(payload, ensure_ascii=False)}\n")

    def _consume_completed(
        self,
        completed_tasks: Iterable[asyncio.Task[KeywordsRecord]],
        output_file: TextIO,
        seen_ids: set[str],
        results: KeywordsOutputs,
    ) -> None:
        for task in completed_tasks:
            result = task.result()
            if result.joke_id in seen_ids:
                continue
            seen_ids.add(result.joke_id)
            self._write_record(record=result, output_file=output_file)
            results.append(result)

    async def run(
        self,
        inputs: ExtractionInputs,
    ) -> tuple[KeywordsOutputs, Path]:
        if "text" not in inputs.column_names:
            msg = "Dataset must contain a 'text' column."
            raise ValueError(msg)
        if "id" not in inputs.column_names:
            msg = "Dataset must contain an 'id' column."
            raise ValueError(msg)

        output_path = DATA_DIR / self.config.results_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)

        seen_ids = self._load_existing_joke_ids(output_path)
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        results: KeywordsOutputs = []
        pending_tasks: set[asyncio.Task[KeywordsRecord]] = set()

        with output_path.open("a", encoding="utf-8") as output_file:
            for row in inputs:
                payload = cast("dict[str, Any]", row)
                joke_id = str(payload["id"])
                if joke_id in seen_ids:
                    continue

                joke_text = str(payload["text"])
                task = asyncio.create_task(self._process_joke(joke_id=joke_id, joke=joke_text, semaphore=semaphore))
                pending_tasks.add(task)

                if len(pending_tasks) >= self.config.max_parallel_requests:
                    done, _ = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                    pending_tasks -= done
                    self._consume_completed(
                        completed_tasks=done,
                        output_file=output_file,
                        seen_ids=seen_ids,
                        results=results,
                    )

            while pending_tasks:
                done, _ = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
                pending_tasks -= done
                self._consume_completed(
                    completed_tasks=done,
                    output_file=output_file,
                    seen_ids=seen_ids,
                    results=results,
                )

        return results, output_path
