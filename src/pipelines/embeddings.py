import asyncio
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from datasets import Dataset, load_dataset
from src.config import EmbeddingsConfig, config
from src.datasets.jokes import build_jokes_dataset
from src.logging import get_logger
from src.models import EmbeddingsInputs, EmbeddingsOutputs
from src.paths import DATA_DIR
from src.pipelines.base import BasePipeline
from src.settings import settings

logger = get_logger(__name__)


class EmbeddingsPipeline(BasePipeline):
    def __init__(
        self,
        pipeline_config: EmbeddingsConfig | None = None,
        output_dir: Path | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.config = pipeline_config or config.embeddings
        self.output_dir = output_dir or DATA_DIR / self.config.hf_config_name
        self.client = client or AsyncOpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=self.config.timeout,
        )
        self.next_part_index = 0

        self.schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), self.config.dimensions)),
            ]
        )

    async def _embed_jokes(
        self,
        batch: EmbeddingsInputs,
        semaphore: asyncio.Semaphore,
    ) -> EmbeddingsOutputs | None:
        async with semaphore:
            filtered_pairs = [
                (item_id, text.strip()) for item_id, text in zip(batch.id, batch.text, strict=True) if text.strip()
            ]
            if not filtered_pairs:
                return None
            filtered_ids = [item_id for item_id, _ in filtered_pairs]
            filtered_texts = [text for _, text in filtered_pairs]

            for attempt in range(1, self.config.max_retries + 1):
                try:
                    response = await self.client.embeddings.create(
                        model=self.config.model,
                        input=filtered_texts,
                        dimensions=self.config.dimensions,
                    )

                    outputs = EmbeddingsOutputs(
                        id=filtered_ids,
                        embedding=[item.embedding for item in response.data],
                    )
                except Exception:
                    if attempt + 1 >= self.config.max_retries:
                        raise
                    await asyncio.sleep(2 ** (attempt - 1))
                else:
                    return outputs

        msg = "Unexpected retry error"
        raise RuntimeError(msg)

    def _get_table(self, write_buffer: list[EmbeddingsOutputs]) -> pa.Table:
        outputs = defaultdict(list)
        for batch in write_buffer:
            for key, value in batch.model_dump().items():
                outputs[key].extend(value)

        return pa.Table.from_pydict(outputs, schema=self.schema)

    def _check_buffer_size(self, write_buffer: list[EmbeddingsOutputs]) -> bool:
        return sum(len(batch.id) for batch in write_buffer) >= self.config.shard_size

    async def run(
        self,
        jokes: Dataset,
        resume: bool = False,
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.next_part_index = self._get_next_part_index()

        dataset = jokes

        if resume:
            seen_ids = self._get_seen_ids()
            dataset = dataset.filter(lambda item: item["id"] not in seen_ids)
        elif self.next_part_index > 0:
            for file in self.output_dir.glob("part-*.parquet"):
                file.unlink()

        dataset = dataset.batch(self.config.batch_size)

        write_buffer: list[EmbeddingsOutputs] = []
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        pending_tasks: set[asyncio.Task[EmbeddingsOutputs | None]] = set()

        for batch in tqdm(dataset):
            batch = cast("dict[str, list[Any]]", batch)
            inputs = EmbeddingsInputs(id=batch["id"], text=batch["text"])

            task = asyncio.create_task(self._embed_jokes(inputs, semaphore))
            pending_tasks.add(task)

            if len(pending_tasks) >= self.config.max_parallel_requests:
                await self._wait_one(
                    pending_tasks=pending_tasks,
                    write_buffer=write_buffer,
                )

        while pending_tasks:
            await self._wait_one(
                pending_tasks=pending_tasks,
                write_buffer=write_buffer,
            )

        self._flush_buffer(write_buffer)

        logger.info(
            "run.done",
            model=self.config.model,
            output_dir=str(self.output_dir),
        )
        return self.output_dir


async def main() -> None:
    jokes_path = DATA_DIR / config.jokes.data_filename
    if not jokes_path.exists():
        jokes_path = build_jokes_dataset()

    jokes = load_dataset("parquet", data_files=str(jokes_path), split="train[:10000]")
    pipeline = EmbeddingsPipeline()
    output_dir = await pipeline.run(jokes, resume=True)
    print(
        {
            "jokes_path": str(jokes_path),
            "embeddings_path": str(output_dir),
        }
    )

    # embeddings = load_dataset("parquet", data_dir=str(output_dir), split="train[:]")
    # print(embeddings[0])


if __name__ == "__main__":
    asyncio.run(main())
