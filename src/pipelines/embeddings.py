import asyncio
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
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
        *,
        pipeline_config: EmbeddingsConfig | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.config = pipeline_config or config.embeddings
        self.client = client or AsyncOpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=self.config.timeout,
        )

        self.output_dir = DATA_DIR / self.config.hf_config_name
        self.next_part_index = 0
        self.schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), self.config.dimensions)),
            ]
        )

    def _get_next_part_index(self) -> int:
        pattern = re.compile(r"part-(\d+)\.parquet")
        indices = []
        for file in self.output_dir.iterdir():
            match = pattern.search(file.name)
            if match:
                indices.append(int(match.group(1)))

        return max(indices) + 1 if indices else 0

    def _get_seen_ids(self) -> set[int]:
        dataset = ds.dataset(self.output_dir, format="parquet")
        array = dataset.to_table(columns=["id"]).column("id").to_numpy()
        seen_ids = np.unique(array).tolist()

        return set(seen_ids)

    async def _embed_jokes(
        self,
        batch: EmbeddingsInputs,
        semaphore: asyncio.Semaphore,
    ) -> EmbeddingsOutputs:
        async with semaphore:
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    response = await self.client.embeddings.create(
                        model=self.config.model,
                        input=batch.text,
                        dimensions=self.config.dimensions,
                    )

                    outputs = EmbeddingsOutputs(
                        id=batch.id,
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

    def _flush_buffer(
        self,
        write_buffer: list[EmbeddingsOutputs],
    ) -> None:
        if not write_buffer:
            return

        outputs = defaultdict(list)
        for batch in write_buffer:
            for key, value in batch.model_dump().items():
                outputs[key].extend(value)

        table = pa.Table.from_pydict(outputs, schema=self.schema)
        path = self.output_dir / f"part-{self.next_part_index:04d}.parquet"
        pq.write_table(
            table,
            where=str(path),
            compression="zstd",
            use_content_defined_chunking=True,
            write_page_index=True,
        )
        self.next_part_index += 1

        write_buffer.clear()

    async def _wait_one(
        self,
        pending_tasks: set[asyncio.Task[EmbeddingsOutputs]],
        write_buffer: list[EmbeddingsOutputs],
    ) -> None:
        done, _ = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            pending_tasks.remove(task)
            outputs = task.result()
            write_buffer.append(outputs)
            if len(write_buffer) * self.config.batch_size >= self.config.shard_size:
                self._flush_buffer(write_buffer)

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
        elif self.next_part_index == 0:
            for file in self.output_dir.iterdir():
                file.unlink()

        dataset = dataset.batch(self.config.batch_size)

        write_buffer: list[EmbeddingsOutputs] = []
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        pending_tasks: set[asyncio.Task[EmbeddingsOutputs]] = set()

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

    jokes = load_dataset("parquet", data_files=str(jokes_path), split="train[:50]")
    pipeline = EmbeddingsPipeline()
    output_dir = await pipeline.run(jokes, resume=True)
    print(
        {
            "jokes_path": str(jokes_path),
            "embeddings_dir": str(output_dir),
        }
    )

    embeddings = load_dataset("parquet", data_dir=str(output_dir), split="train[:]")
    print(embeddings[0])


if __name__ == "__main__":
    asyncio.run(main())
