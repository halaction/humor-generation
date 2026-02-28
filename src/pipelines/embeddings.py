import asyncio
from typing import Any, cast
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq
from openai import AsyncOpenAI
from pydantic import BaseModel
from tqdm.auto import tqdm

from datasets import Dataset, load_dataset
from src.config import EmbeddingsConfig, config
from src.datasets.jokes import build_jokes_dataset
from src.logging import get_logger
from src.paths import DATA_DIR
from src.settings import settings
from src.models import JokesBatch, EmbeddingsBatch, EmbeddingsOutputs

logger = get_logger(__name__)


class EmbeddingsPipeline:
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

        self.output_path = DATA_DIR / self.config.data_filename
        self.schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), self.config.dimensions)),
            ]
        )

    async def _embed_batch(
        self,
        batch: JokesBatch,
        semaphore: asyncio.Semaphore,
    ) -> EmbeddingsBatch:
        async with semaphore:
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    response = await self.client.embeddings.create(
                        model=self.config.model,
                        input=batch.text,
                        dimensions=self.config.dimensions,
                    )

                    outputs = EmbeddingsBatch(
                        id=batch.id,
                        embedding=response.data,  # type: ignore
                    )
                except Exception:
                    if attempt + 1 >= self.config.max_retries:
                        raise
                    await asyncio.sleep(2 ** (attempt - 1))
                else:
                    return outputs

        msg = "Unexpected retry error"
        raise RuntimeError(msg)

    def _flush_writer(
        self,
        writer: pq.ParquetWriter,
        write_buffer: list[EmbeddingsBatch],
    ) -> None:
        if not write_buffer:
            return

        outputs = defaultdict(list)
        for batch in write_buffer:
            for key, value in batch.model_dump().items():
                outputs[key].extend(value)

        table = pa.Table.from_pydict(outputs, schema=self.schema)
        writer.write_table(table)
        write_buffer.clear()

    async def _wait_one(
        self,
        pending_tasks: set[asyncio.Task[EmbeddingsBatch]],
        write_buffer: list[EmbeddingsBatch],
        writer: pq.ParquetWriter,
    ) -> None:
        done, _ = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            pending_tasks.remove(task)
            outputs = task.result()
            write_buffer.append(outputs)
            if len(write_buffer) * self.config.batch_size >= self.config.shard_size:
                self._flush_writer(writer, write_buffer)

    async def run(
        self,
        jokes: Dataset,
        resume: bool = False,
    ) -> EmbeddingsOutputs:

        writer = pq.ParquetWriter(
            where=str(self.output_path),
            schema=self.schema,
            compression="zstd",
            use_content_defined_chunking=True,
            write_page_index=True,
        )
        write_buffer: list[EmbeddingsBatch] = []

        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        pending_tasks: set[asyncio.Task[EmbeddingsBatch]] = set()

        dataset = jokes.batch(self.config.batch_size)

        for batch in tqdm(dataset, total=len(dataset)):
            batch = cast("dict[str, list[Any]]", batch)
            batch = JokesBatch(id=batch["id"], text=batch["text"])

            task = asyncio.create_task(self._embed_batch(batch, semaphore))
            pending_tasks.add(task)

            if len(pending_tasks) >= self.config.max_parallel_requests:
                await self._wait_one(
                    pending_tasks=pending_tasks,
                    write_buffer=write_buffer,
                    writer=writer,
                )

        while pending_tasks:
            await self._wait_one(
                pending_tasks=pending_tasks,
                write_buffer=write_buffer,
                writer=writer,
            )

        self._flush_writer(
            write_buffer=write_buffer,
            writer=writer,
        )

        writer.close()

        if not self.output_path.exists():
            msg = "No embedding outputs were written."
            raise ValueError(msg)

        logger.info(
            "run.done",
            model=self.config.model,
            output_path=str(self.output_path),
        )
        return EmbeddingsOutputs(output_path=self.output_path)


async def main() -> None:
    jokes_path = DATA_DIR / config.jokes.data_filename
    if not jokes_path.exists():
        jokes_path = build_jokes_dataset()

    inputs = load_dataset("parquet", data_files=str(jokes_path), split="train")
    pipeline = EmbeddingsPipeline()
    outputs = await pipeline.run(inputs)
    print(
        {
            "jokes_path": str(jokes_path),
            "embeddings_path": str(outputs.output_path),
        }
    )


if __name__ == "__main__":
    asyncio.run(main())
