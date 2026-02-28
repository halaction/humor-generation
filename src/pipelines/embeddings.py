import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
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

        self.output_path = DATA_DIR / self.config.data_filename
        self.schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), self.config.dimensions)),
            ]
        )

    def _get_seen_ids(self) -> set[int]:

        if not self.output_path.exists():
            return set()

        array = pq.read_table(self.output_path, columns=["id"]).column("id").to_numpy()
        seen_ids = array.unique().to_list()

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

    def _flush_writer(
        self,
        writer: pq.ParquetWriter,
        write_buffer: list[EmbeddingsOutputs],
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
        pending_tasks: set[asyncio.Task[EmbeddingsOutputs]],
        write_buffer: list[EmbeddingsOutputs],
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
    ) -> Path:

        writer = pq.ParquetWriter(
            where=str(self.output_path),
            schema=self.schema,
            compression="zstd",
            use_content_defined_chunking=True,
            write_page_index=True,
        )
        write_buffer: list[EmbeddingsOutputs] = []

        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        pending_tasks: set[asyncio.Task[EmbeddingsOutputs]] = set()

        dataset = jokes

        if resume:
            seen_ids = self._get_seen_ids()
            dataset = dataset.filter(lambda item: item["id"] in seen_ids)
        elif not self.output_path.exists():
            self.output_path.unlink()

        dataset = dataset.batch(self.config.batch_size)

        for batch in tqdm(dataset):
            batch = cast("dict[str, list[Any]]", batch)
            inputs = EmbeddingsInputs(id=batch["id"], text=batch["text"])

            task = asyncio.create_task(self._embed_jokes(inputs, semaphore))
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
        return self.output_path


async def main() -> None:
    jokes_path = DATA_DIR / config.jokes.data_filename
    if not jokes_path.exists():
        jokes_path = build_jokes_dataset()

    jokes = load_dataset("parquet", data_files=str(jokes_path), split="train[:30]")
    pipeline = EmbeddingsPipeline()
    output_path = await pipeline.run(jokes, resume=False)
    print(
        {
            "jokes_path": str(jokes_path),
            "embeddings_path": str(output_path),
        }
    )

    embeddings = load_dataset("parquet", data_files=str(output_path), split="train[:30]")
    print(embeddings)


if __name__ == "__main__":
    asyncio.run(main())
