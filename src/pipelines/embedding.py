import asyncio
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from openai import AsyncOpenAI
from pydantic import BaseModel

from datasets import Dataset
from src.logging import get_logger
from src.paths import EMBEDDING_CONFIG_PATH, EMBEDDINGS_DATA_PATH
from src.settings import settings

logger = get_logger(__name__)


class EmbeddingConfig(BaseModel):
    model: str
    embedding_dim: int
    batch_size: int
    shard_size: int
    max_parallel_requests: int
    timeout: int
    max_retries: int


class JokesItem(BaseModel):
    id: str
    text: str


class EmbeddingsItem(BaseModel):
    id: str
    embedding: list[float]


EmbeddingInputs = Dataset


class EmbeddingOutputs(BaseModel):
    data_path: Path


def _build_table(outputs: list[EmbeddingsItem], embedding_dim: int) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source_name", pa.string()),
            pa.field("source_filename", pa.string()),
            pa.field("source_id", pa.string()),
            pa.field("embedding", pa.list_(pa.float32(), embedding_dim)),
        ]
    )
    payload = [item.model_dump(mode="python") for item in outputs]
    return pa.Table.from_pylist(payload, schema=schema)


class EmbeddingPipeline:
    def __init__(self, config_path: Path = EMBEDDING_CONFIG_PATH) -> None:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        self.config = EmbeddingConfig.model_validate(payload)
        self.client = AsyncOpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=self.config.timeout,
        )

    async def _embed_batch(
        self,
        batch: list[JokesItem],
        semaphore: asyncio.Semaphore,
    ) -> list[EmbeddingsItem]:
        async with semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    response = await self.client.embeddings.create(
                        model=self.config.model,
                        input=[item.text for item in batch],
                    )

                    outputs: list[EmbeddingsItem] = []
                    for input_item, output_item in zip(batch, response.data, strict=True):
                        outputs.append(
                            EmbeddingsItem(
                                id=input_item.id,
                                embedding=output_item.embedding,
                            )
                        )
                except Exception:
                    if attempt + 1 >= self.config.max_retries:
                        raise
                    await asyncio.sleep(2**attempt)
                else:
                    return outputs

        raise RuntimeError("Unexpected embedding retry flow")

    def _flush_writer(
        self,
        write_buffer: list[EmbeddingsItem],
        writer: pq.ParquetWriter,
    ) -> None:
        if not write_buffer:
            return

        for item in write_buffer:
            if len(item.embedding) != self.config.embedding_dim:
                raise ValueError(
                    f"Inconsistent embedding size for id={item.id}: expected {self.config.embedding_dim}, "
                    f"got {len(item.embedding)}"
                )

        table = _build_table(outputs=write_buffer, embedding_dim=self.config.embedding_dim)
        writer.write_table(table)
        write_buffer.clear()

    async def _wait_one(
        self,
        pending_tasks: set[asyncio.Task[list[EmbeddingsItem]]],
        write_buffer: list[EmbeddingsItem],
        writer: pq.ParquetWriter,
    ) -> None:
        done, _ = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            pending_tasks.remove(task)
            outputs = task.result()
            write_buffer.extend(outputs)
            if len(write_buffer) >= self.config.shard_size:
                self._flush_writer(
                    write_buffer=write_buffer,
                    writer=writer,
                )

    async def run(
        self,
        inputs: EmbeddingInputs,
        output_path: Path = EMBEDDINGS_DATA_PATH,
    ) -> EmbeddingOutputs:
        if "id" not in inputs.column_names:
            raise ValueError("Dataset must contain an 'id' column.")
        if "text" not in inputs.column_names:
            raise ValueError("Dataset must contain a 'text' column.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            output_path.unlink()

        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)

        pending_tasks: set[asyncio.Task[list[EmbeddingsItem]]] = set()
        request_batch: list[JokesItem] = []
        write_buffer: list[EmbeddingsItem] = []
        schema = pa.schema(
            [
                pa.field("id", pa.string()),
                pa.field("text", pa.string()),
                pa.field("source_name", pa.string()),
                pa.field("source_filename", pa.string()),
                pa.field("source_id", pa.string()),
                pa.field("embedding", pa.list_(pa.float32(), self.config.embedding_dim)),
            ]
        )
        writer = pq.ParquetWriter(
            where=str(output_path),
            schema=schema,
            compression="zstd",
            use_content_defined_chunking=True,
            write_page_index=True,
        )

        for item in inputs:
            request_batch.append(
                JokesItem(
                    id=item["id"],
                    text=item["text"],
                )
            )
            if len(request_batch) >= self.config.batch_size:
                pending_tasks.add(asyncio.create_task(self._embed_batch(request_batch, semaphore)))
                request_batch = []
                if len(pending_tasks) >= self.config.max_parallel_requests:
                    await self._wait_one(
                        pending_tasks=pending_tasks,
                        write_buffer=write_buffer,
                        writer=writer,
                    )

        if request_batch:
            pending_tasks.add(asyncio.create_task(self._embed_batch(request_batch, semaphore)))

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

        if not output_path.exists():
            raise ValueError("No embedding outputs were written.")

        logger.info(
            "run.done",
            model=self.config.model,
            output_path=str(output_path),
        )
        return EmbeddingOutputs(data_path=output_path)
