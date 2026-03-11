import asyncio
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Generic, ParamSpec, TypeVar

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from openai import AsyncOpenAI
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)
P = ParamSpec("P")


class BasePipeline(ABC, Generic[P, T]):
    def __init__(
        self,
        pipeline_config: T,
        output_dir: Path,
        client: AsyncOpenAI,
    ) -> None:
        self.config = pipeline_config
        self.output_dir = output_dir
        self.client = client
        self.next_part_index = 0

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
        if not dataset.files:
            return set()

        array = np.asarray(dataset.to_table(columns=["id"]).column("id").to_numpy(), dtype=np.int64)
        seen_ids = np.unique(array).tolist()
        return {int(item) for item in seen_ids}

    def _get_table(self, write_buffer: list[T]) -> pa.Table:
        raise NotImplementedError

    def _flush_buffer(
        self,
        write_buffer: list[T],
    ) -> None:
        if not write_buffer:
            return

        table = self._get_table(write_buffer)
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

    def _check_buffer_size(self, write_buffer: list[T]) -> bool:
        raise NotImplementedError

    async def _wait_one(
        self,
        pending_tasks: set[asyncio.Task[T | None]],
        write_buffer: list[T],
    ) -> None:
        done, _ = await asyncio.wait(pending_tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            pending_tasks.remove(task)
            outputs = task.result()
            if outputs is None:
                continue
            write_buffer.append(outputs)
            if self._check_buffer_size(write_buffer):
                self._flush_buffer(write_buffer)

    @abstractmethod
    async def run(self, *args: P.args, **kwargs: P.kwargs) -> Path:
        raise NotImplementedError
