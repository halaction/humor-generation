import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from src.config import EmbeddingsConfig, config
from src.logging import get_logger
from src.models import EmbeddingsInputs, EmbeddingsOutputs
from src.paths import DATA_DIR
from src.pipelines.base import BasePipeline
from src.pipelines.jokes import JokesPipeline
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
                pa.field("id", pa.int64()),
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
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.next_part_index = self._get_next_part_index()

        dataset = self._check_progress(jokes, resume)

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

    def build(
        self,
        split: str = "train",
        resume: bool = True,
    ) -> None:
        jokes_dir = DATA_DIR / config.jokes.hf_config_name
        if not jokes_dir.exists():
            JokesPipeline().build()

        jokes = load_dataset("parquet", data_dir=str(jokes_dir), split=split)
        asyncio.run(self.run(jokes=jokes, resume=resume))
        logger.info(
            "build.done",
            jokes_dir=str(jokes_dir),
            output_dir=str(self.output_dir),
        )

    def publish(
        self,
        repo_id: str = settings.HF_DATASET_REPO_ID,
        config_name: str = config.embeddings.hf_config_name,
        split: str = "train",
        private: bool = False,
    ) -> None:
        if not self.output_dir.exists():
            self.build()

        dataset = load_dataset("parquet", data_dir=str(self.output_dir), split=split)
        api = HfApi(token=settings.HF_TOKEN)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        dataset.push_to_hub(
            repo_id=repo_id,
            config_name=config_name,
            token=settings.HF_TOKEN,
            private=private,
        )
        logger.info(
            "publish.done",
            repo_id=repo_id,
            config_name=config_name,
            output_dir=str(self.output_dir),
            split=split,
        )


def main() -> None:
    pipeline = EmbeddingsPipeline()
    pipeline.build(split="train[:10005]", resume=True)
    pipeline.publish()

    logger.info(
        "main.done",
        embeddings_dir=str(pipeline.output_dir),
        repo_id=settings.HF_DATASET_REPO_ID,
        config_name=pipeline.config.hf_config_name,
    )


if __name__ == "__main__":
    main()
