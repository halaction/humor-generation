import argparse
import asyncio
from collections import defaultdict
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from src.config import CandidatesConfig, config
from src.logging import get_logger
from src.models import CandidateOutput
from src.paths import DATA_DIR
from src.pipelines.base import BasePipeline
from src.pipelines.references import ReferencesPipeline
from src.settings import settings
from src.templates import environment

logger = get_logger(__name__)


class CandidatesPipeline(BasePipeline):
    def __init__(
        self,
        pipeline_config: CandidatesConfig | None = None,
        output_dir: Path | None = None,
        client: AsyncOpenAI | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.config = pipeline_config or config.candidates
        self.root_dir = output_dir or DATA_DIR / self.config.hf_config_name
        self.output_dir = self.root_dir
        self.client = client or AsyncOpenAI(
            base_url=base_url or settings.OPENAI_BASE_URL,
            api_key=api_key or settings.OPENAI_API_KEY,
            timeout=self.config.timeout,
        )
        self.next_part_index = 0
        self.prompt_template = environment.get_template("reference_prompt.j2")
        self.schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("keywords", pa.list_(pa.string())),
                pa.field("model", pa.string()),
                pa.field("text", pa.string()),
            ]
        )

    def _get_table(self, write_buffer: list[CandidateOutput]) -> pa.Table:
        output = defaultdict(list)
        for row in write_buffer:
            for key, value in row.model_dump().items():
                output[key].append(value)
        return pa.Table.from_pydict(output, schema=self.schema)

    def _check_buffer_size(self, write_buffer: list[CandidateOutput]) -> bool:
        return len(write_buffer) >= self.config.shard_size

    async def _generate_candidate(
        self,
        row_id: int,
        keywords: list[str],
        model: str,
        semaphore: asyncio.Semaphore,
    ) -> CandidateOutput:
        prompt = self.prompt_template.render(keywords=keywords).strip()
        async with semaphore:
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    completion = await self.client.chat.completions.create(
                        model=model,
                        temperature=self.config.temperature,
                        max_completion_tokens=self.config.max_completion_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    message = completion.choices[0].message
                    text = message.content or ""
                    return CandidateOutput(
                        id=row_id,
                        keywords=keywords,
                        model=model,
                        text=text,
                    )
                except Exception:
                    if attempt >= self.config.max_retries:
                        raise
                    await asyncio.sleep(2 ** (attempt - 1))

        msg = "Unexpected generation retry error."
        raise RuntimeError(msg)

    async def run(
        self,
        references: Dataset,
        model: str,
        resume: bool = False,
    ) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.next_part_index = self._get_next_part_index()

        dataset = self._check_progress(references, resume)

        write_buffer: list[CandidateOutput] = []
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        pending_tasks: set[asyncio.Task[CandidateOutput]] = set()

        for row in tqdm(dataset, desc="Generate candidates"):
            row = cast("dict[str, Any]", row)
            pending_tasks.add(
                asyncio.create_task(
                    self._generate_candidate(
                        row_id=cast("int", row["id"]),
                        keywords=cast("list[str]", row["keywords"]),
                        model=model,
                        semaphore=semaphore,
                    )
                )
            )
            if len(pending_tasks) >= self.config.max_parallel_requests:
                await self._wait_one(
                    pending_tasks=cast("set[asyncio.Task[CandidateOutput | None]]", pending_tasks),
                    write_buffer=cast("list[CandidateOutput]", write_buffer),
                )

        while pending_tasks:
            await self._wait_one(
                pending_tasks=cast("set[asyncio.Task[CandidateOutput | None]]", pending_tasks),
                write_buffer=cast("list[CandidateOutput]", write_buffer),
            )

        self._flush_buffer(write_buffer)

        logger.info(
            "run.done",
            output_dir=str(self.output_dir),
            model=model,
        )

    def build(
        self,
        model: str,
        split: str,
        resume: bool = True,
    ) -> None:
        references_dir = DATA_DIR / config.references.hf_config_name
        if not references_dir.exists():
            ReferencesPipeline().build()

        references = load_dataset(
            "parquet",
            data_dir=str(references_dir),
            split=split,
        )
        self.output_dir = self.root_dir / model / split
        asyncio.run(
            self.run(
                references=references,
                model=model,
                resume=resume,
            )
        )
        logger.info("build.done", output_dir=str(self.output_dir))

    def publish(
        self,
        repo_id: str = settings.HF_DATASET_REPO_ID,
        config_name: str | None = None,
        model: str = "baseline",
        split: str = "test",
        private: bool = False,
    ) -> None:
        config_name = f"{config_name}/{model}"
        split_dir = self.root_dir / model / split

        has_split_data = split_dir.exists() and any(split_dir.glob("part-*.parquet"))
        if not has_split_data:
            msg = f"Expected candidate parquet files for split={split!r} under {split_dir}"
            raise FileNotFoundError(msg)

        data_files = {split: str(split_dir / "part-*.parquet")}

        dataset = load_dataset("parquet", data_files=data_files, split=split)
        api = HfApi(token=settings.HF_TOKEN)
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        dataset.push_to_hub(
            repo_id=repo_id,
            config_name=config_name,
            token=settings.HF_TOKEN,
            private=private,
            split=split,
        )
        logger.info(
            "publish.done",
            repo_id=repo_id,
            config_name=config_name,
            model=model,
            split=split,
            output_dir=str(self.output_dir),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run candidate generation.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    pipeline = CandidatesPipeline()
    pipeline.build(
        model=args.model,
        split=args.split,
        resume=args.resume,
    )
    pipeline.publish(model=args.model, split=args.split)

    logger.info(
        "main.done",
        candidates_dir=str(pipeline.output_dir),
        repo_id=settings.HF_DATASET_REPO_ID,
        config_name=pipeline.config.hf_config_name,
        model=args.model,
        split=args.split,
    )


if __name__ == "__main__":
    main()
