import asyncio
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from openai import AsyncOpenAI
from tqdm.auto import tqdm

from datasets import Dataset, concatenate_datasets, load_dataset
from src.config import EvaluationConfig, config
from src.logging import get_logger
from src.models import EvaluationCandidate, EvaluationJudgeDecision, EvaluationOutputs, EvaluationPair
from src.paths import DATA_DIR
from src.pipelines.base import BasePipeline
from src.settings import settings
from src.templates import environment

logger = get_logger(__name__)


class EvaluationPipeline(BasePipeline):
    def __init__(
        self,
        pipeline_config: EvaluationConfig | None = None,
        output_dir: Path | None = None,
        client: AsyncOpenAI | None = None,
    ) -> None:
        self.config = pipeline_config or config.evaluation
        self.root_dir = output_dir or DATA_DIR / self.config.hf_config_name
        self.evaluation_dir = self.root_dir / "evaluation"
        self.leaderboard_dir = self.root_dir / "leaderboard"
        self.output_dir = self.evaluation_dir
        self.client = client or AsyncOpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=self.config.timeout,
        )
        self.next_part_index = 0

        self.prompt_template = environment.get_template("reference_prompt.j2")
        self.system_template = environment.get_template("evaluation_system.j2")
        self.user_template = environment.get_template("evaluation_user.j2")

        self.schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("reference_id", pa.int64()),
                pa.field("prompt", pa.string()),
                pa.field("left_model_id", pa.string()),
                pa.field("right_model_id", pa.string()),
                pa.field("left_model", pa.string()),
                pa.field("right_model", pa.string()),
                pa.field("left_text", pa.string()),
                pa.field("right_text", pa.string()),
                pa.field("winner", pa.string()),
            ]
        )
        self.leaderboard_schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("model_id", pa.string()),
                pa.field("model", pa.string()),
                pa.field("bt_score", pa.float64()),
                pa.field("wins", pa.float64()),
                pa.field("losses", pa.float64()),
                pa.field("n_comparisons", pa.int64()),
                pa.field("win_rate", pa.float64()),
            ]
        )
        self._evaluation_columns = [
            "id",
            "reference_id",
            "prompt",
            "left_model_id",
            "right_model_id",
            "left_model",
            "right_model",
            "left_text",
            "right_text",
            "winner",
        ]

    @staticmethod
    def _bt_scores(models: list[str], results: list[tuple[str, str, float, float]]) -> dict[str, float]:
        if not models:
            return {}

        index = {model: i for i, model in enumerate(models)}
        n_models = len(models)
        counts = np.zeros((n_models, n_models), dtype=np.float64)
        wins = np.zeros(n_models, dtype=np.float64)
        epsilon = 1e-12

        for left, right, left_score, right_score in results:
            i = index[left]
            j = index[right]
            counts[i, j] += 1.0
            counts[j, i] += 1.0
            wins[i] += left_score
            wins[j] += right_score

        strengths = np.ones(n_models, dtype=np.float64)
        for _ in range(200):
            updated = strengths.copy()
            for i in range(n_models):
                denominator = 0.0
                for j in range(n_models):
                    if i == j or counts[i, j] == 0.0:
                        continue
                    denominator += counts[i, j] / max(strengths[i] + strengths[j], epsilon)
                if denominator > 0.0:
                    updated[i] = max(wins[i] / denominator, epsilon)

            updated = updated / max(float(np.exp(np.mean(np.log(updated)))), epsilon)
            if float(np.max(np.abs(updated - strengths))) < 1e-9:
                strengths = updated
                break
            strengths = updated

        return {model: float(np.log(strengths[index[model]])) for model in models}

    def _get_table(self, write_buffer: list[EvaluationOutputs]) -> pa.Table:
        output = defaultdict(list)
        for batch in write_buffer:
            for key, value in batch.model_dump().items():
                output[key].extend(value)
        return pa.Table.from_pydict(output, schema=self.schema)

    def _check_buffer_size(self, write_buffer: list[EvaluationOutputs]) -> bool:
        return sum(len(batch.id) for batch in write_buffer) >= self.config.shard_size

    def _collect_candidates_per_reference(
        self,
        candidates: Dataset,
    ) -> dict[int, dict[str, list[EvaluationCandidate]]]:
        required_columns = {"id", "keywords", "model_id", "model", "text"}
        missing = required_columns - set(candidates.column_names)
        if missing:
            msg = f"Missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        frame = cast("pl.DataFrame", candidates.to_polars()).select(["id", "keywords", "model_id", "model", "text"])
        frame = frame.with_columns(
            pl.col("id").cast(pl.Int64, strict=True),
            pl.col("keywords").cast(pl.List(pl.String), strict=False),
            pl.col("model_id").cast(pl.String, strict=False).str.strip_chars(),
            pl.col("model").cast(pl.String, strict=False).str.strip_chars(),
            pl.col("text").cast(pl.String, strict=False).str.strip_chars(),
        )
        frame = frame.with_columns(pl.col("text").str.slice(0, self.config.max_response_chars).alias("text"))

        invalid_rows = frame.filter(
            pl.col("id").is_null()
            | pl.col("keywords").is_null()
            | pl.col("model_id").is_null()
            | pl.col("model").is_null()
            | pl.col("text").is_null()
            | (pl.col("model_id") == "")
            | (pl.col("model") == "")
            | (pl.col("text") == "")
        )
        if invalid_rows.height > 0:
            msg = "Candidates contain null or empty required values."
            raise ValueError(msg)

        frame = frame.with_columns(pl.col("keywords").list.eval(pl.element().str.strip_chars()))
        frame = frame.with_columns(
            pl.col("keywords")
            .list.eval(pl.element().filter(pl.element() != ""))
            .list.unique(maintain_order=True)
            .alias("keywords")
        )
        if frame.filter(pl.col("keywords").list.len() == 0).height > 0:
            msg = "Each candidate row must contain at least one keyword."
            raise ValueError(msg)

        inconsistent_keywords_count = (
            frame.group_by("id")
            .agg(pl.col("keywords").n_unique().alias("n_keywords"))
            .filter(pl.col("n_keywords") > 1)
            .height
        )
        if inconsistent_keywords_count > 0:
            msg = "Inconsistent keywords detected for one or more reference ids."
            raise ValueError(msg)

        frame = frame.sort(["id", "model_id"])
        candidates_per_reference: dict[int, dict[str, list[EvaluationCandidate]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for row in frame.iter_rows(named=True):
            keywords = cast("list[str]", row["keywords"])
            candidate = EvaluationCandidate(
                id=cast("int", row["id"]),
                keywords=keywords,
                model_id=cast("str", row["model_id"]),
                model=cast("str", row["model"]),
                text=cast("str", row["text"]),
            )
            candidates_per_reference[candidate.id][candidate.model_id].append(candidate)
        return dict(candidates_per_reference)

    def _build_pairs(
        self, candidates_per_reference: dict[int, dict[str, list[EvaluationCandidate]]]
    ) -> list[EvaluationPair]:
        rng = np.random.default_rng(self.config.random_seed)
        pairs: list[EvaluationPair] = []
        next_id = 0
        for reference_id in sorted(candidates_per_reference):
            groups = candidates_per_reference[reference_id]
            model_ids = sorted(groups)
            for left_model_id, right_model_id in combinations(model_ids, 2):
                left_group = groups[left_model_id]
                right_group = groups[right_model_id]
                for left_candidate in left_group:
                    for right_candidate in right_group:
                        if float(rng.random()) < 0.5:
                            first = left_candidate
                            second = right_candidate
                        else:
                            first = right_candidate
                            second = left_candidate
                        prompt = self.prompt_template.render(keywords=first.keywords).strip()
                        pairs.append(
                            EvaluationPair(
                                id=next_id,
                                reference_id=reference_id,
                                prompt=prompt[: self.config.max_prompt_chars],
                                left_model_id=first.model_id,
                                right_model_id=second.model_id,
                                left_model=first.model,
                                right_model=second.model,
                                left_text=first.text,
                                right_text=second.text,
                            )
                        )
                        next_id += 1
        return pairs

    def _read_evaluation_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.evaluation_dir.glob("part-*.parquet")):
            rows.extend(pq.read_table(path).to_pylist())
        return rows

    def _unlink_parts(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        for part in directory.glob("part-*.parquet"):
            part.unlink()

    def _write_rows_to_evaluation_parts(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        for start in range(0, len(rows), self.config.shard_size):
            chunk = rows[start : start + self.config.shard_size]
            table = pa.Table.from_pylist(chunk, schema=self.schema)
            path = self.evaluation_dir / f"part-{self.next_part_index:04d}.parquet"
            pq.write_table(
                table,
                where=str(path),
                compression="zstd",
                use_content_defined_chunking=True,
                write_page_index=True,
            )
            self.next_part_index += 1

    def _to_existing_rows_frame(self, rows: list[dict[str, Any]]) -> pl.DataFrame:
        return pl.DataFrame(rows).select(self._evaluation_columns)

    @staticmethod
    def _pairs_frame(pairs: list[EvaluationPair]) -> pl.DataFrame:
        if not pairs:
            return pl.DataFrame(
                {
                    "id": pl.Series([], dtype=pl.Int64),
                    "reference_id": pl.Series([], dtype=pl.Int64),
                    "left_model_id": pl.Series([], dtype=pl.String),
                    "right_model_id": pl.Series([], dtype=pl.String),
                }
            )
        return pl.DataFrame(
            {
                "id": [pair.id for pair in pairs],
                "reference_id": [pair.reference_id for pair in pairs],
                "left_model_id": [pair.left_model_id for pair in pairs],
                "right_model_id": [pair.right_model_id for pair in pairs],
            }
        )

    def _filter_rows_for_resume(
        self,
        existing_rows: list[dict[str, Any]],
        pairs: list[EvaluationPair],
    ) -> list[dict[str, Any]]:
        if not existing_rows or not pairs:
            return []

        existing_frame = self._to_existing_rows_frame(existing_rows)
        if existing_frame.is_empty():
            return []

        pair_frame = self._pairs_frame(pairs)
        valid_rows = existing_frame.join(
            pair_frame.select(["id", "reference_id", "left_model_id", "right_model_id"]),
            on=["id", "reference_id", "left_model_id", "right_model_id"],
            how="inner",
        )
        if valid_rows.is_empty():
            return []
        retained_rows = valid_rows.sort("id").to_dicts()
        return cast("list[dict[str, Any]]", retained_rows)

    async def _judge_pair(self, pair: EvaluationPair, semaphore: asyncio.Semaphore) -> EvaluationOutputs:
        system_prompt = self.system_template.render().strip()
        user_prompt = self.user_template.render(
            prompt=pair.prompt,
            left_text=pair.left_text,
            right_text=pair.right_text,
        ).strip()

        async with semaphore:
            for attempt in range(1, self.config.max_retries + 1):
                try:
                    completion = await self.client.chat.completions.parse(
                        model=self.config.model,
                        temperature=self.config.judge_temperature,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        response_format=EvaluationJudgeDecision,
                    )
                    message = completion.choices[0].message
                    parsed = message.parsed
                    if parsed is None:
                        msg = "No parsed response in completion message."
                        raise ValueError(msg)
                    return EvaluationOutputs(
                        id=[pair.id],
                        reference_id=[pair.reference_id],
                        prompt=[pair.prompt],
                        left_model_id=[pair.left_model_id],
                        right_model_id=[pair.right_model_id],
                        left_model=[pair.left_model],
                        right_model=[pair.right_model],
                        left_text=[pair.left_text],
                        right_text=[pair.right_text],
                        winner=[parsed.winner],
                    )
                except Exception:
                    if attempt >= self.config.max_retries:
                        raise
                    await asyncio.sleep(2 ** (attempt - 1))

        msg = "Unexpected judge retry branch."
        raise RuntimeError(msg)

    def _write_leaderboard(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            self._unlink_parts(self.leaderboard_dir)
            return

        frame = self._to_existing_rows_frame(rows)
        labels_frame = pl.concat(
            [
                frame.select(pl.col("left_model_id").alias("model_id"), pl.col("left_model").alias("model")),
                frame.select(pl.col("right_model_id").alias("model_id"), pl.col("right_model").alias("model")),
            ]
        )
        inconsistent_labels = (
            labels_frame.group_by("model_id")
            .agg(pl.col("model").n_unique().alias("n_labels"))
            .filter(pl.col("n_labels") > 1)
            .height
        )
        if inconsistent_labels > 0:
            msg = "Inconsistent model label for one or more model ids."
            raise ValueError(msg)

        labels = {
            cast("str", row["model_id"]): cast("str", row["model"])
            for row in labels_frame.unique(subset=["model_id"]).to_dicts()
        }
        wins: dict[str, float] = {}
        losses: dict[str, float] = {}
        comparisons: dict[str, int] = {}
        bt_input: list[tuple[str, str, float, float]] = []

        for row in frame.to_dicts():
            left_model_id = cast("str", row["left_model_id"])
            right_model_id = cast("str", row["right_model_id"])
            winner = cast("str", row["winner"])

            wins.setdefault(left_model_id, 0.0)
            wins.setdefault(right_model_id, 0.0)
            losses.setdefault(left_model_id, 0.0)
            losses.setdefault(right_model_id, 0.0)
            comparisons[left_model_id] = comparisons.get(left_model_id, 0) + 1
            comparisons[right_model_id] = comparisons.get(right_model_id, 0) + 1

            if winner == "left":
                wins[left_model_id] += 1.0
                losses[right_model_id] += 1.0
                bt_input.append((left_model_id, right_model_id, 1.0, 0.0))
            elif winner == "right":
                wins[right_model_id] += 1.0
                losses[left_model_id] += 1.0
                bt_input.append((left_model_id, right_model_id, 0.0, 1.0))
            else:
                msg = f"Unexpected winner value: {winner}"
                raise ValueError(msg)

        model_ids = sorted(set(wins))
        bt_scores = self._bt_scores(models=model_ids, results=bt_input)
        leaderboard_rows = []
        for idx, model_id in enumerate(model_ids):
            n = comparisons.get(model_id, 0)
            w = wins.get(model_id, 0.0)
            leaderboard_rows.append(
                {
                    "id": idx,
                    "model_id": model_id,
                    "model": labels[model_id],
                    "bt_score": bt_scores.get(model_id, 0.0),
                    "wins": w,
                    "losses": losses.get(model_id, 0.0),
                    "n_comparisons": n,
                    "win_rate": (w / n) if n > 0 else 0.0,
                }
            )

        self._unlink_parts(self.leaderboard_dir)
        table = pa.Table.from_pylist(leaderboard_rows, schema=self.leaderboard_schema)
        pq.write_table(table, self.leaderboard_dir / "part-0000.parquet", compression="zstd")

    async def run(self, candidates: Dataset, resume: bool = False) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        self.leaderboard_dir.mkdir(parents=True, exist_ok=True)

        candidates_per_reference = self._collect_candidates_per_reference(candidates)
        pairs = self._build_pairs(candidates_per_reference)
        existing_rows = self._read_evaluation_rows() if resume else []
        retained_rows = self._filter_rows_for_resume(existing_rows=existing_rows, pairs=pairs) if resume else []

        if not resume:
            self._unlink_parts(self.evaluation_dir)
            self.next_part_index = 0
        elif len(existing_rows) != len(retained_rows):
            self._unlink_parts(self.evaluation_dir)
            self.next_part_index = 0
            self._write_rows_to_evaluation_parts(retained_rows)
        else:
            self.next_part_index = self._get_next_part_index()

        seen_ids = {cast("int", row["id"]) for row in retained_rows}
        write_buffer: list[EvaluationOutputs] = []
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        pending_tasks: set[asyncio.Task[EvaluationOutputs]] = set()

        for pair in tqdm(pairs, desc="Evaluate pairs"):
            if pair.id in seen_ids:
                continue
            pending_tasks.add(asyncio.create_task(self._judge_pair(pair=pair, semaphore=semaphore)))
            if len(pending_tasks) >= self.config.max_parallel_requests:
                await self._wait_one(
                    pending_tasks=cast("set[asyncio.Task[EvaluationOutputs | None]]", pending_tasks),
                    write_buffer=write_buffer,
                )

        while pending_tasks:
            await self._wait_one(
                pending_tasks=cast("set[asyncio.Task[EvaluationOutputs | None]]", pending_tasks),
                write_buffer=write_buffer,
            )

        self._flush_buffer(write_buffer)
        all_rows = self._read_evaluation_rows()
        self._write_leaderboard(rows=all_rows)
        logger.info(
            "run.done",
            output_dir=str(self.root_dir),
            pair_count=len(all_rows),
            model_count=len({str(r["left_model_id"]) for r in all_rows} | {str(r["right_model_id"]) for r in all_rows}),
        )
        return self.root_dir

    def build(
        self,
        *,
        candidate_paths: list[Path],
        split: str = "train",
        resume: bool = True,
    ) -> Path:
        datasets = []
        for path in candidate_paths:
            dataset = load_dataset(
                "parquet",
                data_files={split: str(path / split / "part-*.parquet")},
                split=split,
            )
            datasets.append(dataset)
        candidates = concatenate_datasets(datasets)
        output_dir = asyncio.run(self.run(candidates=candidates, resume=resume))
        logger.info("build.done", output_dir=str(output_dir))
        return output_dir


def main() -> None:
    pipeline = EvaluationPipeline()
    candidates_root = DATA_DIR / config.candidates.hf_config_name
    model_dirs = [path for path in candidates_root.iterdir() if path.is_dir()] if candidates_root.exists() else []
    if not model_dirs:
        msg = f"Expected per-model candidate directories under {candidates_root}"
        raise FileNotFoundError(msg)
    output_dir = pipeline.build(candidate_paths=model_dirs, split="test", resume=False)
    print({"evaluation_dir": str(output_dir)})


if __name__ == "__main__":
    main()
