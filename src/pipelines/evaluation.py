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

from datasets import Dataset, load_dataset
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

        self.system_template = environment.get_template("evaluation_system.j2")
        self.user_template = environment.get_template("evaluation_user.j2")

        self.schema = pa.schema(
            [
                pa.field("id", pa.int64()),
                pa.field("prompt_id", pa.string()),
                pa.field("prompt", pa.string()),
                pa.field("left_candidate_id", pa.int64()),
                pa.field("right_candidate_id", pa.int64()),
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
            "prompt_id",
            "prompt",
            "left_candidate_id",
            "right_candidate_id",
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

    def _collect_candidates_per_prompt(
        self,
        candidates: Dataset,
    ) -> dict[str, dict[str, list[EvaluationCandidate]]]:
        required_columns = {"id", "prompt_id", "prompt", "model_id", "model", "text"}
        missing = required_columns - set(candidates.column_names)
        if missing:
            msg = f"Missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        frame = cast("pl.DataFrame", candidates.to_polars()).select(
            ["id", "prompt_id", "prompt", "model_id", "model", "text"]
        )
        frame = frame.with_columns(
            pl.col("id").cast(pl.Int64, strict=True),
            pl.col("prompt_id").cast(pl.String, strict=False).str.strip_chars(),
            pl.col("prompt").cast(pl.String, strict=False).str.strip_chars(),
            pl.col("model_id").cast(pl.String, strict=False).str.strip_chars(),
            pl.col("model").cast(pl.String, strict=False).str.strip_chars(),
            pl.col("text").cast(pl.String, strict=False).str.strip_chars(),
        ).with_columns(
            pl.col("prompt").str.slice(0, self.config.max_prompt_chars),
            pl.col("text").str.slice(0, self.config.max_response_chars),
        )

        invalid_rows = frame.filter(
            pl.col("id").is_null()
            | pl.col("prompt_id").is_null()
            | pl.col("prompt").is_null()
            | pl.col("model_id").is_null()
            | pl.col("model").is_null()
            | pl.col("text").is_null()
            | (pl.col("prompt_id") == "")
            | (pl.col("prompt") == "")
            | (pl.col("model_id") == "")
            | (pl.col("model") == "")
            | (pl.col("text") == "")
        )
        if invalid_rows.height > 0:
            msg = "Candidates contain null or empty required values."
            raise ValueError(msg)

        duplicate_id_count = frame.filter(pl.col("id").is_duplicated()).height
        if duplicate_id_count > 0:
            msg = "Duplicate candidate id detected."
            raise ValueError(msg)

        inconsistent_prompt_count = (
            frame.group_by("prompt_id")
            .agg(pl.col("prompt").n_unique().alias("n_prompts"))
            .filter(pl.col("n_prompts") > 1)
            .height
        )
        if inconsistent_prompt_count > 0:
            msg = "Inconsistent prompt text detected for one or more prompt ids."
            raise ValueError(msg)

        frame = frame.sort(["prompt_id", "model_id", "id"])

        candidates_per_prompt: dict[str, dict[str, list[EvaluationCandidate]]] = defaultdict(lambda: defaultdict(list))
        for row in frame.iter_rows(named=True):
            candidate = EvaluationCandidate(
                id=cast("int", row["id"]),
                prompt_id=cast("str", row["prompt_id"]),
                prompt=cast("str", row["prompt"]),
                model_id=cast("str", row["model_id"]),
                model=cast("str", row["model"]),
                text=cast("str", row["text"]),
            )
            candidates_per_prompt[candidate.prompt_id][candidate.model_id].append(candidate)

        return dict(candidates_per_prompt)

    def _build_pairs(
        self, candidates_per_prompt: dict[str, dict[str, list[EvaluationCandidate]]]
    ) -> list[EvaluationPair]:
        rng = np.random.default_rng(self.config.random_seed)
        pairs: list[EvaluationPair] = []
        next_id = 0
        for prompt_id in sorted(candidates_per_prompt):
            groups = candidates_per_prompt[prompt_id]
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

                        pairs.append(
                            EvaluationPair(
                                id=next_id,
                                prompt_id=prompt_id,
                                prompt=first.prompt,
                                left_candidate_id=first.id,
                                right_candidate_id=second.id,
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
                    "left_candidate_id": pl.Series([], dtype=pl.Int64),
                    "right_candidate_id": pl.Series([], dtype=pl.Int64),
                    "left_model_id": pl.Series([], dtype=pl.String),
                    "right_model_id": pl.Series([], dtype=pl.String),
                }
            )

        return pl.DataFrame(
            {
                "id": [pair.id for pair in pairs],
                "left_candidate_id": [pair.left_candidate_id for pair in pairs],
                "right_candidate_id": [pair.right_candidate_id for pair in pairs],
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
            pair_frame.select(["id", "left_candidate_id", "right_candidate_id"]),
            on=["id", "left_candidate_id", "right_candidate_id"],
            how="inner",
        )
        if valid_rows.is_empty():
            return []

        expected_neighbors = pl.concat(
            [
                pair_frame.select(
                    pl.col("left_candidate_id").alias("candidate_id"),
                    pl.col("right_candidate_id").alias("neighbor_id"),
                ),
                pair_frame.select(
                    pl.col("right_candidate_id").alias("candidate_id"),
                    pl.col("left_candidate_id").alias("neighbor_id"),
                ),
            ]
        ).unique(subset=["candidate_id", "neighbor_id"])
        expected_counts = expected_neighbors.group_by("candidate_id").len(name="expected_neighbor_count")

        observed_neighbors = pl.concat(
            [
                valid_rows.select(
                    pl.col("left_candidate_id").alias("candidate_id"),
                    pl.col("right_candidate_id").alias("neighbor_id"),
                ),
                valid_rows.select(
                    pl.col("right_candidate_id").alias("candidate_id"),
                    pl.col("left_candidate_id").alias("neighbor_id"),
                ),
            ]
        ).unique(subset=["candidate_id", "neighbor_id"])
        observed_counts = observed_neighbors.group_by("candidate_id").len(name="observed_neighbor_count")

        completed_candidate_ids = (
            expected_counts.join(observed_counts, on="candidate_id", how="left")
            .with_columns(pl.col("observed_neighbor_count").fill_null(0))
            .filter(pl.col("observed_neighbor_count") >= pl.col("expected_neighbor_count"))
            .get_column("candidate_id")
            .to_list()
        )
        if not completed_candidate_ids:
            return []

        retained_rows = (
            valid_rows.filter(
                pl.col("left_candidate_id").is_in(completed_candidate_ids)
                & pl.col("right_candidate_id").is_in(completed_candidate_ids)
            )
            .sort("id")
            .to_dicts()
        )
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
                        prompt_id=[pair.prompt_id],
                        prompt=[pair.prompt],
                        left_candidate_id=[pair.left_candidate_id],
                        right_candidate_id=[pair.right_candidate_id],
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
                frame.select(
                    pl.col("left_model_id").alias("model_id"),
                    pl.col("left_model").alias("model"),
                ),
                frame.select(
                    pl.col("right_model_id").alias("model_id"),
                    pl.col("right_model").alias("model"),
                ),
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

        candidates_per_prompt = self._collect_candidates_per_prompt(candidates)
        pairs = self._build_pairs(candidates_per_prompt)
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
        candidates_path: Path,
        split: str = "train",
        resume: bool = True,
    ) -> Path:
        candidates = load_dataset("parquet", data_files=str(candidates_path), split=split)
        output_dir = asyncio.run(self.run(candidates=candidates, resume=resume))
        logger.info("build.done", output_dir=str(output_dir))
        return output_dir


def main() -> None:
    pipeline = EvaluationPipeline()
    candidates_path = DATA_DIR / "evaluation_candidates.parquet"
    if not candidates_path.exists():
        msg = f"Expected candidates parquet at {candidates_path}"
        raise FileNotFoundError(msg)
    output_dir = pipeline.build(candidates_path=candidates_path, resume=False)
    print({"evaluation_dir": str(output_dir)})


if __name__ == "__main__":
    main()
