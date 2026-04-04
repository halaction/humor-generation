import asyncio
from itertools import combinations
from pathlib import Path
from typing import Any, cast

import numpy as np
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
        self.output_dir = self.root_dir / "parts"
        self.client = client or AsyncOpenAI(
            base_url=settings.OPENAI_BASE_URL,
            api_key=settings.OPENAI_API_KEY,
            timeout=self.config.timeout,
        )
        self.next_part_index = 0

        self.evaluation_path = self.root_dir / self.config.data_filename
        self.leaderboard_path = self.root_dir / self.config.leaderboard_filename
        self.system_template = environment.get_template("evaluation_system.j2")
        self.user_template = environment.get_template("evaluation_user.j2")

        self.schema = pa.schema(
            [
                pa.field("prompt_id", pa.string()),
                pa.field("prompt", pa.string()),
                pa.field("left_model", pa.string()),
                pa.field("right_model", pa.string()),
                pa.field("left_text", pa.string()),
                pa.field("right_text", pa.string()),
                pa.field("winner", pa.string()),
            ]
        )

    @staticmethod
    def _clean(value: Any) -> str:
        return str(value).strip()

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
        output: dict[str, list[Any]] = {field.name: [] for field in self.schema}
        for batch in write_buffer:
            output["prompt_id"].extend(batch.prompt_id)
            output["prompt"].extend(batch.prompt)
            output["left_model"].extend(batch.left_model)
            output["right_model"].extend(batch.right_model)
            output["left_text"].extend(batch.left_text)
            output["right_text"].extend(batch.right_text)
            output["winner"].extend(batch.winner)
        return pa.Table.from_pydict(output, schema=self.schema)

    def _check_buffer_size(self, write_buffer: list[EvaluationOutputs]) -> bool:
        return sum(len(batch.prompt_id) for batch in write_buffer) >= self.config.shard_size

    def _resolve_columns(self, columns: list[str]) -> tuple[str, str]:
        model_column = "model" if "model" in columns else "system_id"
        text_column = "text" if "text" in columns else "response_text"
        if model_column not in columns:
            msg = "Candidates dataset must include `model` or `system_id`."
            raise ValueError(msg)
        if text_column not in columns:
            msg = "Candidates dataset must include `text` or `response_text`."
            raise ValueError(msg)
        return model_column, text_column

    def _collect_candidates(self, candidates: Dataset) -> dict[str, list[EvaluationCandidate]]:
        required_columns = {"prompt_id", "prompt"}
        missing = required_columns - set(candidates.column_names)
        if missing:
            msg = f"Missing required columns: {sorted(missing)}"
            raise ValueError(msg)

        model_column, text_column = self._resolve_columns(candidates.column_names)
        per_prompt: dict[str, list[EvaluationCandidate]] = {}
        seen: set[tuple[str, str]] = set()

        for batch in tqdm(candidates.iter(batch_size=self.config.input_batch_size), desc="Collect candidates"):
            typed_batch = cast("dict[str, list[Any]]", batch)
            size = len(typed_batch["prompt_id"])
            for i in range(size):
                prompt_id = self._clean(typed_batch["prompt_id"][i])
                prompt = self._clean(typed_batch["prompt"][i])[: self.config.max_prompt_chars]
                model = self._clean(typed_batch[model_column][i])
                text = self._clean(typed_batch[text_column][i])[: self.config.max_response_chars]
                if not prompt_id or not prompt or not model or not text:
                    msg = (
                        "Empty values are not allowed in prompt_id/prompt/model/text. "
                        f"prompt_id={prompt_id!r} model={model!r}"
                    )
                    raise ValueError(msg)

                key = (prompt_id, model)
                if key in seen:
                    msg = f"Duplicate row for (prompt_id, model): {key}"
                    raise ValueError(msg)
                seen.add(key)

                candidate = EvaluationCandidate(prompt_id=prompt_id, prompt=prompt, model=model, text=text)
                per_prompt.setdefault(prompt_id, []).append(candidate)

        return per_prompt

    def _build_pairs(self, per_prompt: dict[str, list[EvaluationCandidate]]) -> list[EvaluationPair]:
        rng = np.random.default_rng(self.config.random_seed)
        pairs: list[EvaluationPair] = []
        for prompt_id in sorted(per_prompt):
            rows = sorted(per_prompt[prompt_id], key=lambda item: item.model)
            if len(rows) < 2:
                continue
            prompt = rows[0].prompt
            for first, second in combinations(rows, 2):
                if float(rng.random()) < 0.5:
                    left = first
                    right = second
                else:
                    left = second
                    right = first
                pairs.append(
                    EvaluationPair(
                        prompt_id=prompt_id,
                        prompt=prompt,
                        left_model=left.model,
                        right_model=right.model,
                        left_text=left.text,
                        right_text=right.text,
                    )
                )
        return pairs

    def _load_existing_evaluation_rows(self) -> list[dict[str, Any]]:
        if not self.evaluation_path.exists():
            return []
        return pq.read_table(self.evaluation_path).to_pylist()

    def _read_part_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for path in sorted(self.output_dir.glob("part-*.parquet")):
            rows.extend(pq.read_table(path).to_pylist())
        return rows

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
                        prompt_id=[pair.prompt_id],
                        prompt=[pair.prompt],
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

    def _write_consolidated_evaluation(self, rows: list[dict[str, Any]]) -> None:
        table = pa.Table.from_pylist(rows, schema=self.schema)
        pq.write_table(table, self.evaluation_path, compression="zstd")

    def _write_leaderboard(self, rows: list[dict[str, Any]]) -> None:
        wins: dict[str, float] = {}
        losses: dict[str, float] = {}
        comparisons: dict[str, int] = {}
        bt_input: list[tuple[str, str, float, float]] = []

        for row in rows:
            left_model = str(row["left_model"])
            right_model = str(row["right_model"])
            winner = str(row["winner"])

            wins.setdefault(left_model, 0.0)
            wins.setdefault(right_model, 0.0)
            losses.setdefault(left_model, 0.0)
            losses.setdefault(right_model, 0.0)
            comparisons[left_model] = comparisons.get(left_model, 0) + 1
            comparisons[right_model] = comparisons.get(right_model, 0) + 1

            if winner == "left":
                wins[left_model] += 1.0
                losses[right_model] += 1.0
                bt_input.append((left_model, right_model, 1.0, 0.0))
            elif winner == "right":
                wins[right_model] += 1.0
                losses[left_model] += 1.0
                bt_input.append((left_model, right_model, 0.0, 1.0))
            else:
                msg = f"Unexpected winner value: {winner}"
                raise ValueError(msg)

        models = sorted(set(wins))
        bt_scores = self._bt_scores(models=models, results=bt_input)
        leaderboard_rows = []
        for model in models:
            n = comparisons.get(model, 0)
            w = wins.get(model, 0.0)
            leaderboard_rows.append(
                {
                    "model": model,
                    "bt_score": bt_scores.get(model, 0.0),
                    "wins": w,
                    "losses": losses.get(model, 0.0),
                    "n_comparisons": n,
                    "win_rate": (w / n) if n > 0 else 0.0,
                }
            )

        table = pa.Table.from_pylist(
            leaderboard_rows,
            schema=pa.schema(
                [
                    pa.field("model", pa.string()),
                    pa.field("bt_score", pa.float64()),
                    pa.field("wins", pa.float64()),
                    pa.field("losses", pa.float64()),
                    pa.field("n_comparisons", pa.int64()),
                    pa.field("win_rate", pa.float64()),
                ]
            ),
        )
        pq.write_table(table, self.leaderboard_path, compression="zstd")

    async def run(self, candidates: Dataset, resume: bool = False) -> Path:
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for part in self.output_dir.glob("part-*.parquet"):
            part.unlink()
        self.next_part_index = 0

        existing_rows = self._load_existing_evaluation_rows() if resume else []
        if not resume:
            if self.evaluation_path.exists():
                self.evaluation_path.unlink()
            if self.leaderboard_path.exists():
                self.leaderboard_path.unlink()

        seen_pairs = {(str(row["prompt_id"]), str(row["left_model"]), str(row["right_model"])) for row in existing_rows}

        per_prompt = self._collect_candidates(candidates)
        pairs = self._build_pairs(per_prompt=per_prompt)

        write_buffer: list[EvaluationOutputs] = []
        semaphore = asyncio.Semaphore(self.config.max_parallel_requests)
        pending_tasks: set[asyncio.Task[EvaluationOutputs]] = set()

        for pair in tqdm(pairs, desc="Evaluate pairs"):
            pair_key = (pair.prompt_id, pair.left_model, pair.right_model)
            if pair_key in seen_pairs:
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
        new_rows = self._read_part_rows()
        all_rows = existing_rows + new_rows
        self._write_consolidated_evaluation(rows=all_rows)
        self._write_leaderboard(rows=all_rows)

        logger.info(
            "run.done",
            output_dir=str(self.root_dir),
            pair_count=len(all_rows),
            model_count=len({str(r["left_model"]) for r in all_rows} | {str(r["right_model"]) for r in all_rows}),
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
