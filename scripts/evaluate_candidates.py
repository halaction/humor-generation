from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.config import EvaluationConfig, config
from src.paths import DATA_DIR
from src.pipelines.evaluation import EvaluationPipeline


def main() -> None:
    load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser(description="Evaluate candidate directories with the pairwise judge pipeline.")
    parser.add_argument("--candidate-dir", action="append", type=Path)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--judge-model")
    parser.add_argument("--max-parallel-requests", type=int)
    parser.add_argument("--judge-temperature", type=float)
    parser.add_argument("--max-retries", type=int)
    parser.add_argument("--limit-references", type=int)
    args = parser.parse_args()

    if args.candidate_dir:
        candidate_dirs = args.candidate_dir
    else:
        root = DATA_DIR / config.candidates.hf_config_name
        candidate_dirs = [path for path in root.iterdir() if path.is_dir()] if root.exists() else []

    if len(candidate_dirs) < 2:
        msg = "At least two candidate directories are required for pairwise evaluation."
        raise ValueError(msg)

    eval_config = config.evaluation.model_copy(
        update={
            key: value
            for key, value in {
                "model": args.judge_model,
                "max_parallel_requests": args.max_parallel_requests,
                "judge_temperature": args.judge_temperature,
                "max_retries": args.max_retries,
            }.items()
            if value is not None
        }
    )
    eval_config = EvaluationConfig.model_validate(eval_config.model_dump())
    pipeline = EvaluationPipeline(pipeline_config=eval_config, output_dir=args.output_dir)
    pipeline.build(
        candidate_paths=candidate_dirs,
        split=args.split,
        resume=args.resume,
        limit_references=args.limit_references,
    )
    print({"evaluation_dir": str(pipeline.output_dir), "leaderboard_dir": str(pipeline.leaderboard_dir)})


if __name__ == "__main__":
    main()
