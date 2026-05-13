from __future__ import annotations

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from huggingface_hub import HfApi


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    commit = result.stdout.strip()
    return commit or None


def _metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "checkpoint_dir": str(args.checkpoint_dir),
        "repo_id": args.repo_id,
        "path_in_repo": args.path_in_repo,
        "base_model": args.base_model,
        "dataset_name": args.dataset_name,
        "dataset_config_name": args.dataset_config_name,
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "wandb_url": args.wandb_url,
        "git_commit": _git_commit(),
    }


def _upload_one(api: HfApi, *, args: argparse.Namespace, folder: Path, path_in_repo: str, token: str) -> None:
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(folder),
        path_in_repo=path_in_repo,
        token=token,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        metadata_path = Path(tmpdir) / "run_metadata.yaml"
        metadata = _metadata(args)
        metadata["uploaded_checkpoint_dir"] = str(folder)
        metadata["uploaded_path_in_repo"] = path_in_repo
        metadata_path.write_text(yaml.safe_dump(metadata, sort_keys=False), encoding="utf-8")
        api.upload_file(
            repo_id=args.repo_id,
            repo_type="model",
            path_or_fileobj=str(metadata_path),
            path_in_repo=str(Path(path_in_repo) / "run_metadata.yaml"),
            token=token,
        )


def _step_path_name(step_dir: Path) -> str:
    match = re.fullmatch(r"step-(\d+)", step_dir.name)
    if match is None:
        return step_dir.name
    return f"step-{int(match.group(1)):04d}"


def main() -> None:
    load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser(description="Upload a LoRA adapter checkpoint folder to Hugging Face Hub.")
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--repo-id", default="halaction/Qwen3-1.7B-humor-lora")
    parser.add_argument("--path-in-repo", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset-name", default="halaction/humor-generation")
    parser.add_argument("--dataset-config-name", default="references")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--wandb-url")
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--upload-step-checkpoints", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    if not args.checkpoint_dir.exists():
        msg = f"Checkpoint directory does not exist: {args.checkpoint_dir}"
        raise FileNotFoundError(msg)

    token = os.environ.get("HF_TOKEN")
    if not token:
        msg = "HF_TOKEN must be set to upload checkpoints."
        raise RuntimeError(msg)

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="model", private=args.private, exist_ok=True)

    uploaded = []
    if args.upload_step_checkpoints:
        run_root = Path(args.path_in_repo).parent
        for step_dir in sorted(args.checkpoint_dir.glob("step-*")):
            if not step_dir.is_dir():
                continue
            step_path = str(run_root / _step_path_name(step_dir))
            _upload_one(api, args=args, folder=step_dir, path_in_repo=step_path, token=token)
            uploaded.append(step_path)

    _upload_one(api, args=args, folder=args.checkpoint_dir, path_in_repo=args.path_in_repo, token=token)
    uploaded.append(args.path_in_repo)

    print(
        {
            "repo_id": args.repo_id,
            "uploaded": uploaded,
        }
    )


if __name__ == "__main__":
    main()
