from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Pre-download Hugging Face assets for offline jobs.")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dataset", default="halaction/humor-generation")
    parser.add_argument("--dataset-config", default="references")
    parser.add_argument("--cache-dir", default=None)
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser() if args.cache_dir else None
    load_dataset(args.dataset, args.dataset_config, cache_dir=str(cache_dir) if cache_dir else None)
    AutoTokenizer.from_pretrained(args.model, cache_dir=str(cache_dir) if cache_dir else None)
    snapshot_download(repo_id=args.model, cache_dir=str(cache_dir) if cache_dir else None)


if __name__ == "__main__":
    main()
