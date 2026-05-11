from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import yaml

from src.training.config import MRVFConfig
from src.training.data import load_reference_splits
from src.training.sft_baseline_trainer import SFTBaselineTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SFT baseline on reference pairs.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = MRVFConfig(**yaml.safe_load(Path(args.config).read_text(encoding="utf-8")))
    train_dataset, eval_dataset = load_reference_splits(
        dataset_name=cfg.dataset_name,
        dataset_config_name=cfg.dataset_config_name,
        train_split=cfg.train_split,
        eval_split=cfg.eval_split,
    )
    result = SFTBaselineTrainer(cfg).train(raw_train_dataset=train_dataset, raw_eval_dataset=eval_dataset)
    print(yaml.safe_dump({"config": asdict(cfg), "result": result}, sort_keys=False))


if __name__ == "__main__":
    main()
