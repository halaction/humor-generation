from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv
import yaml

from src.training.config import MRVFConfig
from src.training.data import load_reference_splits
from src.training.mrvf_trainer import MRVFTrainer


def _load_config(path: Path) -> MRVFConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return MRVFConfig(**raw)


def main() -> None:
    load_dotenv(Path(".env"))
    parser = argparse.ArgumentParser(description="Train MRVF model with grouped trace sampling.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    train_dataset, eval_dataset = load_reference_splits(
        dataset_name=cfg.dataset_name,
        dataset_config_name=cfg.dataset_config_name,
        train_split=cfg.train_split,
        eval_split=cfg.eval_split,
    )
    metrics = MRVFTrainer(cfg).train(raw_train_dataset=train_dataset, raw_eval_dataset=eval_dataset)
    print(yaml.safe_dump({"config": asdict(cfg), "result": metrics}, sort_keys=False))


if __name__ == "__main__":
    main()
