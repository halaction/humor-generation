from __future__ import annotations

from pathlib import Path

import yaml

from src.training.config import MRVFConfig
from src.training.data import load_reference_splits
from src.training.mrvf_trainer import MRVFTrainer


class TrainingPipeline:
    def __init__(self, config_path: str) -> None:
        raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        self.cfg = MRVFConfig(**raw)

    def build(self) -> dict:
        train_dataset, eval_dataset = load_reference_splits(
            dataset_name=self.cfg.dataset_name,
            dataset_config_name=self.cfg.dataset_config_name,
            train_split=self.cfg.train_split,
            eval_split=self.cfg.eval_split,
        )
        return MRVFTrainer(self.cfg).train(raw_train_dataset=train_dataset, raw_eval_dataset=eval_dataset)
