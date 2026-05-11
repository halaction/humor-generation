from __future__ import annotations

from src.training.config import MRVFConfig
from src.training.data import load_reference_splits
from src.training.mrvf_trainer import MRVFTrainer


def main() -> None:
    cfg = MRVFConfig(
        model_name_or_path="sshleifer/tiny-gpt2",
        output_dir="data/checkpoints/mrvf-tiny",
        max_steps=3,
        max_completion_length=16,
        max_reference_length=16,
        num_reference_samples=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
    )
    train_dataset, eval_dataset = load_reference_splits(
        dataset_name=cfg.dataset_name,
        dataset_config_name=cfg.dataset_config_name,
        train_split=cfg.train_split,
        eval_split=cfg.eval_split,
    )
    MRVFTrainer(cfg).train(raw_train_dataset=train_dataset, raw_eval_dataset=eval_dataset)


if __name__ == "__main__":
    main()
