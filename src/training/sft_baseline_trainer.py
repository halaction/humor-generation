from __future__ import annotations

from dataclasses import asdict
from typing import Any

from datasets import Dataset

from src.training.config import MRVFConfig
from src.training.data import prepare_mrvf_dataset


def _load_sft_stack() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as error:  # pragma: no cover
        msg = (
            "SFT dependencies are missing. Install one of: "
            "`requirements/training-cpu.txt`, `requirements/training-colab.txt`, "
            "or `requirements/training-hpc.txt`."
        )
        raise RuntimeError(msg) from error
    return AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer, None


class SFTBaselineTrainer:
    def __init__(self, cfg: MRVFConfig) -> None:
        self.cfg = cfg

    def train(self, raw_train_dataset: Any, raw_eval_dataset: Any) -> dict[str, Any]:
        AutoModelForCausalLM, AutoTokenizer, SFTConfig, SFTTrainer, _ = _load_sft_stack()
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model_name_or_path, use_fast=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(self.cfg.model_name_or_path)

        train_dataset = prepare_mrvf_dataset(raw_train_dataset, max_reference_samples=self.cfg.num_reference_samples)
        eval_dataset = prepare_mrvf_dataset(raw_eval_dataset, max_reference_samples=self.cfg.num_reference_samples)

        flattened_train_rows: list[dict[str, str]] = []
        for row in train_dataset:
            for item in row["references"]:
                flattened_train_rows.append({"prompt": row["prompt"], "completion": item})

        flattened_eval_rows: list[dict[str, str]] = []
        for row in eval_dataset:
            for item in row["references"]:
                flattened_eval_rows.append({"prompt": row["prompt"], "completion": item})

        flattened_train = Dataset.from_list(flattened_train_rows)
        flattened_eval = Dataset.from_list(flattened_eval_rows)

        args = SFTConfig(
            output_dir=self.cfg.output_dir,
            learning_rate=self.cfg.learning_rate,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            max_steps=self.cfg.max_steps,
            eval_strategy="steps",
            eval_steps=max(1, min(50, self.cfg.max_steps)),
            logging_steps=max(1, min(10, self.cfg.max_steps)),
            save_steps=max(1, min(100, self.cfg.max_steps)),
            max_length=self.cfg.max_completion_length + self.cfg.max_reference_length,
            report_to=[],
            seed=self.cfg.seed,
        )
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=flattened_train,
            eval_dataset=flattened_eval,
            processing_class=tokenizer,
        )
        result = trainer.train()
        trainer.save_model(self.cfg.output_dir)
        return {"metrics": result.metrics, "config": asdict(self.cfg)}
