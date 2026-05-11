from dataclasses import dataclass
from typing import Literal


@dataclass
class MRVFConfig:
    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    output_dir: str = "data/checkpoints/mrvf"
    dataset_name: str = "halaction/humor-generation"
    dataset_config_name: str = "references"
    train_split: str = "train"
    eval_split: str = "validation"
    max_steps: int = 100
    learning_rate: float = 1e-5
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    num_generations: int = 2
    max_completion_length: int = 64
    max_trace_length: int = 64
    max_reference_length: int = 128
    num_reference_samples: int = 5
    objective_mode: Literal["exact_scaled", "log_mass_surrogate", "mrvf_lite"] = "exact_scaled"
    reward_transform: Literal["log_mass", "centered_prob_mass"] = "centered_prob_mass"
    advantage_mode: Literal["loo", "grpo_zscore"] = "loo"
    reference_length_normalization: Literal["none", "token_mean", "sqrt"] = "token_mean"
    trace_loss_coef: float = 1.0
    reference_loss_coef: float = 0.5
    beta: float = 0.02
    temperature: float = 1.0
    top_p: float = 0.95
    trace_instruction: str = "First think briefly about setup and twist. Return only the plan."
    answer_prefix: str = "\nFinal joke:\n"
    seed: int = 42
    use_thinking: bool = True
    strip_thinking_in_outputs: bool = True
    use_peft: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    gradient_checkpointing: bool = False
    eval_every_steps: int = 0

    def validate(self) -> None:
        if self.num_generations < 2:
            msg = "`num_generations` must be >= 2."
            raise ValueError(msg)
        if self.trace_loss_coef < 0 or self.reference_loss_coef < 0 or self.beta < 0:
            msg = "`trace_loss_coef`, `reference_loss_coef`, and `beta` must be non-negative."
            raise ValueError(msg)
