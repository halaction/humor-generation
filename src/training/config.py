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
    objective_mode: Literal["exact_scaled", "log_mass_surrogate", "mrvf_lite"] = "log_mass_surrogate"
    reward_transform: Literal["log_mass", "centered_prob_mass"] = "log_mass"
    advantage_mode: Literal["loo", "grpo_zscore"] = "loo"
    reference_length_normalization: Literal["none", "token_mean", "sqrt"] = "token_mean"
    trace_loss_coef: float = 1.0
    reference_loss_coef: float = 0.5
    use_kl: bool = False
    beta: float = 0.0
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int | None = None
    repetition_penalty: float = 1.0
    trace_instruction: str = (
        "Think briefly about a joke plan. Use at most three short sentences: setup, wordplay, twist. "
        "Do not write the final joke."
    )
    trace_prompt_template: str = "training_trace_prompt.j2"
    force_close_thinking: bool = False
    forced_thinking_suffix: str = (
        "\n\nConsidering the limited time, I will now answer from this reasoning.\n</think>\n\n"
    )
    answer_prefix: str = "\nFinal joke:\n"
    seed: int = 42
    use_thinking: bool = True
    strip_thinking_in_outputs: bool = True
    use_peft: bool = False
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    torch_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    gradient_checkpointing: bool = False
    eval_every_steps: int = 0
    eval_sample_size: int = 16
    trace_format: Literal["plain", "qwen_chat_thinking"] = "plain"
    logging_steps: int = 10
    save_steps: int = 100
    metrics_log_path: str = "data/logs/mrvf_metrics.jsonl"
    sample_log_path: str = "data/logs/mrvf_samples.jsonl"
    report_to_wandb: bool = False
    wandb_project: str = "humor-generation"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_tags: tuple[str, ...] = ()

    def validate(self) -> None:
        if self.num_generations < 2:
            msg = "`num_generations` must be >= 2."
            raise ValueError(msg)
        if self.trace_loss_coef < 0 or self.reference_loss_coef < 0 or self.beta < 0:
            msg = "`trace_loss_coef`, `reference_loss_coef`, and `beta` must be non-negative."
            raise ValueError(msg)
        if not self.use_kl and self.beta != 0:
            msg = "`use_kl=False` requires `beta=0.0`."
            raise ValueError(msg)
        if self.use_kl:
            msg = "KL mode is not implemented yet; set `use_kl=False`."
            raise ValueError(msg)
        if self.objective_mode == "exact_scaled" and self.reference_length_normalization != "none":
            msg = "`objective_mode=exact_scaled` requires `reference_length_normalization=none`."
            raise ValueError(msg)
        if self.eval_every_steps < 0 or self.eval_sample_size < 0:
            msg = "`eval_every_steps` and `eval_sample_size` must be non-negative."
            raise ValueError(msg)
        if self.top_k is not None and self.top_k <= 0:
            msg = "`top_k` must be positive when set."
            raise ValueError(msg)
        if self.repetition_penalty <= 0:
            msg = "`repetition_penalty` must be positive."
            raise ValueError(msg)
        if not self.trace_prompt_template.strip():
            msg = "`trace_prompt_template` must not be empty."
            raise ValueError(msg)
        if self.force_close_thinking and not self.forced_thinking_suffix.strip():
            msg = "`forced_thinking_suffix` must not be empty when `force_close_thinking=True`."
            raise ValueError(msg)
