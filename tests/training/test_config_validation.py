import pytest
import yaml

from src.training.config import MRVFConfig


def test_exact_scaled_requires_none_length_norm() -> None:
    cfg = MRVFConfig(objective_mode="exact_scaled", reference_length_normalization="token_mean")
    with pytest.raises(ValueError, match="exact_scaled"):
        cfg.validate()


def test_num_generations_validation() -> None:
    cfg = MRVFConfig(num_generations=1, objective_mode="log_mass_surrogate")
    with pytest.raises(ValueError, match="num_generations"):
        cfg.validate()


def test_beta_requires_kl_disabled_mode() -> None:
    cfg = MRVFConfig(beta=0.01, use_kl=False)
    with pytest.raises(ValueError, match="use_kl=False"):
        cfg.validate()


def test_use_kl_not_implemented() -> None:
    cfg = MRVFConfig(use_kl=True)
    with pytest.raises(ValueError, match="not implemented"):
        cfg.validate()


def test_default_config_is_stable_surrogate_mode() -> None:
    cfg = MRVFConfig()
    assert cfg.objective_mode == "log_mass_surrogate"
    assert cfg.reward_transform == "log_mass"
    assert cfg.reward_baseline_mode == "none"
    assert cfg.reference_length_normalization == "token_mean"
    assert cfg.use_kl is False
    assert cfg.beta == 0.0
    cfg.validate()


def test_prompt_relative_reward_requires_log_mass_transform() -> None:
    cfg = MRVFConfig(reward_baseline_mode="prompt_relative", reward_transform="centered_prob_mass")
    with pytest.raises(ValueError, match="prompt_relative"):
        cfg.validate()


def test_qwen3_17b_hpc_config_is_valid() -> None:
    data = yaml.safe_load(open("configs/models/qwen3-17b-hpc.yaml", encoding="utf-8"))
    cfg = MRVFConfig(**data)
    cfg.validate()
    assert cfg.num_generations == 4
    assert cfg.max_trace_length == 96
    assert cfg.reference_loss_coef == 0.25
    assert cfg.eval_every_steps == 25
    assert cfg.save_steps == 50


def test_free_thinking_probe_config_is_valid() -> None:
    data = yaml.safe_load(open("configs/models/qwen3-17b-free-think-probe.yaml", encoding="utf-8"))
    cfg = MRVFConfig(**data)
    cfg.validate()
    assert cfg.num_generations == 3
    assert cfg.max_trace_length == 1024
    assert cfg.reference_loss_coef == 0.1
    assert cfg.advantage_mode == "grpo_zscore"
    assert cfg.trace_prompt_template == "training_trace_prompt.j2"
    assert cfg.gradient_checkpointing is True
    assert cfg.temperature == 0.6
    assert cfg.top_k == 20


def test_free_thinking_768_probe_config_is_valid() -> None:
    data = yaml.safe_load(open("configs/models/qwen3-17b-free-think-768x3-probe.yaml", encoding="utf-8"))
    cfg = MRVFConfig(**data)
    cfg.validate()
    assert cfg.num_generations == 3
    assert cfg.max_trace_length == 768
    assert cfg.max_steps == 5
    assert cfg.temperature == 0.6
    assert cfg.top_k == 20


def test_free_thinking_r4_config_is_valid() -> None:
    data = yaml.safe_load(open("configs/models/qwen3-17b-free-think-r4.yaml", encoding="utf-8"))
    cfg = MRVFConfig(**data)
    cfg.validate()
    assert cfg.max_steps == 80
    assert cfg.num_generations == 2
    assert cfg.max_trace_length == 768
    assert cfg.save_steps == 40
    assert cfg.gradient_checkpointing is True
    assert cfg.temperature == 0.6
    assert cfg.top_k == 20


def test_free_thinking_r5_budgeted_config_is_valid() -> None:
    data = yaml.safe_load(open("configs/models/qwen3-17b-free-think-r5-budgeted.yaml", encoding="utf-8"))
    cfg = MRVFConfig(**data)
    cfg.validate()
    assert cfg.max_steps == 80
    assert cfg.num_generations == 2
    assert cfg.max_trace_length == 512
    assert cfg.force_close_thinking is True
    assert cfg.reference_loss_coef == 0.1
    assert cfg.advantage_mode == "grpo_zscore"
    assert cfg.save_steps == 40
    assert cfg.gradient_checkpointing is True
    assert cfg.temperature == 0.6
    assert cfg.top_k == 20


def test_qwen3_4b_budgeted_config_is_valid() -> None:
    data = yaml.safe_load(open("configs/models/qwen3-4b-free-think-r1-budgeted.yaml", encoding="utf-8"))
    cfg = MRVFConfig(**data)
    cfg.validate()
    assert cfg.model_name_or_path == "Qwen/Qwen3-4B"
    assert cfg.max_steps == 35
    assert cfg.num_generations == 2
    assert cfg.max_trace_length == 384
    assert cfg.force_close_thinking is True
    assert cfg.lora_r == 16
    assert cfg.gradient_accumulation_steps == 2
