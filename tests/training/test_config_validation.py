import pytest

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
    assert cfg.reference_length_normalization == "token_mean"
    assert cfg.use_kl is False
    assert cfg.beta == 0.0
    cfg.validate()
