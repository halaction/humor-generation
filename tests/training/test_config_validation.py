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


def test_beta_must_be_zero_for_now() -> None:
    cfg = MRVFConfig(beta=0.01)
    with pytest.raises(ValueError, match="KL regularization is temporarily disabled"):
        cfg.validate()
