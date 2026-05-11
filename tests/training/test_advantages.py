import pytest

from src.training.advantages import grpo_zscore_advantages, loo_advantages

torch = pytest.importorskip("torch")


def test_loo_advantages_pairwise() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = loo_advantages(rewards, num_generations=2)
    expected = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    assert torch.allclose(result, expected)


def test_grpo_advantages_finite() -> None:
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
    result = grpo_zscore_advantages(rewards, num_generations=2)
    assert torch.isfinite(result).all()
