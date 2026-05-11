from __future__ import annotations

from typing import Any


def loo_advantages(rewards: Any, num_generations: int) -> Any:
    if num_generations < 2:
        msg = "`num_generations` must be >= 2 for leave-one-out advantages."
        raise ValueError(msg)
    grouped = rewards.view(-1, num_generations)
    baseline = (grouped.sum(dim=1, keepdim=True) - grouped) / (num_generations - 1)
    return (grouped - baseline).reshape(-1)


def grpo_zscore_advantages(rewards: Any, num_generations: int, eps: float = 1e-4) -> Any:
    grouped = rewards.view(-1, num_generations)
    means = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, keepdim=True).clamp_min(eps)
    return ((grouped - means) / std).reshape(-1)
