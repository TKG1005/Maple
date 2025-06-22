from __future__ import annotations

from typing import Sequence

import numpy as np


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    gamma: float = 0.99,
    lam: float = 0.95,
    last_value: float = 0.0,
) -> np.ndarray:
    """Compute Generalized Advantage Estimation (GAE)."""
    rewards_arr = np.asarray(rewards, dtype=np.float32)
    values_arr = np.asarray(values, dtype=np.float32)
    if len(rewards_arr) != len(values_arr):
        raise ValueError("rewards and values must have the same length")

    advantages = np.zeros_like(rewards_arr, dtype=np.float32)
    gae = 0.0
    for t in range(len(rewards_arr) - 1, -1, -1):
        next_value = values_arr[t + 1] if t + 1 < len(values_arr) else last_value
        delta = rewards_arr[t] + gamma * next_value - values_arr[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    return advantages

__all__ = ["compute_gae"]
