from __future__ import annotations

from collections import deque
from typing import Any, Deque, Tuple

import numpy as np


class ReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity: int, observation_shape: Tuple[int, ...]):
        self.capacity = int(capacity)
        self.observation_shape = observation_shape
        self.observations: Deque[np.ndarray] = deque(maxlen=self.capacity)
        self.actions: Deque[int] = deque(maxlen=self.capacity)
        self.rewards: Deque[float] = deque(maxlen=self.capacity)
        self.dones: Deque[bool] = deque(maxlen=self.capacity)
        self.next_observations: Deque[np.ndarray] = deque(maxlen=self.capacity)

    def add(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        done: bool,
        next_observation: np.ndarray,
    ) -> None:
        self.observations.append(np.asarray(observation, dtype=np.float32))
        self.actions.append(int(action))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self.next_observations.append(np.asarray(next_observation, dtype=np.float32))

    def sample(self, batch_size: int) -> dict[str, Any]:
        indices = np.random.randint(0, len(self), size=batch_size)
        obs = np.stack([self.observations[i] for i in indices])
        actions = np.array([self.actions[i] for i in indices], dtype=np.int64)
        rewards = np.array([self.rewards[i] for i in indices], dtype=np.float32)
        dones = np.array([self.dones[i] for i in indices], dtype=np.float32)
        next_obs = np.stack([self.next_observations[i] for i in indices])
        return {
            "observations": obs,
            "actions": actions,
            "rewards": rewards,
            "dones": dones,
            "next_observations": next_obs,
        }

    def __len__(self) -> int:
        return len(self.observations)
