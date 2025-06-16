from __future__ import annotations

from typing import Tuple, List

import numpy as np


class ReplayBuffer:
    """Simple experience replay buffer."""

    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.position = 0

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in the buffer."""
        data = (np.asarray(state), int(action), float(reward), np.asarray(next_state), bool(done))

        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Randomly sample a batch of transitions."""
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*(self.buffer[i] for i in indices))
        return (
            np.stack(states),
            np.asarray(actions),
            np.asarray(rewards, dtype=np.float32),
            np.stack(next_states),
            np.asarray(dones, dtype=np.float32),
        )


__all__ = ["ReplayBuffer"]
