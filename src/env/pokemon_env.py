"""Gymnasium environment skeleton for Pokémon battles."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

import gymnasium as gym


class PokemonEnv(gym.Env):
    """A placeholder Gymnasium environment for Pokémon battles."""

    metadata = {"render_modes": [None]}

    def __init__(
        self,
        opponent_player: Any,
        state_observer: Any,
        action_helper: Any,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.opponent_player = opponent_player
        self.state_observer = state_observer
        self.action_helper = action_helper
        self.rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Any, dict]:
        """Reset the environment and return the initial observation and info."""
        super().reset(seed=seed)
        observation = None  # TODO: replace with initial observation
        info: dict = {}
        return observation, info

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, dict]:
        """Take a step in the environment using the given action."""
        observation = None  # TODO: next observation
        reward: float = 0.0
        terminated: bool = True
        truncated: bool = False
        info: dict = {}
        return observation, reward, terminated, truncated, info

    def render(self) -> None:
        """Render the environment if applicable."""
        return None

    def close(self) -> None:
        """Clean up resources used by the environment."""
        pass
