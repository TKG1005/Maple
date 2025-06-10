from __future__ import annotations

from typing import Any

from src.env.pokemon_env import PokemonEnv


class MapleAgent:
    """Base agent class for interacting with :class:`PokemonEnv`."""

    def __init__(self, env: PokemonEnv) -> None:
        self.env = env

    def teampreview(self, battle: Any) -> None:
        """Hook for team preview phase. Override in subclasses."""
        pass

    def select_action(self, observation: Any) -> int:
        """Return an action index given an observation."""
        raise NotImplementedError

