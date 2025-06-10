from __future__ import annotations

from typing import Any

from src.env.pokemon_env import PokemonEnv


class MapleAgent:
    """Base agent class for interacting with :class:`PokemonEnv`."""

    def __init__(self, env: PokemonEnv) -> None:
        self.env = env
        # 環境側にエージェントを登録して相互参照を確立しておく
        if hasattr(self.env, "register_agent"):
            self.env.register_agent(self)

    def teampreview(self, battle: Any) -> None:
        """Hook for team preview phase. Override in subclasses."""
        pass

    def select_action(self, observation: Any, action_mapping: Any) -> int:
        """Return an action index given the current observation and actions.

        Parameters
        ----------
        observation : Any
            Observation vector generated from :class:`PokemonEnv`.
        action_mapping : Any
            Mapping of available action indices as provided by
            ``action_helper.get_available_actions_with_details``.
        """
        raise NotImplementedError

