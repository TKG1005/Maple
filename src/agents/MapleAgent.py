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
        """Select a random valid action and execute ``env.step``.

        Parameters
        ----------
        observation : Any
            Observation vector generated from :class:`PokemonEnv`.
        action_mapping : Any
            Mapping of available action indices as provided by
            ``action_helper.get_available_actions_with_details``.
        """

        # ``action_mapping`` は ``action_helper.get_available_actions_with_details``
        # から得られる辞書で、キーに選択可能な行動 index が含まれている。マッピングが
        # 空の場合は ``action_space`` 全体からサンプリングする。
        if action_mapping:
            action_idx = int(self.env.rng.choice(list(action_mapping.keys())))
        else:
            action_idx = int(self.env.action_space.sample())

        # ランダムに選んだ行動を即座に環境へ反映させる
        # ``step`` の戻り値はここでは利用しない
        self.env.step(action_idx)

        return action_idx

    def play_until_done(self, observation: Any, action_mapping: Any) -> None:
        """Keep calling :meth:`PokemonEnv.step` until ``done`` becomes ``True``.

        Parameters
        ----------
        observation : Any
            Initial observation vector returned by ``env.reset``.
        action_mapping : Any
            Mapping of available actions corresponding to ``observation``.
        """

        done = False
        current_obs = observation
        current_map = action_mapping

        while not done:
            if current_map:
                action_idx = int(self.env.rng.choice(list(current_map.keys())))
            else:
                action_idx = int(self.env.action_space.sample())

            current_obs, current_map, _, done, _ = self.env.step(action_idx)


