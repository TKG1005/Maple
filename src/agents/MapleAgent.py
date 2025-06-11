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

    def teampreview(self, battle: Any) -> str:
        """Return a team selection string for the team preview phase.

        This default implementation randomly selects three Pokémon from
        ``battle.team`` and returns a ``"/team"`` command string.  Subclasses
        can override this method to implement a custom selection strategy.
        """

        team_size = len(getattr(battle, "team", [])) if battle else 0
        if team_size <= 0:
            return "/team 123"  # フォールバック

        num_to_select = min(3, team_size)
        indices = self.env.rng.choice(team_size, size=num_to_select, replace=False)
        indices = sorted(int(i) + 1 for i in indices)
        return "/team " + "".join(str(i) for i in indices)

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

    def play_until_done(self, observation: Any, action_mapping: Any, info: dict | None = None) -> None:
        """Keep calling :meth:`PokemonEnv.step` until ``done`` becomes ``True``.

        Parameters
        ----------
        observation : Any
            Initial observation vector returned by ``env.reset``.
        action_mapping : Any
            Mapping of available actions corresponding to ``observation``.
        ``info`` may include ``"request_teampreview"``.  When this flag is
        present and ``True``, a team selection is performed automatically before
        entering the main loop.
        """

        done = False
        current_obs = observation
        current_map = action_mapping

        if info and info.get("request_teampreview"):
            battle = info.get("battle")
            team_order = self.teampreview(battle)
            current_obs, current_map, _, done, info = self.env.step(team_order)

        while not done:
            if current_map:
                action_idx = int(self.env.rng.choice(list(current_map.keys())))
            else:
                action_idx = int(self.env.action_space.sample())

            current_obs, current_map, _, done, _ = self.env.step(action_idx)


