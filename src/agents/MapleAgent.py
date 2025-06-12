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

    def choose_team(self, observation: Any) -> str:
        """Return a team selection string for the team preview phase.

        The base implementation ignores ``observation`` and simply selects
        three random Pokémon assuming a team size of six.  Subclasses can
        override this method to implement a custom selection strategy.
        """

        print("チームプレビューリクエスト確認")

        team_size = 6
        num_to_select = min(3, team_size)
        indices = self.env.rng.choice(team_size, size=num_to_select, replace=False)
        indices = sorted(int(i) + 1 for i in indices)
        return "/team " + "".join(str(i) for i in indices)

    def select_action(self, observation: Any, action_mapping: Any) -> int:
        """Return a random valid action index.

        Parameters
        ----------
        observation : Any
            Observation vector generated from :class:`PokemonEnv`.
        action_mapping : Any
            Mapping of available action indices as provided by
            ``action_helper.get_available_actions_with_details``.
        Returns
        -------
        int
            Selected action index.
        """

        # ``action_mapping`` は ``action_helper.get_available_actions_with_details``
        # から得られる辞書で、キーに選択可能な行動 index が含まれている。マッピングが
        # 空の場合は ``action_space`` 全体からサンプリングする。
        if action_mapping:
            action_idx = int(self.env.rng.choice(list(action_mapping.keys())))
        else:
            action_idx = int(self.env.action_space.sample())

        return action_idx

    def play_until_done(self, observation: Any, action_mapping: Any, info: dict | None = None) -> None:
        """Keep acting until ``done`` becomes ``True``.

        Parameters
        ----------
        observation : Any
            Initial observation vector returned by ``env.reset``.
        action_mapping : Any
            Mapping of available actions corresponding to ``observation``.
        ``info`` may include ``"request_teampreview"``.  When this flag is
        present and ``True``, a team selection is performed automatically by
        calling :meth:`choose_team` before entering the main loop.

        The selected index from :meth:`select_action` is passed to
        :meth:`PokemonEnv.step` each iteration.
        """

        done = False
        current_obs = observation
        current_map = action_mapping

        if info and info.get("request_teampreview"):
            team_order = self.choose_team(current_obs)
            current_obs, current_map, _, done, info = self.env.step(team_order)

        while not done:
            action_idx = self.select_action(current_obs, current_map)
            print(f"Agent select /move {action_idx}")
            current_obs, current_map, _, done, _ = self.env.step(action_idx)


