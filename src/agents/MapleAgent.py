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

        team_size = 6
        num_to_select = min(3, team_size)
        indices = self.env.rng.choice(team_size, size=num_to_select, replace=False)
        indices = sorted(int(i) + 1 for i in indices)
        team_cmd = "/team " + "".join(str(i) for i in indices)
        return team_cmd

    def select_action(self, observation: Any, action_mask: Any) -> int:
        """Return a random valid action index.

        Parameters
        ----------
        observation : Any
            Observation vector generated from :class:`PokemonEnv`.
        action_mask : Any
            Action availability mask vector as provided by
            ``action_helper.get_available_actions``.
        Returns
        -------
        int
            Selected action index.
        """

        # ``action_mask`` は ``action_helper.get_available_actions`` から得られる
        # ベクトルで、値が 1 のインデックスが選択可能な行動を示す。マスクが無効な
        # 場合は ``action_space`` 全体からサンプリングする。
        try:
            valid_indices = [i for i, flag in enumerate(action_mask) if flag]
        except Exception:
            valid_indices = []
        print(f"available mask = {action_mask}")
        if valid_indices:
            action_idx = int(self.env.rng.choice(valid_indices))
        else:
            # ``action_space`` は ``gym.spaces.Dict`` なので ``sample`` の
            # 戻り値は辞書となる。ここでは単一エージェント分の離散
            # 空間からサンプルする。
            subspace = self.env.action_space[self.env.agent_ids[0]]
            action_idx = int(subspace.sample())

        return action_idx

    def play_until_done(
        self, observation: Any, action_mask: Any, info: dict | None = None
    ) -> None:
        """Keep acting until ``done`` becomes ``True``.

        Parameters
        ----------
        observation : Any
            Initial observation vector returned by ``env.reset``.
        action_mask : Any
            Action availability mask corresponding to ``observation``.
        ``info`` may include ``"request_teampreview"``.  When this flag is
        present and ``True``, a team selection is performed automatically by
        calling :meth:`choose_team` before entering the main loop.

        The selected index from :meth:`select_action` is passed to
        :meth:`PokemonEnv.step` each iteration.
        """

        done = False
        current_obs = observation
        current_mask = action_mask

        while not done:
            if info and info.get("request_teampreview"):
                team_order = self.choose_team(current_obs)
                current_obs, current_mask, _, done, info = self.env.step(team_order)
            else:
                action_idx = self.select_action(current_obs, current_mask)
                print(f"Agent select /move {action_idx}")
                current_obs, current_mask, _, done, _ = self.env.step(action_idx)
