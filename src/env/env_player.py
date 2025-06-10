from __future__ import annotations

from typing import Any

from poke_env.player import Player


class EnvPlayer(Player):
    """poke_env Player subclass controlled via an action queue."""

    def __init__(self, env: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env = env

    async def choose_move(self, battle):
        # battle 情報を PokemonEnv に渡して行動を決定してもらう
        action_idx: int = self._env.process_battle(battle)
        return self._env.action_helper.action_index_to_order(self, battle, action_idx)

