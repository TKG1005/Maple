from __future__ import annotations

from typing import Any

from poke_env.player import Player


class EnvPlayer(Player):
    """poke_env Player subclass controlled via an action queue."""

    def __init__(self, env: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env = env

    async def choose_move(self, battle):
        action_idx: int = await self._env._action_queue.get()
        return self._env.action_helper.action_index_to_order(self, battle, action_idx)

