from __future__ import annotations

from typing import Any

from poke_env.player import Player


class EnvPlayer(Player):
    """poke_env Player subclass controlled via an action queue."""

    def __init__(self, env: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env = env

    async def choose_move(self, battle):
        """Return the order chosen by the external agent via :class:`PokemonEnv`."""

        # PokemonEnv に最新の battle オブジェクトを送信
        await self._env._battle_queue.put(battle)

        # PokemonEnv.step からアクションが投入されるまで待機
        action_idx: int = await self._env._action_queue.get()

        # 取得したインデックスを BattleOrder に変換して返す
        return self._env.action_helper.action_index_to_order(self, battle, action_idx)
