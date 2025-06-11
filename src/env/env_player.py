from __future__ import annotations

import random
from typing import Any, Awaitable

from poke_env.environment.abstract_battle import AbstractBattle
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
        action_data = await self._env._action_queue.get()

        # choose_move は行動インデックスのみを想定する
        if not isinstance(action_data, int):
            raise ValueError("Expected action index from PokemonEnv")

        # 取得したインデックスを BattleOrder に変換して返す
        return self._env.action_helper.action_index_to_order(self, battle, action_data)

    #Playerクラスの_handle_battle_requestをオーバーライド
    async def _handle_battle_request(
        self,
        battle: AbstractBattle,
        from_teampreview_request: bool = False,
        maybe_default_order: bool = False,
    ):

        if maybe_default_order and (
            "illusion" in [p.ability for p in battle.team.values()]
            or random.random() < self.DEFAULT_CHOICE_CHANCE
        ):
            message = self.choose_default_move().message
        elif from_teampreview_request:
            # チーム選択を PokemonEnv に通知して待機
            await self._env._battle_queue.put(battle)
            message = await self._env._action_queue.get()
        else:
            if maybe_default_order:
                self._trying_again.set()
            choice = self.choose_move(battle)
            if isinstance(choice, Awaitable):
                choice = await choice
            message = choice.message

        await self.ps_client.send_message(message, battle.battle_tag)