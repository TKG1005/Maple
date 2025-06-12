from __future__ import annotations

import random
import asyncio
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
        print("[DBG_STOP] EnvPlayer.choose_move put battle", battle.battle_tag, "turn", battle.turn)
        await self._env._battle_queue.put(battle)

        # PokemonEnv.step からアクションが投入されるまで待機
        print("[DBG_STOP] EnvPlayer.choose_move wait action")
        action_data = await asyncio.wait_for(
            self._env._action_queue.get(), self._env.timeout
        )
        self._env._action_queue.task_done()

        print("[DBG_STOP] EnvPlayer.choose_move got action", action_data)

        # 文字列はそのまま、整数は BattleOrder へ変換
        if isinstance(action_data, int):
            return self._env.action_helper.action_index_to_order(
                self, battle, action_data
            )
        return action_data

    # Playerクラスの_handle_battle_requestをオーバーライド
    async def _handle_battle_request(
        self,
        battle: AbstractBattle,
        from_teampreview_request: bool = False,
        maybe_default_order: bool = False,
    ):

        print(
            f"[DBG_STOP] EnvPlayer._handle_battle_request start tp={from_teampreview_request} turn={battle.turn} user={battle.player_username}"
        )

        # 最初のターンでは ``battle.available_moves`` が更新されるまで待機する
        if battle.turn == 1 and not battle.available_moves:

            async def _wait_moves() -> None:
                while not battle.available_moves:
                    await asyncio.sleep(0.1)

            try:
                await asyncio.wait_for(_wait_moves(), timeout=10.0)
            except asyncio.TimeoutError as exc:
                raise RuntimeError("No available moves after 10 seconds") from exc

        if maybe_default_order and (
            "illusion" in [p.ability for p in battle.team.values()]
            or random.random() < self.DEFAULT_CHOICE_CHANCE
        ):
            message = self.choose_default_move().message
        elif from_teampreview_request:
            # チーム選択を PokemonEnv に通知して待機
            put_result = await asyncio.wait_for(
                self._env._battle_queue.put(battle), self._env.timeout
            )
            if put_result is not None:
                self._env._battle_queue.task_done()
            message = await asyncio.wait_for(
                self._env._action_queue.get(), self._env.timeout
            )
            self._env._action_queue.task_done()
            print(f"チーム選択を送信 {message}　プレイヤー＝{battle.player_username}")
        else:
            if maybe_default_order:
                self._trying_again.set()
            choice = self.choose_move(battle)
            if isinstance(choice, Awaitable):
                choice = await choice
            message = choice.message

        await self.ps_client.send_message(
            message,
            battle.battle_tag,
        )
        print("[DBG_STOP] EnvPlayer._handle_battle_request sent message", message)
