from __future__ import annotations

import random
import asyncio
import logging
from typing import Any, Awaitable

from poke_env.concurrency import POKE_LOOP

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import Player


class EnvPlayer(Player):
    """poke_env Player subclass controlled via an action queue."""

    def __init__(self, env: Any, player_id: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env = env
        self.player_id = player_id
        self._logger = logging.getLogger(__name__)

    async def battle_against(
        self,
        *opponents: Player,
        n_battles: int = 1,
        battle_seed: int | None = None,
    ):
        """Wrapper to propagate ``battle_seed`` to :class:`Player`."""
        return await super().battle_against(
            *opponents, n_battles=n_battles, battle_seed=battle_seed
        )

    async def choose_move(self, battle):
        """Return the order chosen by the external agent via :class:`PokemonEnv`."""

        # PokemonEnv に最新の battle オブジェクトを送信
        self._logger.debug(
            "[DBG] %s queue battle -> %s", self.player_id, battle.battle_tag
        )
        await self._env._battle_queues[self.player_id].put(battle)

        # PokemonEnv.step からアクションが投入されるまで待機
        action_data = await asyncio.wait_for(
            self._env._action_queues[self.player_id].get(), self._env.timeout
        )
        self._env._action_queues[self.player_id].task_done()
        self._logger.debug("[DBG] %s action received %s", self.player_id, action_data)

        # 文字列はそのまま、整数は BattleOrder へ変換
        if isinstance(action_data, int):
            order = self._env.action_helper.action_index_to_order(
                self, battle, action_data
            )
            self._logger.debug(
                "[DBG] %s action index %s -> %s",
                self.player_id,
                action_data,
                order.message,
            )
            return order
        self._logger.debug("[DBG] %s direct order %s", self.player_id, action_data)
        return action_data

    def _battle_finished_callback(self, battle: AbstractBattle) -> None:
        """Called when a battle ends to notify :class:`PokemonEnv`."""

        self._logger.debug(
            "[DBG] %s finished battle %s", self.player_id, battle.battle_tag
        )
        asyncio.run_coroutine_threadsafe(
            self._env._battle_queues[self.player_id].put(battle),
            POKE_LOOP,
        )

    # Playerクラスの_handle_battle_requestをオーバーライド
    async def _handle_battle_request(
        self,
        battle: AbstractBattle,
        from_teampreview_request: bool = False,
        maybe_default_order: bool = False,
    ):

        # ``battle.available_moves`` が空の場合は更新を待機する
        if not battle.available_moves:

            async def _wait_moves() -> None:
                while (
                    not battle.available_moves
                    and not battle.force_switch
                    and not battle.move_on_next_request
                    and not battle.teampreview
                ):
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
            self._logger.debug(
                "[DBG] %s send team preview request for %s",
                self.player_id,
                battle.battle_tag,
            )
            put_result = await asyncio.wait_for(
                self._env._battle_queues[self.player_id].put(battle),
                self._env.timeout,
            )
            if put_result is not None:
                self._env._battle_queues[self.player_id].task_done()
            message = await asyncio.wait_for(
                self._env._action_queues[self.player_id].get(),
                self._env.timeout,
            )
            self._env._action_queues[self.player_id].task_done()
            self._logger.debug(
                "[DBG] %s team preview message %s (%s)",
                self.player_id,
                message,
                battle.player_username,
            )
        else:
            if maybe_default_order:
                self._trying_again.set()
            choice = self.choose_move(battle)
            if isinstance(choice, Awaitable):
                choice = await choice
            message = choice.message

        self._logger.debug(
            "[DBG] %s send message '%s' to battle %s",
            self.player_id,
            message,
            battle.battle_tag,
        )
        await self.ps_client.send_message(message, battle.battle_tag)
