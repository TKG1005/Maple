from __future__ import annotations

import random
import asyncio
import logging
from typing import Any, Awaitable

from poke_env.concurrency import POKE_LOOP

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import Player
from poke_env.data.gen_data import GenData
from poke_env.environment.pokemon import Pokemon
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from .custom_battle import CustomBattle


class EnvPlayer(Player):
    """poke_env Player subclass controlled via an action queue."""

    def __init__(self, env: Any, player_id: str, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env = env
        self.player_id = player_id
        self._logger = logging.getLogger(__name__)
        # 前回処理した Battle.last_request を保持しておく
        self._last_request: Any | None = None

    async def _send_battle_message_local_aware(self, battle: AbstractBattle, message: str) -> None:
        """Send a battle message via IPC in local mode, otherwise via PSClient.

        - Local (IPC) mode: route to Node controller using room mapping.
        - Online mode: fallback to PSClient (standard poke-env behavior).
        """
        try:
            # Prefer IPC route when available and configured for local mode
            if hasattr(self, "mode") and getattr(self, "mode") == "local" and hasattr(self, "ipc_client_wrapper"):
                # Resolve room key (room_tag) from battle tag if mapping exists
                room_key = battle.battle_tag
                try:
                    if hasattr(self, "get_room_tag"):
                        mapped = self.get_room_tag(battle.battle_tag)  # type: ignore[attr-defined]
                        if isinstance(mapped, str) and mapped:
                            room_key = mapped
                except Exception:
                    pass

                # Map python player id to Showdown id (p1/p2)
                sd_id = "p1" if self.player_id == "player_0" else "p2"
                self._logger.debug(
                    "[DBG] %s IPC send protocol battle_id=%s sd_id=%s data=%s",
                    self.player_id,
                    room_key,
                    sd_id,
                    message,
                )
                payload = {
                    "type": "protocol",
                    "battle_id": room_key,
                    "player_id": sd_id,
                    "data": message,
                }
                # Use IPC client wrapper to send protocol to the controller
                await self.ipc_client_wrapper.send(payload)  # type: ignore[attr-defined]
                return

            # Fallback to PSClient (online mode)
            await self.ps_client.send_message(message, battle.battle_tag)
        except Exception:
            # Ensure exceptions propagate but log context for diagnostics
            self._logger.exception(
                "Failed to send battle message (mode=%s, room=%s, msg=%s)",
                getattr(self, "mode", "unknown"),
                getattr(battle, "battle_tag", "unknown"),
                message,
            )
            raise

    async def choose_move(self, battle):
        """Return the order chosen by the external agent via :class:`PokemonEnv`."""
        
        # PokemonEnv に最新の battle オブジェクトを送信
        self._logger.debug(
            "[DBG] %s queue battle -> %s trapped = %s", self.player_id, battle.battle_tag,battle.trapped
        )
        await self._env._battle_queues[self.player_id].put(battle)

        # PokemonEnv.step からアクションが投入されるまで待機
        try:
            self._logger.debug(
                "[DBG] %s waiting action (qsize=%d)", 
                self.player_id,
                self._env._action_queues[self.player_id].qsize(),
            )
            action_data = await asyncio.wait_for(
                self._env._action_queues[self.player_id].get(), self._env.timeout
            )
            self._env._action_queues[self.player_id].task_done()
            self._logger.debug(
                "[DBG] %s action received %s", self.player_id, action_data
            )
        except asyncio.TimeoutError:
            self._logger.error(
                "[TIMEOUT] %s action queue empty=%s waiting=%s trying_again=%s",
                self.player_id,
                self._env._action_queues[self.player_id].empty(),
                self._waiting.is_set(),
                self._trying_again.is_set(),
            )
            raise

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
        """Handle battle requests with teampreview prioritized (poke-env parity).

        Standard poke-env behavior does not wait for available moves during
        teampreview. To avoid deadlocks with sequential message processing,
        we short-circuit the teampreview branch before any waits.
        """

        # 受信した request の内容をデバッグ出力する
        self._logger.debug(
            "[DBG] %s last_request=%s",
            self.player_id,
            battle.last_request,
        )

        # --- Teampreview: handle first, no waiting ---
        if from_teampreview_request:
            self._logger.debug(
                "[DBG] %s send team preview request for %s",
                self.player_id,
                battle.battle_tag,
            )
            try:
                # Notify env and wait for team command from agent
                put_result = await asyncio.wait_for(
                    self._env._battle_queues[self.player_id].put(battle),
                    self._env.timeout,
                )
                if put_result is not None:
                    self._env._battle_queues[self.player_id].task_done()
                self._logger.debug(
                    "[DBG] %s waiting team preview action (qsize=%d)",
                    self.player_id,
                    self._env._action_queues[self.player_id].qsize(),
                )
                message = await asyncio.wait_for(
                    self._env._action_queues[self.player_id].get(),
                    self._env.timeout,
                )
                self._env._action_queues[self.player_id].task_done()
                # Normalize team selection command for local IPC: '/team 123' -> '/choose team 123'
                try:
                    if (
                        isinstance(message, str)
                        and message.startswith("/team ")
                        and getattr(self, "mode", None) == "local"
                    ):
                        message = "/choose team " + message[len("/team "):]
                except Exception:
                    pass
                self._logger.debug(
                    "[DBG] %s team preview message %s (%s)",
                    self.player_id,
                    message,
                    battle.player_username,
                )

                # Send immediately and return to keep pipeline flowing
                self._logger.debug(
                    "[DBG] %s send message '%s' to battle %s last = %s",
                    self.player_id,
                    message,
                    battle.battle_tag,
                    battle.last_request,
                )
                await self._send_battle_message_local_aware(battle, message)
                self._last_request = battle.last_request
                return
            except asyncio.TimeoutError:
                self._logger.error(
                    "[TIMEOUT] %s team preview action queue size=%d",
                    self.player_id,
                    self._env._action_queues[self.player_id].qsize(),
                )
                raise

        # --- Non-teampreview: original behavior ---
        # 同一の request が続いた場合は更新を待機する
        if battle.last_request is self._last_request:
            self._logger.debug(
                "[DBG] %s waiting last_request update", self.player_id
            )

            async def _wait_update() -> None:
                while battle.last_request is self._last_request:
                    await asyncio.sleep(0.1)

            try:
                await asyncio.wait_for(_wait_update(), timeout=self._env.timeout)
            except asyncio.TimeoutError as exc:
                self._logger.error(
                    "[TIMEOUT] %s last_request not updated", self.player_id
                )
                raise RuntimeError("Battle request not updated") from exc

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
        ):
            message = self.choose_default_move().message
        else:
            if maybe_default_order:
                self._trying_again.set()
            choice = self.choose_move(battle)
            if isinstance(choice, Awaitable):
                choice = await choice
            message = choice.message

        self._logger.debug(
            "[DBG] %s send message '%s' to battle %s last = %s",
            self.player_id,
            message,
            battle.battle_tag,
            battle.last_request,
        )
        await self._send_battle_message_local_aware(battle, message)
        # 処理した last_request を記録
        self._last_request = battle.last_request

    async def _create_battle(self, split_message: list[str]) -> AbstractBattle:
        """Override _create_battle to use CustomBattle class for fail/immune tracking.
        
        This method is based on the parent class implementation but uses CustomBattle
        instead of the default Battle class to enable tracking of -fail and -immune messages.
        """
        # We check that the battle has the correct format
        if split_message[1] == self._format and len(split_message) >= 2:
            # Battle initialisation
            battle_tag = "-".join(split_message)[1:]

            if battle_tag in self._battles:
                return self._battles[battle_tag]
            else:
                gen = GenData.from_format(self._format).gen
                if self.format_is_doubles:
                    from poke_env.environment.double_battle import DoubleBattle
                    battle: AbstractBattle = DoubleBattle(
                        battle_tag=battle_tag,
                        username=self.username,
                        logger=self.logger,
                        save_replays=self._save_replays,
                        gen=gen,
                    )
                else:
                    # Use CustomBattle instead of Battle
                    battle = CustomBattle(
                        battle_tag=battle_tag,
                        username=self.username,
                        logger=self.logger,
                        gen=gen,
                        save_replays=self._save_replays,
                    )

                # Add our team as teampreview_team, as part of battle initialisation
                if isinstance(self._team, ConstantTeambuilder):
                    battle.teampreview_team = set(
                        [
                            Pokemon(gen=gen, teambuilder=tb_mon)
                            for tb_mon in self._team.team
                        ]
                    )

                await self._battle_count_queue.put(None)
                if battle_tag in self._battles:
                    await self._battle_count_queue.get()
                    return self._battles[battle_tag]
                async with self._battle_start_condition:
                    self._battle_semaphore.release()
                    self._battle_start_condition.notify_all()
                    self._battles[battle_tag] = battle

                if self._start_timer_on_battle_start:
                    await self.ps_client.send_message("/timer on", battle.battle_tag)

                return battle
        else:
            self.logger.critical(
                "Unmanaged battle initialisation message received: %s", split_message
            )
