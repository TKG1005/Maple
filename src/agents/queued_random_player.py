"""Random player Agent that sends chosen action index via an async queue."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
import random
import orjson
import time
from poke_env.player import Player
from poke_env.environment.battle import Battle
from poke_env.environment.abstract_battle import AbstractBattle
from typing import Any, Awaitable, Dict, List, Optional, Union

from src.action.action_helper import get_available_actions, action_index_to_order


class QueuedRandomPlayer(Player):
    """Choose random legal moves and report index through an async queue."""

    def __init__(
        self,
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._rng = np.random.default_rng(seed)



    def teampreview(self, battle: Battle) -> str:  # pragma: no cover - runtime
        """Randomly select three Pokémon from 1–6."""
        indices = self._rng.choice(range(1, 7), size=3, replace=False)
        order = "/team " + "".join(str(i) for i in indices)
        print(f'[DBG:queued_random_player.py player={battle.player_username}] order = {order}')
        return order


    async def choose_move(self, battle) -> Any:  # pragma: no cover - runtime behaviour
        """ターンごとの行動を決定する。"""
        mask, mapping = get_available_actions(battle)
        print(f"[DBG:queued_random_player.py]mask = {mask}, mapping = {mapping} ")
        if mapping:
            action_idx = int(self._rng.choice(list(mapping.keys())))
            return action_index_to_order(self, battle, action_idx)
        return self.choose_random_move(battle)

    async def _handle_battle_message(self, split_messages: List[List[str]]): #親クラスの_handle_battle_messageをオーバーライド
        """Handles a battle message.

        :param split_message: The received battle message.
        :type split_message: str
        """
        recv_ts = time.time()
        print(f"[DBG] _handle_battle_message start t={recv_ts:.6f} msg={split_messages}")

        # Battle messages can be multiline
        before_battle_ts = time.time()
        if (
            len(split_messages) > 1
            and len(split_messages[1]) > 1
            and split_messages[1][1] == "init"
        ):
            battle_info = split_messages[0][0].split("-")
            battle = await self._create_battle(battle_info)
        else:
            battle = await self._get_battle(split_messages[0][0])
        after_battle_ts = time.time()
        print(
            f"[DBG] battle obtained t={after_battle_ts:.6f} (dt={after_battle_ts-before_battle_ts:.6f}) teampreview={battle.teampreview} v"
        )
        
        for split_message in split_messages[1:]:
            start_ts = time.time()
            print(f"[DBG] handle sub msg start t={start_ts:.6f} ")
            if len(split_message) <= 1:
                continue
            elif split_message[1] == "":
                battle.parse_message(split_message)
                print(f"[DBG] parsed message t={time.time():.6f}")
            elif split_message[1] in self.MESSAGES_TO_IGNORE:
                pass
            elif split_message[1] == "request":
                if split_message[2]:
                    request = orjson.loads(split_message[2])
                    battle.parse_request(request)
                    print(f"[DBG] parsed request t={time.time():.6f} teampreview={battle.teampreview}")
                    if battle._wait:
                        self._waiting.set()
                    if battle.move_on_next_request:
                        if split_message[2]["teamPreview"]:
                            await self._handle_battle_request(battle,from_teampreview_request=True)
                        else:
                            await self._handle_battle_request(battle)
                        battle.move_on_next_request = False
            elif split_message[1] == "win" or split_message[1] == "tie":
                if split_message[1] == "win":
                    battle.won_by(split_message[2])
                else:
                    battle.tied()
                await self._battle_count_queue.get()
                self._battle_count_queue.task_done()
                self._battle_finished_callback(battle)
                async with self._battle_end_condition:
                    self._battle_end_condition.notify_all()
            elif split_message[1] == "error":
                self.logger.log(
                    25, "Error message received: %s", "|".join(split_message)
                )
                if split_message[2].startswith(
                    "[Invalid choice] Sorry, too late to make a different move"
                ):
                    if battle.trapped:
                        self._trying_again.set()
                        await self._handle_battle_request(battle)
                elif split_message[2].startswith(
                    "[Unavailable choice] Can't switch: The active Pokémon is "
                    "trapped"
                ) or split_message[2].startswith(
                    "[Invalid choice] Can't switch: The active Pokémon is trapped"
                ):
                    battle.trapped = True
                    self._trying_again.set()
                    await self._handle_battle_request(battle)
                elif split_message[2].startswith("[Invalid choice] Can't pass: "):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't switch: You can't switch to an active "
                    "Pokémon"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't switch: You can't switch to a fainted "
                    "Pokémon"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: Invalid target for"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: You can't choose a target for"
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: "
                ) and split_message[2].endswith("needs a target"):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif (
                    split_message[2].startswith("[Invalid choice] Can't move: Your")
                    and " doesn't have a move matching " in split_message[2]
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Incomplete choice: "
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Unavailable choice]"
                ) and split_message[2].endswith("is disabled"):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith("[Invalid choice]") and split_message[
                    2
                ].endswith("is disabled"):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: You sent more choices than unfainted"
                    " Pokémon."
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                elif split_message[2].startswith(
                    "[Invalid choice] Can't move: You can only Terastallize once per battle."
                ):
                    await self._handle_battle_request(battle, maybe_default_order=True)
                else:
                    self.logger.critical("Unexpected error message: %s", split_message)
            elif split_message[1] == "turn":
                battle.parse_message(split_message)
                await self._handle_battle_request(battle)
            elif split_message[1] == "teampreview":
                battle.parse_message(split_message)
                await self._handle_battle_request(battle, from_teampreview_request=True)
            elif split_message[1] == "bigerror":
                self.logger.warning("Received 'bigerror' message: %s", split_message)
            elif split_message[1] == "uhtml" and split_message[2] == "otsrequest":
                await self._handle_ots_request(battle.battle_tag)
            else:
                battle.parse_message(split_message)
