"""Random player Agent that sends chosen action index via an async queue."""

from __future__ import annotations

import asyncio
from typing import Any
import json

import numpy as np
from poke_env.player import Player
from poke_env.environment.battle import Battle

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
        event = self._battle_ready.get(battle.battle_tag)
        if event is not None and not event.is_set():
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(event.wait())
            else:
                fut = asyncio.run_coroutine_threadsafe(event.wait(), loop)
                fut.result()

        indices = self._rng.choice(range(1, 7), size=3, replace=False)
        order = "/team " + "".join(str(i) for i in indices)
        print(f'[DBG:queued_random_player.py] order = {order}')
        return order


    async def choose_move(self, battle) -> Any:  # pragma: no cover - runtime behaviour
        """ターンごとの行動を決定する。"""
        event = self._battle_ready.get(battle.battle_tag)
        if event is not None and not event.is_set():
            await event.wait()

        # チームプレビュー判定
        print(battle.teampreview)
        if battle.teampreview:
            return self.teampreview(battle)
        
        mask, mapping = get_available_actions(battle)
        print(f"[DBG:queued_random_player.py]mask = {mask}, mapping = {mapping}")
        if mapping:
            action_idx = int(self._rng.choice(list(mapping.keys())))
            return action_index_to_order(self, battle, action_idx)
        return self.choose_random_move(battle)

    async def _handle_battle_message(self, message: str) -> None:
        """Handle battle messages and process team preview requests."""
        if message.startswith(">") and "|request|" in message:
            battle_tag, rest = message[1:].split("|", 1)
            if rest.startswith("request|"):
                json_str = rest[len("request|") :]
                try:
                    request = json.loads(json_str)
                except Exception:
                    request = None
                battle = await self._create_battle(battle_tag)
                if request and request.get("teamPreview") and getattr(battle, "teampreview", False):
                    self.from_teampreview_request = True
                    await self._handle_battle_request(battle, request)
                    self.from_teampreview_request = False
                    return

        await super()._handle_battle_message(message)

