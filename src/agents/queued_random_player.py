"""Random player Agent that sends chosen action index via an async queue."""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np
from poke_env.player import Player

from src.action.action_helper import get_available_actions, action_index_to_order


class QueuedRandomPlayer(Player):
    """Choose random legal moves and report index through an async queue."""

    def __init__(
        self,
        action_queue: asyncio.Queue[int],
        *,
        seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._queue = action_queue
        self._rng = np.random.default_rng(seed)

    async def choose_team(self, battle: Any) -> str:  # pragma: no cover - runtime
        """Select the first three Pokémon when team preview occurs."""
        return "123"

    async def choose_move(self, battle) -> Any:  # pragma: no cover - runtime behaviour
        mask, mapping = get_available_actions(battle)
        print(f"[DBG:queued_random_player.py]mask = {mask}, mapping = {mapping}")
        if mapping:
            action_idx = int(self._rng.choice(list(mapping.keys())))
            await self._queue.put(action_idx)
            return action_index_to_order(self, battle, action_idx)
        return self.choose_random_move(battle)

