"""Random player Agent that sends chosen action index via an async queue."""

from __future__ import annotations

import asyncio
from typing import Any

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
        self._battle_ready: dict[str, asyncio.Event] = {}

    async def _create_battle(self, *args: Any, **kwargs: Any) -> Any:
        """Wrap ``Player._create_battle`` and signal completion."""
        battle_tag = None
        if args:
            battle_tag = args[0]
        if 'battle_tag' in kwargs:
            battle_tag = kwargs['battle_tag']

        event = asyncio.Event()
        if battle_tag is not None:
            self._battle_ready[battle_tag] = event

        battle = await super()._create_battle(*args, **kwargs)

        if battle_tag is None and hasattr(battle, "battle_tag"):
            battle_tag = getattr(battle, "battle_tag")
            self._battle_ready[battle_tag] = event

        event.set()
        return battle

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

