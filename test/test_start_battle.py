"""Manual test for the action queue behaviour."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import numpy as np


# Ensure src path is available
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.env.pokemon_env import PokemonEnv


class DummyObserver:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def get_observation_dimension(self) -> int:
        return self.dim

    def observe(self, battle: Any) -> np.ndarray:  # pragma: no cover - simple stub
        return np.zeros(self.dim, dtype=np.float32)


class DummyActionHelper:
    def action_index_to_order(self, player: Any, battle: Any, idx: int) -> str:
        """Return a simple string so that the test is self contained."""
        return f"order_{idx}"


async def _run() -> None:
    env = PokemonEnv(
        opponent_player=object(),
        state_observer=DummyObserver(1),
        action_helper=DummyActionHelper(),
    )

    class EnvPlayer:
        async def choose_move(self, battle: Any) -> str:
            idx = await env._action_queue.get()
            return env.action_helper.action_index_to_order(self, battle, idx)

    env._env_player = EnvPlayer()

    env._action_queue.put_nowait(5)
    order = await env._env_player.choose_move(None)
    print("choose_move returned", order)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

