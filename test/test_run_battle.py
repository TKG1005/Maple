from __future__ import annotations

"""Run one PokemonEnv battle and print result for manual inspection."""

import asyncio
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pytest.importorskip("numpy")
pytest.importorskip("poke_env")

from src.env import PokemonEnv
from src.state.state_observer import StateObserver
from src.action import action_helper
from src.agents.queued_random_player import QueuedRandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration


async def main() -> None:
    team_file = ROOT / "config" / "my_team.txt"
    try:
        team = team_file.read_text()
    except OSError:
        team = None

    opponent = QueuedRandomPlayer(
        asyncio.Queue(),
        battle_format="gen9ou",
        server_configuration=LocalhostServerConfiguration,
        team=team,
    )
    observer = StateObserver("config/state_spec.yml")
    env = PokemonEnv(opponent, observer, action_helper)

    obs, info = env.reset()
    terminated = truncated = False
    total_reward = 0.0
    turns = 0

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        turns += 1
        await asyncio.sleep(0)

    print(
        f"terminated={terminated} truncated={truncated} reward={total_reward} turns={turns}"
    )
    env.close()


if __name__ == "__main__":
    asyncio.run(main())
