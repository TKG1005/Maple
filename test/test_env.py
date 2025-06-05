from __future__ import annotations

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    import numpy as np  # noqa: F401
except Exception as exc:  # pragma: no cover - dependency check
    import pytest

    pytest.skip("numpy not available", allow_module_level=True)

from src.env import PokemonEnv
from src.state.state_observer import StateObserver
from src.action import action_helper


def main() -> None:
    """Run one episode and print the result."""
    try:
        from poke_env.player.random_player import RandomPlayer
    except Exception as exc:  # pragma: no cover - runtime check
        print("poke_env is not available:", exc)
        return

    opponent = RandomPlayer(battle_format="gen9ou")
    observer = StateObserver(str(Path("config") / "state_spec.yml"))
    env = PokemonEnv(opponent, observer, action_helper)

    obs, info = env.reset()
    terminated = truncated = False
    total_reward = 0.0

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    print(
        f"terminated={terminated} truncated={truncated} turns={info.get('turn')} reward={total_reward}"
    )
    env.close()


if __name__ == "__main__":
    main()
