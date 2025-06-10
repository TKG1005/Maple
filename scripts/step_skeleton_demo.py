from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.env.pokemon_env import PokemonEnv
from src.agents.MapleAgent import MapleAgent


class DummyObserver:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def get_observation_dimension(self) -> int:
        return self.dim

    def observe(self, battle) -> np.ndarray:
        return np.zeros(self.dim, dtype=np.float32)


class DummyActionHelper:
    def action_index_to_order(self, *args, **kwargs):
        return None


class DummyOpponent:
    def reset_battles(self) -> None:
        pass


class RandomAgent(MapleAgent):
    def select_action(self, observation: object) -> int:
        return self.env.action_space.sample()


def main() -> None:
    env = PokemonEnv(
        opponent_player=DummyOpponent(),
        state_observer=DummyObserver(5),
        action_helper=DummyActionHelper(),
    )
    agent = RandomAgent(env)

    action = agent.select_action(None)
    result = env.step(action)
    print("Step result:", result)


if __name__ == "__main__":
    main()

