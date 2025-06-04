import numpy as np
import sys
from pathlib import Path

# Ensure the repository root is on the Python path so that src can be imported
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.env.pokemon_env import PokemonEnv


class DummyObserver:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def get_observation_dimension(self) -> int:
        return self.dim

    def observe(self, battle) -> np.ndarray:  # pragma: no cover - placeholder
        return np.zeros(self.dim, dtype=np.float32)


class DummyActionHelper:
    pass


class DummyOpponent:
    def reset_battles(self) -> None:  # pragma: no cover - placeholder
        pass


def test_observation_space_contains():
    dim = 5
    env = PokemonEnv(
        opponent_player=DummyOpponent(),
        state_observer=DummyObserver(dim),
        action_helper=DummyActionHelper(),
    )
    dummy_state = np.zeros(dim, dtype=np.float32)
    assert env.observation_space.contains(dummy_state)
