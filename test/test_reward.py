import sys
from pathlib import Path
import numpy as np

# Ensure src can be imported
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.env.pokemon_env import PokemonEnv


class DummyObserver:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def get_observation_dimension(self) -> int:
        return self.dim

    def observe(self, battle) -> np.ndarray:
        return np.zeros(self.dim, dtype=np.float32)


class DummyActionHelper:
    pass


class DummyOpponent:
    pass


class DummyBattle:
    def __init__(self, finished: bool, won: bool) -> None:
        self.finished = finished
        self.won = won


def _make_env() -> PokemonEnv:
    return PokemonEnv(
        opponent_player=DummyOpponent(),
        state_observer=DummyObserver(1),
        action_helper=DummyActionHelper(),
    )


def test_calc_reward_win():
    env = _make_env()
    battle = DummyBattle(finished=True, won=True)
    assert env._calc_reward(battle) == 1.0


def test_calc_reward_loss():
    env = _make_env()
    battle = DummyBattle(finished=True, won=False)
    assert env._calc_reward(battle) == -1.0


def test_calc_reward_ongoing():
    env = _make_env()
    battle = DummyBattle(finished=False, won=False)
    assert env._calc_reward(battle) == 0.0
