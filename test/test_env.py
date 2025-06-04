import numpy as np
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
