import sys
from pathlib import Path
import numpy as np

# Add repository root to path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.env.pokemon_env import PokemonEnv
from src.agents.my_simple_player import MySimplePlayer


class DummyObserver:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def get_observation_dimension(self) -> int:
        return self.dim

    def observe(self, battle) -> np.ndarray:  # pragma: no cover - simple default
        return np.zeros(self.dim, dtype=np.float32)


class DummyActionHelper:
    pass


def main() -> None:
    opponent = MySimplePlayer(battle_format="gen9randombattle")
    env = PokemonEnv(
        opponent_player=opponent,
        state_observer=DummyObserver(5),
        action_helper=DummyActionHelper(),
    )
    obs1, info1 = env.reset()
    print("First reset info:", info1)
    obs2, info2 = env.reset()
    print("Second reset info:", info2)


if __name__ == "__main__":
    main()
