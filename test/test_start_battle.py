import asyncio
import sys
from pathlib import Path
import numpy as np


# Ensure src path is available
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.env.pokemon_env import PokemonEnv
from src.agents.my_simple_player import MySimplePlayer

#------------------チームファイルを読み込み-----------------
TEAM_FILE = ROOT_DIR / "config" / "my_team.txt"
try:
    TEAM = TEAM_FILE.read_text()
except OSError:
    TEAM = None
    print("[DBG]team open error")
    
    
class DummyObserver:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def get_observation_dimension(self) -> int:
        return self.dim

    def observe(self, battle) -> np.ndarray:
        return np.zeros(self.dim, dtype=np.float32)


class DummyActionHelper:
    pass


async def _run() -> None:


    opponent = MySimplePlayer(battle_format="gen9ou",team=TEAM)
    env = PokemonEnv(
        opponent_player=opponent,
        state_observer=DummyObserver(5),
        action_helper=DummyActionHelper(),
    )
    obs, info = env.reset()
    print("reset returned", info)
    # step が例外なく 5 要素タプルを返すか簡易チェック
    action = env.action_space.sample()
    result = env.step(action)
    print("step returned", result)
    await asyncio.sleep(1)
    env.close()


def main() -> None:
    """Entry point for manual testing."""
    asyncio.run(_run())


if __name__ == "__main__":
    main()
