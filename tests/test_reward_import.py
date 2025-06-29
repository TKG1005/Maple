import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rewards import RewardBase


def test_reward_base_import() -> None:
    assert hasattr(RewardBase, "reset")
    assert hasattr(RewardBase, "calc")
