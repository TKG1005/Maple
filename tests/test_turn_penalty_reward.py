import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rewards import TurnPenaltyReward


def test_turn_penalty_returns_constant_value():
    r = TurnPenaltyReward(penalty=-0.1)
    r.reset(None)
    assert r(None) == -0.1
    assert r(None) == -0.1
    assert r.turn_count == 2
    assert r.calc(None) == -0.1
