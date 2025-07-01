import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rewards import TurnPenaltyReward


class DummyBattle:
    def __init__(self, turn: int) -> None:
        self.turn = turn


def test_turn_penalty_per_turn():
    battle = DummyBattle(turn=1)
    r = TurnPenaltyReward(penalty=-0.1)
    r.reset(battle)

    assert r(battle) == -0.1
    # 同じターン内ではペナルティは重複しない
    assert r(battle) == 0.0

    battle.turn = 2
    assert r(battle) == -0.1
    assert r.turn_count == 2
    # calc も同一のロジックを通る
    assert r.calc(battle) == 0.0
