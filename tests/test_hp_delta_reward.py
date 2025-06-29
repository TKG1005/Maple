import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rewards import HPDeltaReward

class DummyMon:
    def __init__(self, current_hp, max_hp):
        self.current_hp = current_hp
        self.max_hp = max_hp

class DummyBattle:
    def __init__(self, mons, finished=False):
        self.team = {i: m for i, m in enumerate(mons)}
        self.opponent_team = {}
        self.finished = finished
        self.won = False


def test_hp_delta_bonus_applied():
    mon = DummyMon(80, 100)
    battle = DummyBattle([mon], finished=False)
    r = HPDeltaReward()
    r.reset(battle)
    battle.finished = True
    assert r.calc(battle) == r.BONUS_VALUE


def test_hp_delta_bonus_not_applied_when_low_hp():
    mon = DummyMon(40, 100)
    battle = DummyBattle([mon], finished=False)
    r = HPDeltaReward()
    r.reset(battle)
    battle.finished = True
    assert r.calc(battle) == 0.0
