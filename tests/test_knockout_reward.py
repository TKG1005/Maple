import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rewards import KnockoutReward


class DummyMon:
    def __init__(self, fainted=False):
        self.fainted = fainted
        self.current_hp = 100



class DummyBattle:
    def __init__(self, my_mons, opp_mons):
        self.team = {i: m for i, m in enumerate(my_mons)}
        self.opponent_team = {i: m for i, m in enumerate(opp_mons)}

        self.finished = False
        self.won = False


def test_knockout_enemy_ko_reward():
    battle = DummyBattle([DummyMon(False)], [DummyMon(False)])
    r = KnockoutReward()
    r.reset(battle)
    battle.opponent_team[0].fainted = True
    assert r.calc(battle) == r.ENEMY_KO_BONUS


def test_knockout_self_ko_penalty():
    battle = DummyBattle([DummyMon(False)], [DummyMon(False)])
    r = KnockoutReward()
    r.reset(battle)
    battle.team[0].fainted = True
    assert r.calc(battle) == r.SELF_KO_PENALTY

