import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rewards import KnockoutReward


class DummyMon:
    def __init__(self, current_hp=0, fainted=False):
        self.current_hp = current_hp
        self.fainted = fainted


class DummyBattle:
    def __init__(self, my_mons, opp_mons):
        self.team = {i: m for i, m in enumerate(my_mons)}
        self.opponent_team = {i: m for i, m in enumerate(opp_mons)}


def test_knockout_reset_records_initial_state():
    my1 = DummyMon(50, False)
    my2 = DummyMon(0, True)
    opp1 = DummyMon(60, False)
    battle = DummyBattle([my1, my2], [opp1])

    r = KnockoutReward()
    r.reset(battle)

    assert r.prev_my_hp[id(my1)] == 50
    assert r.prev_my_hp[id(my2)] == 0
    assert r.prev_my_alive[id(my1)] is True
    assert r.prev_my_alive[id(my2)] is False
    assert r.prev_opp_hp[id(opp1)] == 60
    assert r.prev_opp_alive[id(opp1)] is True
    assert r.enemy_ko == 0
    assert r.self_ko == 0


def test_knockout_calc_counts_new_faints():
    my1 = DummyMon(50, False)
    my2 = DummyMon(40, False)
    opp1 = DummyMon(60, False)
    battle = DummyBattle([my1, my2], [opp1])

    r = KnockoutReward()
    r.reset(battle)

    my1.current_hp = 0
    my1.fainted = True
    opp1.current_hp = 0
    opp1.fainted = True

    reward = r.calc(battle)

    assert reward == 0.0
    assert r.self_ko == 1
    assert r.enemy_ko == 1

    # Calling again without additional faint should not change counts
    reward2 = r.calc(battle)
    assert reward2 == 0.0
    assert r.self_ko == 1
    assert r.enemy_ko == 1
