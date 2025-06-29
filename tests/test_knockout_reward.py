import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rewards import KnockoutReward


class DummyMon:
    def __init__(self, current_hp: int, fainted: bool = False) -> None:
        self.current_hp = current_hp
        self.max_hp = current_hp
        self.fainted = fainted


class DummyBattle:
    def __init__(self, team, opp_team) -> None:
        self.team = {i: m for i, m in enumerate(team)}
        self.opponent_team = {i: m for i, m in enumerate(opp_team)}


def test_knockout_reset_initializes_state():
    mons = [DummyMon(100), DummyMon(50)]
    opps = [DummyMon(80)]
    battle = DummyBattle(mons, opps)
    r = KnockoutReward()
    r.reset(battle)

    assert set(r.my_ids.keys()) == {id(m) for m in mons}
    assert set(r.opp_ids.keys()) == {id(m) for m in opps}
    for m in mons:
        assert r.prev_my_hp[id(m)] == m.current_hp
        assert r.prev_my_alive[id(m)] is True
    for m in opps:
        assert r.prev_opp_hp[id(m)] == m.current_hp
        assert r.prev_opp_alive[id(m)] is True

    r.reset(None)
    assert r.my_ids == {}
    assert r.opp_ids == {}
    assert r.prev_my_hp == {}
