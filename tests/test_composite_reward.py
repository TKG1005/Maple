import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rewards import CompositeReward, RewardBase, KnockoutReward


class ConstReward(RewardBase):
    def __init__(self, value: float) -> None:
        self.value = value
        self.reset_called = False

    def reset(self, battle: object | None = None) -> None:
        self.reset_called = True

    def calc(self, battle: object) -> float:
        return float(self.value)


class DummyMon:
    def __init__(self, fainted: bool = False):
        self.fainted = fainted
        self.current_hp = 100


class DummyBattle:
    def __init__(self, my_mons: list[DummyMon], opp_mons: list[DummyMon]):
        self.team = {i: m for i, m in enumerate(my_mons)}
        self.opponent_team = {i: m for i, m in enumerate(opp_mons)}
        self.finished = False
        self.won = False


def test_composite_reward(tmp_path: Path) -> None:
    yaml_path = tmp_path / "reward.yaml"
    yaml_text = (
        "rewards:\n"
        "  a:\n"
        "    weight: 2.0\n"
        "    enabled: true\n"
        "  b:\n"
        "    weight: 0.5\n"
        "    enabled: true\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

    comp = CompositeReward(
        str(yaml_path),
        reward_map={"a": lambda: ConstReward(1.0), "b": lambda: ConstReward(1.0)},
    )
    comp.reset(None)
    assert all(r.reset_called for r in comp.rewards.values())
    reward = comp.calc(None)
    assert reward == 2.0 * 1.0 + 0.5 * 1.0
    assert comp.last_values["a"] == 1.0
    assert comp.last_values["b"] == 1.0


def test_composite_reward_knockout_from_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "reward.yaml"
    yaml_text = (
        "rewards:\n"
        "  knockout:\n"
        "    weight: 2.0\n"
        "    enabled: true\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

    comp = CompositeReward(str(yaml_path))

    battle = DummyBattle([DummyMon(False)], [DummyMon(False)])
    comp.reset(battle)
    battle.opponent_team[0].fainted = True
    reward = comp.calc(battle)

    assert reward == 2.0 * KnockoutReward.ENEMY_KO_BONUS
    assert comp.last_values["knockout"] == KnockoutReward.ENEMY_KO_BONUS


def test_composite_reward_turn_penalty_from_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "reward.yaml"
    yaml_text = (
        "rewards:\n"
        "  turn_penalty:\n"
        "    weight: 2.0\n"
        "    enabled: true\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

    comp = CompositeReward(str(yaml_path))

    class Battle:
        def __init__(self, turn: int = 1) -> None:
            self.turn = turn

    battle = Battle()
    comp.reset(battle)
    reward = comp.calc(battle)

    assert reward == 2.0 * (-0.01)
    assert comp.last_values["turn_penalty"] == -0.01


def test_composite_reward_fail_immune_from_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "reward.yaml"
    yaml_text = (
        "rewards:\n"
        "  fail_immune:\n"
        "    weight: 2.0\n"
        "    enabled: true\n"
    )
    yaml_path.write_text(yaml_text, encoding="utf-8")

    comp = CompositeReward(str(yaml_path))

    class Battle:
        def __init__(self, last_fail_action: bool = False, last_immune_action: bool = False) -> None:
            self.last_fail_action = last_fail_action
            self.last_immune_action = last_immune_action

    # Test no penalty
    battle = Battle(last_fail_action=False, last_immune_action=False)
    comp.reset(battle)
    reward = comp.calc(battle)
    assert reward == 0.0
    assert comp.last_values["fail_immune"] == 0.0

    # Test penalty with fail action
    battle.last_fail_action = True
    reward = comp.calc(battle)
    assert reward == 2.0 * (-0.02)
    assert comp.last_values["fail_immune"] == -0.02

    # Test penalty with immune action
    battle.last_fail_action = False
    battle.last_immune_action = True
    reward = comp.calc(battle)
    assert reward == 2.0 * (-0.02)
    assert comp.last_values["fail_immune"] == -0.02
