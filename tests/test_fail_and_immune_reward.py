import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rewards import FailAndImmuneReward


class DummyBattle:
    def __init__(self, last_fail_action: bool = False, last_immune_action: bool = False) -> None:
        self.last_fail_action = last_fail_action
        self.last_immune_action = last_immune_action


def test_fail_and_immune_penalty():
    """Test that FailAndImmuneReward gives penalty when either flag is True."""
    # Test no penalty when no invalid action
    battle = DummyBattle(last_fail_action=False, last_immune_action=False)
    reward = FailAndImmuneReward()
    reward.reset(battle)
    
    assert reward.calc(battle) == 0.0
    
    # Test penalty when fail action
    battle.last_fail_action = True
    assert reward.calc(battle) == -0.02
    
    # Reset fail, test penalty when immune action
    battle.last_fail_action = False
    battle.last_immune_action = True
    assert reward.calc(battle) == -0.02
    
    # Test penalty when both actions
    battle.last_fail_action = True
    battle.last_immune_action = True
    assert reward.calc(battle) == -0.02
    
    # Test with custom penalty
    reward_custom = FailAndImmuneReward(penalty=-0.05)
    reward_custom.reset(battle)
    assert reward_custom.calc(battle) == -0.05


def test_fail_and_immune_no_attribute():
    """Test that FailAndImmuneReward handles battles without action flags."""
    battle = object()  # No action flags
    reward = FailAndImmuneReward()
    reward.reset(battle)
    
    assert reward.calc(battle) == 0.0


def test_fail_and_immune_reset():
    """Test that reset doesn't affect the reward behavior since it's stateless."""
    battle = DummyBattle(last_fail_action=True, last_immune_action=True)
    reward = FailAndImmuneReward()
    
    # Before reset
    assert reward.calc(battle) == -0.02
    
    # After reset
    reward.reset(battle)
    assert reward.calc(battle) == -0.02
    
    # Reset with None
    reward.reset(None)
    assert reward.calc(battle) == -0.02


def test_individual_flags():
    """Test individual flag behavior."""
    reward = FailAndImmuneReward()
    
    # Only fail action
    battle_fail = DummyBattle(last_fail_action=True, last_immune_action=False)
    assert reward.calc(battle_fail) == -0.02
    
    # Only immune action
    battle_immune = DummyBattle(last_fail_action=False, last_immune_action=True)
    assert reward.calc(battle_immune) == -0.02
    
    # Neither
    battle_none = DummyBattle(last_fail_action=False, last_immune_action=False)
    assert reward.calc(battle_none) == 0.0