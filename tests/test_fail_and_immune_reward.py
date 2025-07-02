import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.rewards import FailAndImmuneReward


class DummyBattle:
    def __init__(self, last_invalid_action: bool = False) -> None:
        self.last_invalid_action = last_invalid_action


def test_fail_and_immune_penalty():
    """Test that FailAndImmuneReward gives penalty when last_invalid_action is True."""
    # Test no penalty when no invalid action
    battle = DummyBattle(last_invalid_action=False)
    reward = FailAndImmuneReward()
    reward.reset(battle)
    
    assert reward.calc(battle) == 0.0
    
    # Test penalty when invalid action
    battle.last_invalid_action = True
    assert reward.calc(battle) == -0.02
    
    # Test with custom penalty
    reward_custom = FailAndImmuneReward(penalty=-0.05)
    reward_custom.reset(battle)
    assert reward_custom.calc(battle) == -0.05


def test_fail_and_immune_no_attribute():
    """Test that FailAndImmuneReward handles battles without last_invalid_action attribute."""
    battle = object()  # No last_invalid_action attribute
    reward = FailAndImmuneReward()
    reward.reset(battle)
    
    assert reward.calc(battle) == 0.0


def test_fail_and_immune_reset():
    """Test that reset doesn't affect the reward behavior since it's stateless."""
    battle = DummyBattle(last_invalid_action=True)
    reward = FailAndImmuneReward()
    
    # Before reset
    assert reward.calc(battle) == -0.02
    
    # After reset
    reward.reset(battle)
    assert reward.calc(battle) == -0.02
    
    # Reset with None
    reward.reset(None)
    assert reward.calc(battle) == -0.02