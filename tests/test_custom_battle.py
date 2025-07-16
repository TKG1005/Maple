"""Tests for custom battle class that intercepts fail and immune messages."""

from __future__ import annotations

import logging
import pytest
from src.env.custom_battle import CustomBattle
from src.rewards.fail_and_immune import FailAndImmuneReward


class TestCustomBattle:
    """Test suite for CustomBattle message interception functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.battle = CustomBattle(
            battle_tag="test-battle",
            username="test_user",
            logger=logging.getLogger(__name__),
            gen=9,
            save_replays=False,
        )
    
    def test_initial_state(self):
        """Test that flags are initially False."""
        assert self.battle.last_fail_action is False
        assert self.battle.last_immune_action is False
    
    def test_fail_message_detection(self):
        """Test that -fail messages are correctly detected."""
        fail_message = ["", "-fail", "p1a: Pikachu", "Thunder Wave", "[notarget]"]
        self.battle.parse_message(fail_message)
        assert self.battle.last_fail_action is True
        assert self.battle.last_immune_action is False
    
    def test_immune_message_detection(self):
        """Test that -immune messages are correctly detected."""
        immune_message = ["", "-immune", "p2a: Garchomp", "[from] ability: Limber"]
        self.battle.parse_message(immune_message)
        assert self.battle.last_fail_action is False
        assert self.battle.last_immune_action is True
    
    def test_both_flags_can_be_set(self):
        """Test that both flags can be set simultaneously."""
        fail_message = ["", "-fail", "p1a: Pikachu", "Thunder Wave", "[notarget]"]
        immune_message = ["", "-immune", "p2a: Garchomp", "[from] ability: Limber"]
        
        self.battle.parse_message(fail_message)
        self.battle.parse_message(immune_message)
        
        assert self.battle.last_fail_action is True
        assert self.battle.last_immune_action is True
    
    def test_multiple_fail_messages(self):
        """Test that multiple -fail messages don't cause issues."""
        fail_message = ["", "-fail", "p1a: Pikachu", "Thunder Wave", "[notarget]"]
        
        self.battle.parse_message(fail_message)
        self.battle.parse_message(fail_message)
        
        assert self.battle.last_fail_action is True
    
    def test_multiple_immune_messages(self):
        """Test that multiple -immune messages don't cause issues."""
        immune_message = ["", "-immune", "p2a: Garchomp", "[from] ability: Limber"]
        
        self.battle.parse_message(immune_message)
        self.battle.parse_message(immune_message)
        
        assert self.battle.last_immune_action is True
    
    def test_other_messages_dont_affect_flags(self):
        """Test that other message types don't affect the flags."""
        # Set flags to True first by triggering actual messages
        fail_message = ["", "-fail", "p1a: Pikachu", "Thunder Wave", "[notarget]"]
        immune_message = ["", "-immune", "p2a: Garchomp", "[from] ability: Limber"]
        self.battle.parse_message(fail_message)
        self.battle.parse_message(immune_message)
        
        # Verify flags are set
        assert self.battle.last_fail_action is True
        assert self.battle.last_immune_action is True
        
        # Process other message types
        other_messages = [
            ["", "-damage", "p1a: Pikachu", "50/100"],
            ["", "move", "p1a: Pikachu", "Thunder Wave"],
            ["", "-heal", "p1a: Pikachu", "100/100"],
        ]
        
        for message in other_messages:
            self.battle.parse_message(message)
        
        # Flags should remain True
        assert self.battle.last_fail_action is True
        assert self.battle.last_immune_action is True
    
    def test_turn_reset_logic(self):
        """Test the turn reset logic for flags."""
        # Set flags to True first by triggering actual messages
        fail_message = ["", "-fail", "p1a: Pikachu", "Thunder Wave", "[notarget]"]
        immune_message = ["", "-immune", "p2a: Garchomp", "[from] ability: Limber"]
        self.battle.parse_message(fail_message)
        self.battle.parse_message(immune_message)
        
        # Verify flags are set
        assert self.battle.last_fail_action is True
        assert self.battle.last_immune_action is True
        
        # Test reset functionality directly
        self.battle.reset_invalid_action()
        
        # Flags should be reset to False
        assert self.battle.last_fail_action is False
        assert self.battle.last_immune_action is False


class TestFailImmuneRewardIntegration:
    """Test suite for FailAndImmuneReward integration with CustomBattle."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.battle = CustomBattle(
            battle_tag="test-battle",
            username="test_user",
            logger=logging.getLogger(__name__),
            gen=9,
            save_replays=False,
        )
        self.reward = FailAndImmuneReward(penalty=-0.05)
    
    def test_no_fail_immune_returns_zero(self):
        """Test that no fail/immune actions return 0.0 reward."""
        reward_value = self.reward.calc(self.battle)
        assert reward_value == 0.0
    
    def test_fail_action_returns_penalty(self):
        """Test that fail action returns penalty."""
        fail_message = ["", "-fail", "p1a: Pikachu", "Thunder Wave", "[notarget]"]
        self.battle.parse_message(fail_message)
        reward_value = self.reward.calc(self.battle)
        assert reward_value == -0.05
    
    def test_immune_action_returns_penalty(self):
        """Test that immune action returns penalty."""
        immune_message = ["", "-immune", "p2a: Garchomp", "[from] ability: Limber"]
        self.battle.parse_message(immune_message)
        reward_value = self.reward.calc(self.battle)
        assert reward_value == -0.05
    
    def test_both_actions_return_single_penalty(self):
        """Test that both fail and immune actions return single penalty."""
        fail_message = ["", "-fail", "p1a: Pikachu", "Thunder Wave", "[notarget]"]
        immune_message = ["", "-immune", "p2a: Garchomp", "[from] ability: Limber"]
        
        self.battle.parse_message(fail_message)
        self.battle.parse_message(immune_message)
        
        reward_value = self.reward.calc(self.battle)
        assert reward_value == -0.05
    
    def test_custom_penalty_value(self):
        """Test that custom penalty values work correctly."""
        custom_reward = FailAndImmuneReward(penalty=-0.1)
        
        fail_message = ["", "-fail", "p1a: Pikachu", "Thunder Wave", "[notarget]"]
        self.battle.parse_message(fail_message)
        
        reward_value = custom_reward.calc(self.battle)
        assert reward_value == -0.1
    
    def test_reward_reset_functionality(self):
        """Test that reward reset functionality works."""
        # Set flags and verify penalty
        fail_message = ["", "-fail", "p1a: Pikachu", "Thunder Wave", "[notarget]"]
        self.battle.parse_message(fail_message)
        
        reward_value = self.reward.calc(self.battle)
        assert reward_value == -0.05
        
        # Reset should not change the battle state (it's stateless)
        self.reward.reset(self.battle)
        reward_value = self.reward.calc(self.battle)
        assert reward_value == -0.05