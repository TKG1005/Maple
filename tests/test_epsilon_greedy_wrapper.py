from __future__ import annotations

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock

import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agents.action_wrapper import EpsilonGreedyWrapper
from src.agents.MapleAgent import MapleAgent
from src.agents.RLAgent import RLAgent
from src.env.pokemon_env import PokemonEnv


class MockAgent(MapleAgent):
    """Mock agent for testing purposes."""
    
    def __init__(self, env, return_probs=True):
        super().__init__(env)
        self.return_probs = return_probs
        self.action_calls = []
        self._test_return_value = None  # For testing specific return values
    
    def set_test_return_value(self, value):
        """Set a specific return value for testing."""
        self._test_return_value = value
    
    def select_action(self, observation, action_mask):
        self.action_calls.append((observation, action_mask))
        
        # If test return value is set, use that
        if self._test_return_value is not None:
            return self._test_return_value
        
        if self.return_probs:
            # Return uniform probabilities over valid actions
            valid_indices = [i for i, flag in enumerate(action_mask) if flag]
            probs = np.zeros(len(action_mask), dtype=np.float32)
            if valid_indices:
                probs[valid_indices] = 1.0 / len(valid_indices)
            return probs
        else:
            # Return first valid action index
            valid_indices = [i for i, flag in enumerate(action_mask) if flag]
            return valid_indices[0] if valid_indices else 0


@pytest.fixture
def mock_env():
    """Create a mock environment."""
    env = Mock(spec=PokemonEnv)
    env.rng = np.random.default_rng(42)  # Fixed seed for reproducible tests
    return env


@pytest.fixture
def mock_agent(mock_env):
    """Create a mock agent."""
    return MockAgent(mock_env)


@pytest.fixture
def epsilon_wrapper(mock_agent):
    """Create epsilon-greedy wrapper with default settings."""
    return EpsilonGreedyWrapper(
        wrapped_agent=mock_agent,
        epsilon_start=1.0,
        epsilon_end=0.1,
        decay_steps=100,
        decay_strategy="linear"
    )


class TestEpsilonGreedyWrapper:
    """Test suite for EpsilonGreedyWrapper."""
    
    def test_initialization(self, mock_agent):
        """Test wrapper initialization."""
        wrapper = EpsilonGreedyWrapper(
            wrapped_agent=mock_agent,
            epsilon_start=0.8,
            epsilon_end=0.05,
            decay_steps=1000,
            decay_strategy="exponential"
        )
        
        assert wrapper.epsilon_start == 0.8
        assert wrapper.epsilon_end == 0.05
        assert wrapper.decay_steps == 1000
        assert wrapper.decay_strategy == "exponential"
        assert wrapper.epsilon == 0.8
        assert wrapper.step_count == 0
        assert wrapper.wrapped_agent is mock_agent
    
    def test_initialization_with_env(self, mock_env):
        """Test wrapper initialization with explicit environment."""
        agent = MockAgent(mock_env)
        wrapper = EpsilonGreedyWrapper(wrapped_agent=agent, env=mock_env)
        assert wrapper.env is mock_env
    
    def test_initialization_no_env_error(self):
        """Test wrapper raises error when no environment available."""
        agent = Mock()  # Agent without env attribute
        agent.env = None  # Explicitly set env to None
        with pytest.raises(ValueError, match="Environment must be provided"):
            EpsilonGreedyWrapper(wrapped_agent=agent)
    
    def test_linear_decay(self, epsilon_wrapper):
        """Test linear epsilon decay."""
        wrapper = epsilon_wrapper
        
        # Test initial value
        assert wrapper.epsilon == 1.0
        
        # Test midpoint
        wrapper.step_count = 50
        wrapper._update_epsilon()
        expected = 1.0 - (1.0 - 0.1) * 0.5  # 0.55
        assert abs(wrapper.epsilon - expected) < 1e-6
        
        # Test end
        wrapper.step_count = 100
        wrapper._update_epsilon()
        assert abs(wrapper.epsilon - 0.1) < 1e-6
        
        # Test beyond end
        wrapper.step_count = 150
        wrapper._update_epsilon()
        assert abs(wrapper.epsilon - 0.1) < 1e-6
    
    def test_exponential_decay(self, mock_agent):
        """Test exponential epsilon decay."""
        wrapper = EpsilonGreedyWrapper(
            wrapped_agent=mock_agent,
            epsilon_start=1.0,
            epsilon_end=0.1,
            decay_steps=100,
            decay_strategy="exponential"
        )
        
        # Test initial value
        assert wrapper.epsilon == 1.0
        
        # Test midpoint (should decay faster than linear)
        wrapper.step_count = 50
        wrapper._update_epsilon()
        linear_midpoint = 0.55
        assert wrapper.epsilon < linear_midpoint
        assert wrapper.epsilon > 0.1
        
        # Test end
        wrapper.step_count = 100
        wrapper._update_epsilon()
        assert wrapper.epsilon >= 0.1  # Should be close to but >= epsilon_end
    
    def test_unknown_decay_strategy(self, mock_agent):
        """Test error handling for unknown decay strategy."""
        wrapper = EpsilonGreedyWrapper(
            wrapped_agent=mock_agent,
            decay_strategy="unknown"
        )
        wrapper.step_count = 1
        
        with pytest.raises(ValueError, match="Unknown decay strategy"):
            wrapper._update_epsilon()
    
    def test_exploration_vs_exploitation(self, epsilon_wrapper):
        """Test exploration vs exploitation behavior."""
        wrapper = epsilon_wrapper
        observation = np.array([1, 2, 3])
        action_mask = np.array([True, True, False, True])
        
        # Set epsilon to 1.0 (always explore)
        wrapper.epsilon = 1.0
        wrapper.env.rng = np.random.default_rng(42)  # Fixed seed
        
        result = wrapper.select_action(observation, action_mask)
        assert isinstance(result, np.ndarray)
        # Should return uniform probabilities over valid actions [0, 1, 3]
        valid_indices = [0, 1, 3]
        expected_prob = 1.0 / len(valid_indices)
        for i in valid_indices:
            assert abs(result[i] - expected_prob) < 1e-6
        assert result[2] == 0.0  # Invalid action should have 0 probability
    
    def test_exploitation_delegates_to_wrapped(self, epsilon_wrapper):
        """Test exploitation delegates to wrapped agent."""
        wrapper = epsilon_wrapper
        observation = np.array([1, 2, 3])
        action_mask = np.array([True, True, False, True])
        
        # Set epsilon to 0.0 (never explore)
        wrapper.epsilon = 0.0
        
        result = wrapper.select_action(observation, action_mask)
        
        # Should have called wrapped agent
        assert len(wrapper.wrapped_agent.action_calls) == 1
        assert np.array_equal(wrapper.wrapped_agent.action_calls[0][0], observation)
        assert np.array_equal(wrapper.wrapped_agent.action_calls[0][1], action_mask)
    
    def test_act_method_probability_sampling(self, epsilon_wrapper):
        """Test act method with probability-returning wrapped agent."""
        wrapper = epsilon_wrapper
        wrapper.wrapped_agent.return_probs = True
        observation = np.array([1, 2, 3])
        action_mask = np.array([True, True, False])
        
        # Set fixed random seed
        wrapper.env.rng = np.random.default_rng(42)
        
        action = wrapper.act(observation, action_mask)
        assert isinstance(action, int)
        assert 0 <= action < len(action_mask)
        assert action_mask[action]  # Should be valid action
    
    def test_act_method_action_index(self, epsilon_wrapper):
        """Test act method with action-index-returning wrapped agent."""
        wrapper = epsilon_wrapper
        wrapper.wrapped_agent.return_probs = False
        observation = np.array([1, 2, 3])
        action_mask = np.array([True, True, False])
        
        action = wrapper.act(observation, action_mask)
        assert isinstance(action, int)
        assert 0 <= action < len(action_mask)
    
    def test_on_policy_distribution_mixing(self, mock_agent):
        """Test that on-policy distribution mixing works correctly."""
        # Create wrapper with known epsilon value
        wrapper = EpsilonGreedyWrapper(
            wrapped_agent=mock_agent,
            epsilon_start=0.5,  # 50% exploration
            epsilon_end=0.5,    # Keep constant
            decay_steps=1000,
            decay_strategy="linear"
        )
        
        # Mock wrapped agent to return known probabilities
        mock_agent.return_probs = True
        policy_probs = np.array([0.8, 0.2, 0.0])  # Strong preference for action 0
        mock_agent.set_test_return_value(policy_probs)
        
        observation = np.array([1, 2, 3])
        action_mask = np.array([True, True, False])  # Only actions 0 and 1 valid
        
        # Get mixed probabilities
        mixed_probs = wrapper.select_action(observation, action_mask)
        
        # Verify it's a probability distribution
        assert isinstance(mixed_probs, np.ndarray)
        assert np.isclose(np.sum(mixed_probs), 1.0)
        assert np.all(mixed_probs >= 0.0)
        
        # Calculate expected mixed distribution
        # uniform_probs = [0.5, 0.5, 0.0] (uniform over valid actions)
        # mixed = (1-0.5) * [0.8, 0.2, 0.0] + 0.5 * [0.5, 0.5, 0.0]
        # mixed = [0.4, 0.1, 0.0] + [0.25, 0.25, 0.0] = [0.65, 0.35, 0.0]
        expected_mixed = np.array([0.65, 0.35, 0.0])
        
        assert np.allclose(mixed_probs, expected_mixed, atol=1e-6)
    
    def test_on_policy_with_action_index_wrapped_agent(self, mock_agent):
        """Test on-policy mixing when wrapped agent returns action indices."""
        wrapper = EpsilonGreedyWrapper(
            wrapped_agent=mock_agent,
            epsilon_start=0.4,  # 40% exploration
            epsilon_end=0.4,    # Keep constant
            decay_steps=1000,
            decay_strategy="linear"
        )
        
        # Mock wrapped agent to return action index (not probabilities)
        mock_agent.return_probs = False
        mock_agent.set_test_return_value(1)  # Always chooses action 1
        
        observation = np.array([1, 2, 3])
        action_mask = np.array([True, True, True])  # All actions valid
        
        # Test multiple times to ensure consistent behavior
        for _ in range(10):
            result = wrapper.select_action(observation, action_mask)
            
            if isinstance(result, np.ndarray):
                # Should return mixed probabilities
                assert np.isclose(np.sum(result), 1.0)
                assert np.all(result >= 0.0)
                
                # Expected: (1-0.4) * [0,1,0] + 0.4 * [1/3,1/3,1/3]
                # = [0,0.6,0] + [0.133,0.133,0.133] = [0.133,0.733,0.133]
                expected = np.array([0.4/3, 0.6 + 0.4/3, 0.4/3])
                assert np.allclose(result, expected, atol=1e-3)
            else:
                # Should return valid action index
                assert isinstance(result, int)
                assert 0 <= result < len(action_mask)
                assert action_mask[result]
    
    def test_exploration_statistics(self, mock_agent):
        """Test exploration statistics tracking with new on-policy approach."""
        # Create wrapper with high epsilon
        wrapper = EpsilonGreedyWrapper(
            wrapped_agent=mock_agent,
            epsilon_start=0.8,  # High epsilon (>0.1 threshold)
            epsilon_end=0.8,    # Keep constant for testing
            decay_steps=1000000,
            decay_strategy="linear"
        )
        
        observation = np.array([1, 2, 3])
        action_mask = np.array([True, True, False])
        
        # Initial stats
        stats = wrapper.get_exploration_stats()
        assert stats['total_actions'] == 0
        assert stats['random_actions'] == 0
        assert stats['exploration_rate'] == 0.0
        
        # Perform several actions with high epsilon
        for _ in range(10):
            wrapper.select_action(observation, action_mask)
        
        # Check updated stats
        stats = wrapper.get_exploration_stats()
        assert stats['total_actions'] == 10
        # With epsilon > 0.1, all actions should be counted as exploration
        assert stats['random_actions'] == 10
        assert stats['exploration_rate'] == 1.0
        assert stats['random_action_rate'] == 1.0  # All actions were random
        
        # Check additional detailed statistics
        assert 'decay_mode' in stats
        assert 'decay_strategy' in stats
        assert 'decay_progress' in stats
        assert 'epsilon_start' in stats
        assert 'epsilon_end' in stats
        
    def test_exploration_statistics_low_epsilon(self, mock_agent):
        """Test exploration statistics with low epsilon."""
        # Create wrapper with low epsilon
        wrapper = EpsilonGreedyWrapper(
            wrapped_agent=mock_agent,
            epsilon_start=0.05,  # Low epsilon (<0.1 threshold)
            epsilon_end=0.05,
            decay_steps=1000,
            decay_strategy="linear"
        )
        
        observation = np.array([1, 2, 3])
        action_mask = np.array([True, True, False])
        
        # Perform several actions with low epsilon
        for _ in range(10):
            wrapper.select_action(observation, action_mask)
        
        # Check stats - with epsilon < 0.1, no actions counted as exploration
        stats = wrapper.get_exploration_stats()
        assert stats['total_actions'] == 10
        assert stats['random_actions'] == 0  # No exploration counted due to low epsilon
        assert stats['exploration_rate'] == 0.0
        assert stats['random_action_rate'] == 0.0  # New metric test
    
    def test_detailed_exploration_statistics(self, mock_agent):
        """Test comprehensive exploration statistics including new metrics."""
        wrapper = EpsilonGreedyWrapper(
            wrapped_agent=mock_agent,
            epsilon_start=0.9,
            epsilon_end=0.1,
            decay_steps=100,
            decay_strategy="exponential",
            decay_mode="episode"
        )
        
        observation = np.array([1, 2, 3])
        action_mask = np.array([True, True, True])
        
        # Perform actions
        for _ in range(20):
            wrapper.select_action(observation, action_mask)
        
        # Get comprehensive stats
        stats = wrapper.get_exploration_stats()
        
        # Test all expected keys exist
        expected_keys = [
            'epsilon', 'step_count', 'episode_count', 'decay_mode', 'decay_strategy',
            'decay_progress', 'decay_steps', 'epsilon_start', 'epsilon_end',
            'random_action_rate', 'total_actions', 'random_actions', 'exploration_rate'
        ]
        for key in expected_keys:
            assert key in stats, f"Missing key: {key}"
        
        # Test value types and ranges
        assert isinstance(stats['epsilon'], float)
        assert 0.0 <= stats['epsilon'] <= 1.0
        assert isinstance(stats['random_action_rate'], float)
        assert 0.0 <= stats['random_action_rate'] <= 1.0
        assert isinstance(stats['decay_progress'], float)
        assert 0.0 <= stats['decay_progress'] <= 1.0
        assert stats['decay_mode'] == "episode"
        assert stats['decay_strategy'] == "exponential"
        assert stats['epsilon_start'] == 0.9
        assert stats['epsilon_end'] == 0.1
        assert stats['decay_steps'] == 100
        assert stats['total_actions'] == 20
        
        # Test episode reset updates episode_count
        old_episode_count = stats['episode_count']
        wrapper.reset_episode_stats()
        new_stats = wrapper.get_exploration_stats()
        assert new_stats['episode_count'] == old_episode_count + 1
    
    def test_reset_episode_stats(self, epsilon_wrapper):
        """Test episode statistics reset."""
        wrapper = epsilon_wrapper
        observation = np.array([1, 2, 3])
        action_mask = np.array([True, True, False])
        
        # Generate some stats with high epsilon
        wrapper.epsilon = 0.8  # High epsilon for exploration counting
        wrapper.select_action(observation, action_mask)
        
        old_stats = wrapper.reset_episode_stats()
        assert old_stats['total_actions'] == 1
        assert old_stats['random_actions'] == 1  # High epsilon counts as exploration
        
        # Check reset
        new_stats = wrapper.get_exploration_stats()
        assert new_stats['total_actions'] == 0
        assert new_stats['random_actions'] == 0
        assert new_stats['exploration_rate'] == 0.0
    
    def test_no_valid_actions_fallback(self, epsilon_wrapper):
        """Test behavior when no valid actions are available."""
        wrapper = epsilon_wrapper
        observation = np.array([1, 2, 3])
        action_mask = np.array([False, False, False])  # No valid actions
        
        wrapper.epsilon = 1.0  # Force exploration
        result = wrapper.select_action(observation, action_mask)
        
        # Should delegate to wrapped agent
        assert len(wrapper.wrapped_agent.action_calls) == 1
    
    def test_delegation_methods(self, epsilon_wrapper):
        """Test delegation of methods to wrapped agent."""
        wrapper = epsilon_wrapper
        
        # Test choose_team delegation
        observation = "test_obs"
        result = wrapper.choose_team(observation)
        assert hasattr(wrapper.wrapped_agent, 'choose_team')
        
        # Test attribute delegation
        if hasattr(wrapper.wrapped_agent, 'some_attribute'):
            assert wrapper.some_attribute == wrapper.wrapped_agent.some_attribute
    
    def test_step_count_increments(self, epsilon_wrapper):
        """Test that step count increments on each action."""
        wrapper = epsilon_wrapper
        observation = np.array([1, 2, 3])
        action_mask = np.array([True, True, False])
        
        initial_count = wrapper.step_count
        wrapper.select_action(observation, action_mask)
        assert wrapper.step_count == initial_count + 1
        
        wrapper.select_action(observation, action_mask)
        assert wrapper.step_count == initial_count + 2
    
    def test_epsilon_bounds(self, epsilon_wrapper):
        """Test epsilon stays within bounds."""
        wrapper = epsilon_wrapper
        
        # Test lower bound
        wrapper.step_count = 1000  # Beyond decay_steps
        wrapper._update_epsilon()
        assert wrapper.epsilon >= wrapper.epsilon_end
        
        # Test upper bound
        wrapper.step_count = 0
        wrapper._update_epsilon()
        assert wrapper.epsilon <= wrapper.epsilon_start
    
    def test_zero_decay_steps_handling(self, mock_agent):
        """Test handling of zero decay steps."""
        wrapper = EpsilonGreedyWrapper(
            wrapped_agent=mock_agent,
            decay_steps=0  # Should be converted to 1
        )
        assert wrapper.decay_steps == 1
    
    def test_logging_methods(self, epsilon_wrapper, caplog):
        """Test logging functionality."""
        wrapper = epsilon_wrapper
        observation = np.array([1, 2, 3])
        action_mask = np.array([True, True, False])
        
        # Generate some activity
        wrapper.select_action(observation, action_mask)
        
        # Test log_exploration_stats doesn't crash
        wrapper.log_exploration_stats()
        
        # Test act method logging
        with caplog.at_level("DEBUG"):
            wrapper.act(observation, action_mask)
            # Should contain debug log about chosen action


if __name__ == "__main__":
    pytest.main([__file__])