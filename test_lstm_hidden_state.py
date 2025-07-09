#!/usr/bin/env python3
"""Test script to verify LSTM hidden state management works correctly."""

import sys
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.agents.enhanced_networks import LSTMPolicyNetwork, LSTMValueNetwork
from src.agents.RLAgent import RLAgent
from src.env.pokemon_env import PokemonEnv
from src.state.state_observer import StateObserver
from src.action import action_helper

def test_lstm_hidden_state_management():
    """Test that LSTM hidden states are properly managed and reset."""
    print("Testing LSTM hidden state management...")
    
    # Create a simple test environment
    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
        reward="composite",
        normalize_rewards=False,
    )
    
    # Create LSTM networks
    obs_space = env.observation_space[env.agent_ids[0]]
    action_space = env.action_space[env.agent_ids[0]]
    
    policy_net = LSTMPolicyNetwork(
        obs_space, action_space, 
        hidden_size=64, lstm_hidden_size=32, 
        use_lstm=True, use_2layer=False
    )
    value_net = LSTMValueNetwork(
        obs_space, 
        hidden_size=64, lstm_hidden_size=32,
        use_lstm=True, use_2layer=False
    )
    
    # Create RLAgent
    agent = RLAgent(env, policy_net, value_net, None)
    
    # Test 1: Check that agent recognizes LSTM networks
    print(f"Agent has hidden states: {agent.has_hidden_states}")
    assert agent.has_hidden_states, "Agent should recognize LSTM networks"
    
    # Test 2: Check initial hidden states are None
    assert policy_net.hidden_state is None, "Initial policy hidden state should be None"
    assert value_net.hidden_state is None, "Initial value hidden state should be None"
    
    # Test 3: Reset hidden states explicitly
    agent.reset_hidden_states()
    assert policy_net.hidden_state is None, "Hidden states should be None after reset"
    assert value_net.hidden_state is None, "Hidden states should be None after reset"
    
    # Test 4: Run forward pass to create hidden states
    obs = np.random.random(obs_space.shape).astype(np.float32)
    mask = np.ones(action_space.n, dtype=bool)
    
    # First action should initialize hidden states
    probs1 = agent.select_action(obs, mask)
    assert policy_net.hidden_state is not None, "Hidden state should be created after first action"
    h1, c1 = policy_net.hidden_state
    
    # Second action should reuse and update hidden states
    probs2 = agent.select_action(obs, mask)
    assert policy_net.hidden_state is not None, "Hidden state should persist across actions"
    h2, c2 = policy_net.hidden_state
    
    # Hidden states should be different (updated)
    assert not torch.equal(h1, h2), "Hidden states should be updated between actions"
    assert not torch.equal(c1, c2), "Cell states should be updated between actions"
    
    # Test 5: Reset should clear hidden states
    agent.reset_hidden_states()
    assert policy_net.hidden_state is None, "Hidden states should be cleared after reset"
    assert value_net.hidden_state is None, "Hidden states should be cleared after reset"
    
    # Test 6: New episode should start with clean hidden states
    probs3 = agent.select_action(obs, mask)
    h3, c3 = policy_net.hidden_state
    
    # Should be different from previous episode's states
    assert not torch.equal(h2, h3), "New episode should have different hidden states"
    assert not torch.equal(c2, c3), "New episode should have different cell states"
    
    print("✓ All LSTM hidden state management tests passed!")

def test_basic_network_compatibility():
    """Test that the fix doesn't break basic (non-LSTM) networks."""
    print("Testing basic network compatibility...")
    
    # Create a simple test environment
    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
        reward="composite",
        normalize_rewards=False,
    )
    
    # Create basic networks (from PolicyNetwork, ValueNetwork)
    from src.agents import PolicyNetwork, ValueNetwork
    
    obs_space = env.observation_space[env.agent_ids[0]]
    action_space = env.action_space[env.agent_ids[0]]
    
    policy_net = PolicyNetwork(obs_space, action_space, hidden_size=64)
    value_net = ValueNetwork(obs_space, hidden_size=64)
    
    # Create RLAgent
    agent = RLAgent(env, policy_net, value_net, None)
    
    # Test that agent recognizes these as non-LSTM networks
    print(f"Agent has hidden states: {agent.has_hidden_states}")
    assert not agent.has_hidden_states, "Agent should not recognize basic networks as having hidden states"
    
    # Test that reset_hidden_states doesn't break anything
    agent.reset_hidden_states()  # Should not raise an error
    
    # Test that action selection works normally
    obs = np.random.random(obs_space.shape).astype(np.float32)
    mask = np.ones(action_space.n, dtype=bool)
    
    probs = agent.select_action(obs, mask)
    assert len(probs) == action_space.n, "Action probabilities should match action space size"
    
    print("✓ Basic network compatibility tests passed!")

if __name__ == "__main__":
    test_lstm_hidden_state_management()
    test_basic_network_compatibility()
    print("\n🎉 All tests passed! LSTM hidden state management is working correctly.")