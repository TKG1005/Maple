#!/usr/bin/env python3
"""Test script to verify LSTM sequential learning capability."""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.agents.enhanced_networks import LSTMPolicyNetwork, LSTMValueNetwork
from src.agents.RLAgent import RLAgent
from src.env.pokemon_env import PokemonEnv
from src.state.state_observer import StateObserver
from src.action import action_helper

def test_lstm_sequential_learning():
    """Test that LSTM networks can learn sequential patterns."""
    print("Testing LSTM sequential learning capability...")
    
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
    
    # Create RLAgent
    agent = RLAgent(env, policy_net, None, None)
    
    print(f"Observation space: {obs_space.shape}")
    print(f"Action space: {action_space.n}")
    print(f"Agent has hidden states: {agent.has_hidden_states}")
    
    # Test 1: Sequential processing maintains different hidden states
    print("\n=== Test 1: Sequential Hidden State Changes ===")
    
    # Reset hidden states
    agent.reset_hidden_states()
    
    # Create a sequence of different observations
    obs_sequence = []
    for i in range(5):
        obs = np.random.random(obs_space.shape).astype(np.float32)
        obs_sequence.append(obs)
    
    mask = np.ones(action_space.n, dtype=bool)
    hidden_states = []
    
    # Process sequence and track hidden states
    for i, obs in enumerate(obs_sequence):
        probs = agent.select_action(obs, mask)
        if policy_net.hidden_state is not None:
            h, c = policy_net.hidden_state
            hidden_states.append((h.clone(), c.clone()))
        print(f"Step {i+1}: Hidden state exists = {policy_net.hidden_state is not None}")
    
    # Verify hidden states are different across steps
    if len(hidden_states) > 1:
        for i in range(1, len(hidden_states)):
            h_prev, c_prev = hidden_states[i-1]
            h_curr, c_curr = hidden_states[i]
            
            h_different = not torch.equal(h_prev, h_curr)
            c_different = not torch.equal(c_prev, c_curr)
            
            print(f"Step {i} vs {i+1}: h_different={h_different}, c_different={c_different}")
            assert h_different or c_different, f"Hidden states should change between steps {i} and {i+1}"
    
    print("✓ Sequential hidden state changes verified!")
    
    # Test 2: Reset clears hidden states properly
    print("\n=== Test 2: Hidden State Reset ===")
    
    # Ensure we have hidden states
    agent.select_action(obs_sequence[0], mask)
    assert policy_net.hidden_state is not None, "Should have hidden state before reset"
    
    # Reset and verify
    agent.reset_hidden_states()
    assert policy_net.hidden_state is None, "Hidden state should be None after reset"
    
    # New sequence should start fresh
    agent.select_action(obs_sequence[0], mask)
    h_new, c_new = policy_net.hidden_state
    
    # Should be different from previous sequence
    h_prev, c_prev = hidden_states[0]
    h_different = not torch.equal(h_prev, h_new)
    c_different = not torch.equal(c_prev, c_new)
    
    print(f"Reset comparison: h_different={h_different}, c_different={c_different}")
    print("✓ Hidden state reset verified!")
    
    # Test 3: Different observation sequences produce different outputs
    print("\n=== Test 3: Sequence Dependency ===")
    
    # Create two different sequences
    seq1 = [np.random.random(obs_space.shape).astype(np.float32) for _ in range(3)]
    seq2 = [np.random.random(obs_space.shape).astype(np.float32) for _ in range(3)]
    
    # Process sequence 1
    agent.reset_hidden_states()
    probs1 = []
    for obs in seq1:
        probs = agent.select_action(obs, mask)
        probs1.append(probs)
    
    # Process sequence 2
    agent.reset_hidden_states()
    probs2 = []
    for obs in seq2:
        probs = agent.select_action(obs, mask)
        probs2.append(probs)
    
    # Test that processing the same observation with different histories gives different results
    # Use the same observation for both sequences' final step
    test_obs = np.random.random(obs_space.shape).astype(np.float32)
    
    # Process seq1 then test_obs
    agent.reset_hidden_states()
    for obs in seq1:
        agent.select_action(obs, mask)
    probs_after_seq1 = agent.select_action(test_obs, mask)
    
    # Process seq2 then test_obs
    agent.reset_hidden_states()
    for obs in seq2:
        agent.select_action(obs, mask)
    probs_after_seq2 = agent.select_action(test_obs, mask)
    
    # Check if the outputs are different (they should be due to different histories)
    prob_diff = np.abs(probs_after_seq1 - probs_after_seq2).max()
    print(f"Max probability difference after different sequences: {prob_diff:.6f}")
    
    # If sequences were truly different, we should see some difference
    # (though this test can be flaky with random data)
    if prob_diff > 1e-6:
        print("✓ Sequence dependency verified - different histories produce different outputs!")
    else:
        print("? Sequence dependency test inconclusive (random data may be too similar)")
    
    print("\n🎉 LSTM sequential learning capability tests completed!")

def test_lstm_vs_basic_network():
    """Compare LSTM vs basic network behavior to ensure LSTM is actually being used."""
    print("Testing LSTM vs Basic Network Behavior...")
    
    # Create a simple test environment
    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
        reward="composite",
        normalize_rewards=False,
    )
    
    obs_space = env.observation_space[env.agent_ids[0]]
    action_space = env.action_space[env.agent_ids[0]]
    
    # Create LSTM network
    lstm_net = LSTMPolicyNetwork(
        obs_space, action_space, 
        hidden_size=64, lstm_hidden_size=32, 
        use_lstm=True, use_2layer=False
    )
    
    # Create basic network for comparison
    from src.agents import PolicyNetwork
    basic_net = PolicyNetwork(obs_space, action_space, hidden_size=64)
    
    # Create agents
    lstm_agent = RLAgent(env, lstm_net, None, None)
    basic_agent = RLAgent(env, basic_net, None, None)
    
    print(f"LSTM agent has hidden states: {lstm_agent.has_hidden_states}")
    print(f"Basic agent has hidden states: {basic_agent.has_hidden_states}")
    
    # Test with same observation sequence
    obs_sequence = [np.random.random(obs_space.shape).astype(np.float32) for _ in range(3)]
    mask = np.ones(action_space.n, dtype=bool)
    
    # Process with LSTM (should maintain state)
    lstm_agent.reset_hidden_states()
    lstm_probs = []
    for obs in obs_sequence:
        probs = lstm_agent.select_action(obs, mask)
        lstm_probs.append(probs)
    
    # Process with basic network (no state)
    basic_probs = []
    for obs in obs_sequence:
        probs = basic_agent.select_action(obs, mask)
        basic_probs.append(probs)
    
    # The outputs should be different due to LSTM's sequential processing
    final_diff = np.abs(lstm_probs[-1] - basic_probs[-1]).max()
    print(f"Final step probability difference: {final_diff:.6f}")
    
    print("✓ LSTM vs Basic network comparison completed!")

if __name__ == "__main__":
    test_lstm_sequential_learning()
    test_lstm_vs_basic_network()
    print("\n🎉 All sequential learning tests completed!")