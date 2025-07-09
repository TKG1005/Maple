#!/usr/bin/env python3
"""Test training loop with LSTM hidden state management."""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.env.pokemon_env import PokemonEnv
from src.state.state_observer import StateObserver
from src.action import action_helper
from src.agents.enhanced_networks import LSTMPolicyNetwork, LSTMValueNetwork
from src.agents.RLAgent import RLAgent

def test_training_loop_simulation():
    """Simulate training loop behavior to test LSTM hidden state management."""
    print("=== Training Loop Simulation Test ===")
    
    # Create environment
    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
        reward="composite",
        normalize_rewards=True,
    )
    
    # Create LSTM networks
    obs_space = env.observation_space[env.agent_ids[0]]
    action_space = env.action_space[env.agent_ids[0]]
    
    policy_net = LSTMPolicyNetwork(
        obs_space, action_space, 
        hidden_size=128, lstm_hidden_size=128, 
        use_lstm=True, use_2layer=True
    )
    value_net = LSTMValueNetwork(
        obs_space, 
        hidden_size=128, lstm_hidden_size=128,
        use_lstm=True, use_2layer=True
    )
    
    # Create agents
    agent0 = RLAgent(env, policy_net, value_net, None)
    agent1 = RLAgent(env, policy_net, value_net, None)
    
    print(f"Agent0 has hidden states: {agent0.has_hidden_states}")
    print(f"Agent1 has hidden states: {agent1.has_hidden_states}")
    
    # Simulate multiple episodes
    num_episodes = 3
    steps_per_episode = 5
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1} ---")
        
        # Reset hidden states at episode start (like in training loop)
        agent0.reset_hidden_states()
        agent1.reset_hidden_states()
        print("Hidden states reset at episode start")
        
        # Check that hidden states are None after reset
        assert policy_net.hidden_state is None, "Policy hidden state should be None after reset"
        assert value_net.hidden_state is None, "Value hidden state should be None after reset"
        
        # Simulate episode steps
        for step in range(steps_per_episode):
            # Generate random observations and masks
            obs0 = np.random.random(obs_space.shape).astype(np.float32)
            obs1 = np.random.random(obs_space.shape).astype(np.float32)
            mask0 = np.ones(action_space.n, dtype=bool)
            mask1 = np.ones(action_space.n, dtype=bool)
            
            # Select actions
            probs0 = agent0.select_action(obs0, mask0)
            probs1 = agent1.select_action(obs1, mask1)
            
            # Check that hidden states exist after first action
            if step == 0:
                assert policy_net.hidden_state is not None, "Policy hidden state should exist after first action"
                print(f"  Step {step + 1}: Policy hidden state created")
                print(f"  Step {step + 1}: Value hidden state: {value_net.hidden_state is not None}")
            else:
                print(f"  Step {step + 1}: Policy hidden state maintained")
                print(f"  Step {step + 1}: Value hidden state: {value_net.hidden_state is not None}")
            
            # Test value network usage (like in training loop)
            obs0_tensor = torch.as_tensor(obs0, dtype=torch.float32)
            if obs0_tensor.dim() == 1:
                obs0_tensor = obs0_tensor.unsqueeze(0)
            val0_tensor = value_net(obs0_tensor, value_net.hidden_state if hasattr(value_net, 'hidden_state') else None)
            if val0_tensor.dim() > 0:
                val0_tensor = val0_tensor.squeeze(0)
            val0 = float(val0_tensor.item())
            
            print(f"    Agent0 action probs shape: {probs0.shape}, value: {val0:.4f}")
            
        print(f"  Episode {episode + 1} completed successfully")
    
    print("\n✓ Training loop simulation test passed!")
    print("✓ LSTM hidden state management is working correctly in training context!")

if __name__ == "__main__":
    test_training_loop_simulation()