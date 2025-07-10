#!/usr/bin/env python3
"""Debug script to test the actual networks created by the training script."""

import sys
import torch
import yaml
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.env.pokemon_env import PokemonEnv
from src.state.state_observer import StateObserver
from src.action import action_helper
from src.agents.network_factory import create_policy_network, create_value_network

def debug_actual_networks():
    """Debug the actual networks created by the training script setup."""
    
    # Load configuration like in training script
    config_path = "config/train_config.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    
    # Create environment to get spaces (like in training script)
    observer = StateObserver(str(ROOT_DIR / "config" / "state_spec.yml"))
    env = PokemonEnv(
        opponent_player=None,
        state_observer=observer,
        action_helper=action_helper,
        reward="composite",
        reward_config_path="config/reward.yaml",
        team_mode="default",
        teams_dir=None,
        normalize_rewards=True,
    )
    
    # Get spaces (extract single agent spaces from multi-agent dict)
    observation_space = env.observation_space["player_0"]
    action_space = env.action_space["player_0"]
    
    print(f"Observation space: {observation_space}")
    print(f"Action space: {action_space}")
    
    # Get network config (like in training script)
    network_config = cfg.get("network", {})
    print(f"Network config: {network_config}")
    
    # Create networks exactly like in training script
    policy_net = create_policy_network(observation_space, action_space, network_config)
    value_net = create_value_network(observation_space, network_config)
    
    print(f"Policy network type: {type(policy_net)}")
    print(f"Value network type: {type(value_net)}")
    
    # Test value network with single observation
    obs_dim = observation_space.shape[0]
    obs_t = torch.randn(1, obs_dim)
    
    print(f"\nTesting with obs_t shape: {obs_t.shape}")
    
    # Test value network
    value_output = value_net(obs_t)
    print(f"Value output type: {type(value_output)}")
    
    if isinstance(value_output, tuple):
        value_t, hidden = value_output
        print(f"Value tensor shape: {value_t.shape}")
        print(f"Hidden state type: {type(hidden)}")
    else:
        print(f"Value output shape: {value_output.shape}")
    
    # Test policy network
    policy_output = policy_net(obs_t)
    print(f"Policy output type: {type(policy_output)}")
    
    if isinstance(policy_output, tuple):
        logits_t, hidden = policy_output
        print(f"Logits tensor shape: {logits_t.shape}")
        print(f"Hidden state type: {type(hidden)}")
    else:
        print(f"Policy output shape: {policy_output.shape}")

    env.close()

if __name__ == "__main__":
    debug_actual_networks()