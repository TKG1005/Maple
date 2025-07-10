#!/usr/bin/env python3
"""Test script to verify value network creation and functionality."""

import sys
sys.path.insert(0, str('copy_of_poke-env'))
from train_selfplay import init_env
from src.agents.network_factory import create_policy_network, create_value_network, get_network_info
from src.agents.RLAgent import RLAgent
from src.algorithms import PPOAlgorithm
import torch
from torch import optim
import yaml
import numpy as np

def main():
    # Load config
    with open('config/train_config.yml', 'r') as f:
        cfg = yaml.safe_load(f)

    # Create environment
    env = init_env(reward='composite', reward_config='config/reward.yaml', team_mode='default', teams_dir=None, normalize_rewards=True)

    # Get network configuration
    network_config = cfg.get('network', {})
    print('Network configuration:', network_config)

    # Create networks using factory
    policy_net = create_policy_network(
        env.observation_space[env.agent_ids[0]],
        env.action_space[env.agent_ids[0]],
        network_config
    )
    value_net = create_value_network(
        env.observation_space[env.agent_ids[0]],
        network_config
    )

    # Create optimizer and algorithm
    params = list(policy_net.parameters()) + list(value_net.parameters())
    optimizer = optim.Adam(params, lr=0.001)
    algorithm = PPOAlgorithm(clip_range=0.2, value_coef=0.6, entropy_coef=0.01)

    # Create RLAgent
    agent = RLAgent(env, policy_net, value_net, optimizer, algorithm=algorithm)

    print()
    print('RLAgent has_hidden_states:', agent.has_hidden_states)
    print('Agent policy_hidden:', agent.policy_hidden)
    print('Agent value_hidden:', agent.value_hidden)

    # Test environment reset
    observations, info, masks = env.reset(return_masks=True)
    obs0 = observations[env.agent_ids[0]]
    mask0 = masks[0]

    print()
    print('Observation shape:', obs0.shape)
    print('Mask shape:', mask0.shape)

    # Test action selection
    agent.reset_hidden_states()
    probs = agent.select_action(obs0, mask0)
    print('Action probabilities shape:', probs.shape)
    print('Action probabilities sum:', np.sum(probs))

    # Test value estimation
    value = agent.get_value(obs0)
    print('Value estimate:', value)
    print('Value type:', type(value))

    # Test hidden state persistence
    print()
    print('After action selection:')
    print('Policy hidden is None:', agent.policy_hidden is None)
    print('Value hidden is None:', agent.value_hidden is None)

    # Test another action selection to verify hidden state persistence
    probs2 = agent.select_action(obs0, mask0)
    value2 = agent.get_value(obs0)
    print('Second action probabilities shape:', probs2.shape)
    print('Second value estimate:', value2)
    values_different = (value != value2)
    print('Values are different (hidden state working):', values_different)

    env.close()
    print()
    print('Test completed successfully!')

if __name__ == '__main__':
    main()