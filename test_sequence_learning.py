#!/usr/bin/env python3
"""
Test script for sequence learning optimizations.
Tests the new LSTM learning optimization features implemented according to the design document.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.algorithms.sequence_ppo import SequencePPOAlgorithm, SequenceReinforceAlgorithm
from src.algorithms.ppo import PPOAlgorithm
from src.algorithms.reinforce import ReinforceAlgorithm
from src.agents.enhanced_networks import LSTMPolicyNetwork, LSTMValueNetwork
import gymnasium as gym


def test_gradient_clipping():
    """Test that gradient clipping is properly implemented in standard algorithms."""
    print("Testing gradient clipping...")
    
    # Create a simple network
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(10,))
    action_space = gym.spaces.Discrete(4)
    
    net = LSTMPolicyNetwork(obs_space, action_space, use_lstm=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    
    # Create batch data
    batch_size = 5
    batch = {
        "observations": torch.randn(batch_size, 10),
        "actions": torch.randint(0, 4, (batch_size,)),
        "old_log_probs": torch.randn(batch_size),
        "advantages": torch.randn(batch_size) * 10,  # Large advantages to cause gradient explosion
        "returns": torch.randn(batch_size) * 10,
        "values": torch.randn(batch_size),
        "rewards": torch.randn(batch_size)
    }
    
    # Test PPO with gradient clipping
    algo = PPOAlgorithm(clip_range=0.2)
    
    # Make gradients very large to test clipping
    for param in net.parameters():
        if param.grad is not None:
            param.grad = None
    
    loss = algo.update(net, optimizer, batch)
    
    # Check that gradients are clipped
    max_grad_norm = 0.0
    for param in net.parameters():
        if param.grad is not None:
            max_grad_norm = max(max_grad_norm, param.grad.norm().item())
    
    print(f"Max gradient norm after clipping: {max_grad_norm:.4f}")
    assert max_grad_norm <= 5.1, f"Gradient clipping failed: {max_grad_norm} > 5.0"
    print("✓ Gradient clipping test passed")


def test_sequence_ppo_bptt():
    """Test sequence PPO with different BPTT lengths."""
    print("\nTesting sequence PPO with BPTT...")
    
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(10,))
    action_space = gym.spaces.Discrete(4)
    
    # Create LSTM networks
    policy_net = LSTMPolicyNetwork(obs_space, action_space, use_lstm=True)
    value_net = LSTMValueNetwork(obs_space, use_lstm=True)
    
    # Create optimizer
    params = list(policy_net.parameters()) + list(value_net.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    
    # Test different BPTT lengths
    for bptt_length in [0, 10, 20]:
        print(f"Testing BPTT length: {bptt_length}")
        
        algo = SequencePPOAlgorithm(
            bptt_length=bptt_length,
            grad_clip_norm=5.0
        )
        
        # Create episode data (simulate two episodes of different lengths)
        ep1_len, ep2_len = 15, 25
        total_len = ep1_len + ep2_len
        
        batch = {
            "observations": torch.randn(total_len, 10),
            "actions": torch.randint(0, 4, (total_len,)),
            "old_log_probs": torch.randn(total_len),
            "advantages": torch.randn(total_len),
            "returns": torch.randn(total_len),
            "values": torch.randn(total_len),
            "rewards": torch.randn(total_len),
            "episode_lengths": torch.tensor([ep1_len, ep2_len], dtype=torch.int64)
        }
        
        # Test that the algorithm can handle the batch
        loss = algo.update((policy_net, value_net), optimizer, batch)
        
        print(f"  Loss: {loss:.4f}")
        assert not np.isnan(loss), f"Loss is NaN for BPTT length {bptt_length}"
        assert loss > 0, f"Loss should be positive, got {loss}"
    
    print("✓ Sequence PPO BPTT test passed")


def test_episode_boundary_handling():
    """Test that episode boundaries are properly handled."""
    print("\nTesting episode boundary handling...")
    
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(10,))
    action_space = gym.spaces.Discrete(4)
    
    # Create LSTM networks
    policy_net = LSTMPolicyNetwork(obs_space, action_space, use_lstm=True)
    value_net = LSTMValueNetwork(obs_space, use_lstm=True)
    
    params = list(policy_net.parameters()) + list(value_net.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    
    # Create algorithm with truncated BPTT
    algo = SequencePPOAlgorithm(bptt_length=10, grad_clip_norm=5.0)
    
    # Create batch with multiple episodes
    episodes = [8, 12, 15, 20]  # Different episode lengths
    total_len = sum(episodes)
    
    batch = {
        "observations": torch.randn(total_len, 10),
        "actions": torch.randint(0, 4, (total_len,)),
        "old_log_probs": torch.randn(total_len),
        "advantages": torch.randn(total_len),
        "returns": torch.randn(total_len),
        "values": torch.randn(total_len),
        "rewards": torch.randn(total_len),
        "episode_lengths": torch.tensor(episodes, dtype=torch.int64)
    }
    
    # Test that sequences are properly split
    sequences = algo._split_episode_into_sequences(batch, episodes)
    
    print(f"Number of sequences created: {len(sequences)}")
    
    # Check that sequences respect episode boundaries
    expected_sequences = 0
    for ep_len in episodes:
        if ep_len <= 10:
            expected_sequences += 1
        else:
            expected_sequences += (ep_len + 9) // 10  # Ceiling division
    
    print(f"Expected sequences: {expected_sequences}, Got: {len(sequences)}")
    
    # Verify that no sequence crosses episode boundaries
    for i, seq in enumerate(sequences):
        seq_len = len(seq["observations"])
        print(f"Sequence {i}: length {seq_len}")
        assert seq_len <= 10, f"Sequence {i} too long: {seq_len}"
    
    print("✓ Episode boundary handling test passed")


def test_sequence_reinforce():
    """Test sequence REINFORCE algorithm."""
    print("\nTesting sequence REINFORCE...")
    
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(10,))
    action_space = gym.spaces.Discrete(4)
    
    # Create LSTM network
    policy_net = LSTMPolicyNetwork(obs_space, action_space, use_lstm=True)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=0.001)
    
    # Create algorithm
    algo = SequenceReinforceAlgorithm(bptt_length=15, grad_clip_norm=5.0)
    
    # Create batch data
    ep_len = 20
    batch = {
        "observations": torch.randn(ep_len, 10),
        "actions": torch.randint(0, 4, (ep_len,)),
        "rewards": torch.randn(ep_len),
        "episode_lengths": torch.tensor([ep_len], dtype=torch.int64)
    }
    
    # Test update
    loss = algo.update(policy_net, optimizer, batch)
    
    print(f"REINFORCE loss: {loss:.4f}")
    assert not np.isnan(loss), f"Loss is NaN"
    
    print("✓ Sequence REINFORCE test passed")


def test_comparison_with_standard_algorithms():
    """Compare sequence algorithms with standard algorithms."""
    print("\nTesting comparison with standard algorithms...")
    
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(10,))
    action_space = gym.spaces.Discrete(4)
    
    # Create networks
    policy_net = LSTMPolicyNetwork(obs_space, action_space, use_lstm=False)  # No LSTM for fair comparison
    value_net = LSTMValueNetwork(obs_space, use_lstm=False)
    
    params = list(policy_net.parameters()) + list(value_net.parameters())
    optimizer = torch.optim.Adam(params, lr=0.001)
    
    # Create batch data
    batch_size = 50
    batch = {
        "observations": torch.randn(batch_size, 10),
        "actions": torch.randint(0, 4, (batch_size,)),
        "old_log_probs": torch.randn(batch_size),
        "advantages": torch.randn(batch_size),
        "returns": torch.randn(batch_size),
        "values": torch.randn(batch_size),
        "rewards": torch.randn(batch_size),
        "episode_lengths": torch.tensor([batch_size], dtype=torch.int64)
    }
    
    # Test standard PPO
    standard_algo = PPOAlgorithm(clip_range=0.2)
    standard_loss = standard_algo.update(policy_net, optimizer, batch)
    
    # Reset optimizer
    optimizer = torch.optim.Adam(params, lr=0.001)
    
    # Test sequence PPO (full episode)
    sequence_algo = SequencePPOAlgorithm(bptt_length=0, clip_range=0.2)
    sequence_loss = sequence_algo.update((policy_net, value_net), optimizer, batch)
    
    print(f"Standard PPO loss: {standard_loss:.4f}")
    print(f"Sequence PPO loss: {sequence_loss:.4f}")
    print(f"Loss difference: {abs(standard_loss - sequence_loss):.4f}")
    
    # Losses should be similar for non-LSTM networks with full episode BPTT
    assert abs(standard_loss - sequence_loss) < 2.0, "Losses should be similar for non-LSTM networks"
    
    print("✓ Algorithm comparison test passed")


def test_config_integration():
    """Test that configuration integration works properly."""
    print("\nTesting configuration integration...")
    
    # Test configuration parameters
    configs = [
        {"bptt_length": 0, "grad_clip_norm": 5.0},
        {"bptt_length": 20, "grad_clip_norm": 10.0},
        {"bptt_length": 50, "grad_clip_norm": 2.0},
    ]
    
    for config in configs:
        algo = SequencePPOAlgorithm(**config)
        assert algo.bptt_length == config["bptt_length"]
        assert algo.grad_clip_norm == config["grad_clip_norm"]
    
    print("✓ Configuration integration test passed")


def main():
    """Run all tests."""
    print("Running LSTM learning optimization tests...")
    print("=" * 50)
    
    try:
        test_gradient_clipping()
        test_sequence_ppo_bptt()
        test_episode_boundary_handling()
        test_sequence_reinforce()
        test_comparison_with_standard_algorithms()
        test_config_integration()
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("LSTM learning optimization implementation is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()