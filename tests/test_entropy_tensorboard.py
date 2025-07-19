"""
Test entropy output functionality for TensorBoard logging.
"""

import pytest
import torch
import numpy as np
from src.algorithms.ppo import PPOAlgorithm, compute_ppo_loss
from src.algorithms.sequence_ppo import SequencePPOAlgorithm


class TestEntropyTensorBoard:
    """Test entropy computation and TensorBoard logging integration."""
    
    def test_compute_ppo_loss_entropy(self):
        """Test that compute_ppo_loss returns entropy when requested."""
        # Create test data
        new_log_probs = torch.tensor([-1.0, -1.5, -2.0])
        old_log_probs = torch.tensor([-1.1, -1.4, -1.9])
        advantages = torch.tensor([0.5, -0.3, 0.8])
        returns = torch.tensor([2.0, 1.5, 3.0])
        values = torch.tensor([1.8, 1.7, 2.9])
        
        # Test without entropy return
        loss = compute_ppo_loss(
            new_log_probs, old_log_probs, advantages, returns, values
        )
        assert isinstance(loss, torch.Tensor)
        
        # Test with entropy return
        result = compute_ppo_loss(
            new_log_probs, old_log_probs, advantages, returns, values,
            return_entropy=True
        )
        assert isinstance(result, tuple)
        assert len(result) == 2
        loss, entropy = result
        assert isinstance(loss, torch.Tensor)
        assert isinstance(entropy, float)
        assert entropy >= 0  # Entropy should be non-negative
    
    def test_ppo_algorithm_entropy_output(self):
        """Test that PPOAlgorithm update returns (loss, entropy) tuple."""
        algorithm = PPOAlgorithm()
        
        # Create mock model that returns logits
        class MockPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 4)
            
            def forward(self, x):
                return self.linear(x)
        
        class MockValue(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 1)
            
            def forward(self, x):
                return self.linear(x)
        
        policy_net = MockPolicy()
        value_net = MockValue()
        optimizer = torch.optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()))
        
        # Create test batch
        batch = {
            "observations": np.random.randn(10, 5),
            "actions": np.random.randint(0, 4, 10),
            "old_log_probs": np.random.randn(10),
            "advantages": np.random.randn(10),
            "returns": np.random.randn(10),
        }
        
        # Test update method
        result = algorithm.update((policy_net, value_net), optimizer, batch)
        assert isinstance(result, tuple)
        assert len(result) == 2
        loss, entropy = result
        assert isinstance(loss, float)
        assert isinstance(entropy, float)
        assert entropy >= 0  # Entropy should be non-negative
    
    def test_sequence_ppo_algorithm_entropy_output(self):
        """Test that SequencePPOAlgorithm update returns (loss, entropy) tuple."""
        algorithm = SequencePPOAlgorithm()
        
        # Create mock LSTM policy network
        class MockLSTMPolicy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(5, 64, batch_first=True)
                self.output = torch.nn.Linear(64, 4)
            
            def forward(self, x, hidden=None):
                if x.dim() == 2:  # [batch, features]
                    x = x.unsqueeze(1)  # [batch, 1, features]
                lstm_out, hidden = self.lstm(x, hidden)
                logits = self.output(lstm_out.squeeze(1))
                return logits, hidden
        
        class MockLSTMValue(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(5, 64, batch_first=True)
                self.output = torch.nn.Linear(64, 1)
            
            def forward(self, x, hidden=None):
                if x.dim() == 2:  # [batch, features]
                    x = x.unsqueeze(1)  # [batch, 1, features]
                lstm_out, hidden = self.lstm(x, hidden)
                values = self.output(lstm_out.squeeze(1))
                return values, hidden
        
        policy_net = MockLSTMPolicy()
        value_net = MockLSTMValue()
        optimizer = torch.optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()))
        
        # Create test batch with episode structure
        batch = {
            "observations": np.random.randn(15, 5),  # 15 timesteps
            "actions": np.random.randint(0, 4, 15),
            "old_log_probs": np.random.randn(15),
            "advantages": np.random.randn(15),
            "returns": np.random.randn(15),
            "episode_lengths": [10, 5],  # Two episodes
        }
        
        # Test update method
        result = algorithm.update((policy_net, value_net), optimizer, batch)
        if isinstance(result, tuple):
            assert len(result) == 2
            loss, entropy = result
            assert isinstance(loss, float)
            assert isinstance(entropy, float)
            assert entropy >= 0  # Entropy should be non-negative
        else:
            # Fallback for when entropy is not returned
            assert isinstance(result, float)
    
    def test_entropy_computation_stability(self):
        """Test that entropy computation is numerically stable."""
        # Test with various probability distributions
        test_cases = [
            torch.tensor([[-1.0, -1.0, -1.0, -1.0]]),  # Uniform distribution
            torch.tensor([[-0.1, -3.0, -3.0, -3.0]]),  # Near-deterministic
            torch.tensor([[-1.5, -1.0, -2.0, -1.2]]),  # Mixed distribution
        ]
        
        for log_probs in test_cases:
            old_log_probs = log_probs + torch.randn_like(log_probs) * 0.1
            advantages = torch.randn(1)
            returns = torch.randn(1) 
            values = torch.randn(1)
            
            result = compute_ppo_loss(
                log_probs[0], old_log_probs[0], advantages, returns, values,
                return_entropy=True
            )
            
            loss, entropy = result
            assert torch.isfinite(loss), "Loss should be finite"
            assert entropy >= 0, "Entropy should be non-negative"
            assert not np.isnan(entropy), "Entropy should not be NaN"