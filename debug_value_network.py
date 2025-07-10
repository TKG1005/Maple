#!/usr/bin/env python3
"""Debug script for ValueNetwork output shapes."""

import sys
import torch
import gymnasium as gym
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.agents.enhanced_networks import LSTMValueNetwork

def debug_value_network():
    """Debug the ValueNetwork output shapes."""
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(622,))  # Same as in the error
    
    # Create LSTM value network
    value_net = LSTMValueNetwork(obs_space, hidden_size=128, use_lstm=True)
    
    # Test with single observation
    obs_t = torch.randn(1, 622)  # [1, obs_dim]
    
    print(f"Input obs_t shape: {obs_t.shape}")
    
    # Forward pass
    value_output = value_net(obs_t)
    
    print(f"value_output type: {type(value_output)}")
    if isinstance(value_output, tuple):
        value_t, hidden = value_output
        print(f"value_t shape: {value_t.shape}")
        print(f"hidden type: {type(hidden)}")
    else:
        print(f"value_output shape: {value_output.shape}")
    
    # Check the MLP output before squeeze
    print("\n--- Internal debugging ---")
    
    # Simulate the forward pass step by step
    x = obs_t
    print(f"Initial x shape: {x.shape}")
    
    if value_net.use_lstm:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        print(f"After unsqueeze x shape: {x.shape}")
        
        hidden = value_net.init_hidden(x.size(0), x.device)
        lstm_out, new_hidden = value_net.lstm(x, hidden)
        print(f"LSTM output shape: {lstm_out.shape}")
        
        x = lstm_out[:, -1, :]  # Use the last output
        print(f"After taking last output x shape: {x.shape}")
    
    mlp_output = value_net.mlp(x)
    print(f"MLP output shape: {mlp_output.shape}")
    
    squeezed = mlp_output.squeeze(-1)
    print(f"After squeeze(-1) shape: {squeezed.shape}")

if __name__ == "__main__":
    debug_value_network()