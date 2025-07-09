"""Reward normalization utilities for stable training."""

from __future__ import annotations

import numpy as np
from typing import Optional


class RewardNormalizer:
    """Running statistics-based reward normalizer for stable RL training.
    
    This class maintains running mean and variance of rewards to normalize
    rewards to have zero mean and unit variance, which helps stabilize
    training, especially with PPO and other policy gradient methods.
    """
    
    def __init__(self, epsilon: float = 1e-8):
        """Initialize the reward normalizer.
        
        Args:
            epsilon: Small constant to avoid division by zero
        """
        self.running_mean: float = 0.0
        self.running_var: float = 1.0
        self.count: int = 0
        self.epsilon: float = epsilon
    
    def update(self, reward: float) -> None:
        """Update running statistics with a new reward.
        
        Uses Welford's online algorithm for numerically stable
        running mean and variance calculation.
        
        Args:
            reward: The new reward value to incorporate
        """
        self.count += 1
        delta = reward - self.running_mean
        self.running_mean += delta / self.count
        delta2 = reward - self.running_mean
        self.running_var += delta * delta2
    
    def normalize(self, reward: float) -> float:
        """Normalize a reward using current running statistics.
        
        Args:
            reward: The reward to normalize
            
        Returns:
            Normalized reward with zero mean and unit variance
        """
        if self.count <= 1:
            return reward
            
        std = np.sqrt(self.running_var / (self.count - 1))
        return (reward - self.running_mean) / (std + self.epsilon)
    
    def get_stats(self) -> dict[str, float]:
        """Get current normalization statistics.
        
        Returns:
            Dictionary containing mean, std, and count
        """
        std = np.sqrt(self.running_var / max(1, self.count - 1)) if self.count > 1 else 1.0
        return {
            'mean': self.running_mean,
            'std': std,
            'count': self.count
        }
    
    def reset(self) -> None:
        """Reset the normalizer to initial state."""
        self.running_mean = 0.0
        self.running_var = 1.0
        self.count = 0


class WindowedRewardNormalizer:
    """Sliding window-based reward normalizer.
    
    This class maintains a fixed-size window of recent rewards
    for normalization, which can be more adaptive to changing
    reward distributions during training.
    """
    
    def __init__(self, window_size: int = 1000, epsilon: float = 1e-8):
        """Initialize the windowed reward normalizer.
        
        Args:
            window_size: Size of the sliding window for statistics
            epsilon: Small constant to avoid division by zero
        """
        self.window_size = window_size
        self.epsilon = epsilon
        self.rewards: list[float] = []
    
    def update(self, reward: float) -> None:
        """Update the reward window with a new reward.
        
        Args:
            reward: The new reward value to incorporate
        """
        self.rewards.append(reward)
        if len(self.rewards) > self.window_size:
            self.rewards.pop(0)
    
    def normalize(self, reward: float) -> float:
        """Normalize a reward using windowed statistics.
        
        Args:
            reward: The reward to normalize
            
        Returns:
            Normalized reward based on window statistics
        """
        if len(self.rewards) < 2:
            return reward
            
        mean = np.mean(self.rewards)
        std = np.std(self.rewards)
        return (reward - mean) / (std + self.epsilon)
    
    def get_stats(self) -> dict[str, float]:
        """Get current normalization statistics.
        
        Returns:
            Dictionary containing mean, std, and count
        """
        if len(self.rewards) < 2:
            return {'mean': 0.0, 'std': 1.0, 'count': len(self.rewards)}
            
        return {
            'mean': float(np.mean(self.rewards)),
            'std': float(np.std(self.rewards)),
            'count': len(self.rewards)
        }
    
    def reset(self) -> None:
        """Reset the normalizer to initial state."""
        self.rewards.clear()