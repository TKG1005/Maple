from __future__ import annotations

import logging
import numpy as np
from typing import Union, Literal

from .MapleAgent import MapleAgent
from src.env.pokemon_env import PokemonEnv


class EpsilonGreedyWrapper(MapleAgent):
    """
    ε-greedy exploration wrapper for agents.
    
    Wraps an agent to provide ε-greedy exploration with configurable decay strategies.
    During exploration, random valid actions are selected with probability ε.
    
    Attributes:
        epsilon: Current exploration rate
        epsilon_start: Initial exploration rate
        epsilon_end: Final exploration rate
        decay_steps: Number of steps for decay
        decay_strategy: Decay strategy ('linear' or 'exponential')
        step_count: Current step count for decay calculation
    """
    
    def __init__(
        self,
        wrapped_agent: MapleAgent,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.1,
        decay_steps: int = 10000,
        decay_strategy: Literal["linear", "exponential"] = "linear",
        decay_mode: Literal["step", "episode"] = "step",
        env: PokemonEnv | None = None,
        initial_episode_count: int = 0,
    ) -> None:
        """
        Initialize ε-greedy wrapper.
        
        Args:
            wrapped_agent: Agent to wrap with ε-greedy exploration
            epsilon_start: Initial exploration rate (default: 1.0)
            epsilon_end: Final exploration rate (default: 0.1)
            decay_steps: Number of steps/episodes to decay from start to end (default: 10000)
            decay_strategy: Decay strategy ('linear' or 'exponential')
            decay_mode: Decay mode ('step' for per-action decay, 'episode' for per-episode decay)
            env: Environment instance (if None, uses wrapped_agent.env)
            initial_episode_count: Starting episode count (for cross-episode persistence)
        """
        # Use environment from wrapped agent if not provided
        env = env or getattr(wrapped_agent, 'env', None)
        if env is None:
            raise ValueError("Environment must be provided or wrapped agent must have env attribute")
        
        super().__init__(env)
        
        self.wrapped_agent = wrapped_agent
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.decay_steps = max(1, decay_steps)  # Prevent division by zero
        self.decay_strategy = decay_strategy
        self.decay_mode = decay_mode
        self.step_count = 0  # For step-based decay
        self.episode_count = initial_episode_count  # For episode-based decay, can start from external count
        self.epsilon = epsilon_start
        
        # Initialize epsilon based on current episode count
        self._update_epsilon()
        
        # Logging setup
        self._logger = logging.getLogger(__name__)
        self._exploration_stats = {
            'total_actions': 0,
            'random_actions': 0,
            'exploration_rate': 0.0
        }
        
        # Ensure wrapped agent uses the same environment
        if hasattr(wrapped_agent, 'env'):
            wrapped_agent.env = self.env
    
    def _update_epsilon(self) -> None:
        """Update epsilon value based on decay strategy and mode."""
        # Choose the appropriate counter based on decay mode
        if self.decay_mode == "episode":
            current_count = self.episode_count
        else:  # step mode
            current_count = self.step_count
        
        if current_count >= self.decay_steps:
            self.epsilon = self.epsilon_end
            return
        
        progress = current_count / self.decay_steps
        
        if self.decay_strategy == "linear":
            # Linear decay: ε = start - (start - end) * progress
            self.epsilon = self.epsilon_start - (self.epsilon_start - self.epsilon_end) * progress
        elif self.decay_strategy == "exponential":
            # Exponential decay: ε = end + (start - end) * exp(-α * progress)
            # Using standard exponential decay rate
            alpha = 5.0  # Standard exponential decay rate
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * np.exp(-alpha * progress)
        else:
            raise ValueError(f"Unknown decay strategy: {self.decay_strategy}")
        
        # Ensure epsilon stays within bounds
        self.epsilon = max(self.epsilon_end, min(self.epsilon_start, self.epsilon))
    
    def select_action(
        self, observation: np.ndarray, action_mask: np.ndarray
    ) -> Union[np.ndarray, int]:
        """
        Select action using ε-greedy strategy with on-policy distribution mixing.
        
        Mixes the policy distribution with uniform distribution to maintain on-policy learning:
        mixed_prob = (1 - ε) * policy_prob + ε * uniform_prob
        
        Args:
            observation: State observation
            action_mask: Available actions mask
            
        Returns:
            Action probabilities (mixed distribution) or action index
        """
        # Update step count and epsilon (for step-based decay)
        if self.decay_mode == "step":
            self.step_count += 1
            self._update_epsilon()
        else:
            # Episode-based decay: epsilon updated only when episode_count changes
            self._update_epsilon()
        
        # Get policy probabilities from wrapped agent
        policy_result = self.wrapped_agent.select_action(observation, action_mask)
        
        # Get valid action indices
        try:
            valid_indices = [i for i, flag in enumerate(action_mask) if flag]
        except Exception:
            valid_indices = []
        
        if not valid_indices:
            # No valid actions, let wrapped agent handle it
            self._logger.warning("No valid actions available, delegating to wrapped agent")
            return policy_result
        
        # Handle both probability and action index returns from wrapped agent
        if isinstance(policy_result, np.ndarray):
            # Wrapped agent returns probabilities
            policy_probs = policy_result.copy()
        else:
            # Wrapped agent returns action index, convert to one-hot probabilities
            policy_probs = np.zeros(len(action_mask), dtype=np.float32)
            if 0 <= policy_result < len(action_mask):
                policy_probs[policy_result] = 1.0
        
        # Create uniform distribution over valid actions
        uniform_probs = np.zeros(len(action_mask), dtype=np.float32)
        uniform_probs[valid_indices] = 1.0 / len(valid_indices)
        
        # Mix policy and uniform distributions: (1-ε) * policy + ε * uniform
        mixed_probs = (1.0 - self.epsilon) * policy_probs + self.epsilon * uniform_probs
        
        # Ensure probabilities sum to 1 and are non-negative
        mixed_probs = np.maximum(mixed_probs, 0.0)
        prob_sum = np.sum(mixed_probs)
        if prob_sum > 0:
            mixed_probs = mixed_probs / prob_sum
        else:
            # Fallback to uniform if something went wrong
            mixed_probs = uniform_probs.copy()
        
        # Update exploration statistics based on how much we deviated from policy
        self._exploration_stats['total_actions'] += 1
        # Consider it "exploration" if epsilon contributed significantly
        if self.epsilon > 0.1:  # Threshold for considering it exploration
            self._exploration_stats['random_actions'] += 1
        self._exploration_stats['exploration_rate'] = (
            self._exploration_stats['random_actions'] / self._exploration_stats['total_actions']
        )
        
        # Return mixed probabilities or sampled action based on original wrapped agent behavior
        if isinstance(policy_result, np.ndarray):
            return mixed_probs
        else:
            # Sample from mixed distribution
            rng = getattr(self.env, "rng", np.random.default_rng())
            return int(rng.choice(len(mixed_probs), p=mixed_probs))
    
    def act(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """Sample an action index using ε-greedy strategy."""
        # Use select_action to determine action
        result = self.select_action(observation, action_mask)
        
        if isinstance(result, np.ndarray):
            # Result is probabilities, sample from them
            rng = getattr(self.env, "rng", np.random.default_rng())
            action = int(rng.choice(len(result), p=result))
        else:
            # Result is already an action index
            action = int(result)
        
        self._logger.debug(
            "%s: chosen action = %s (ε=%.3f, step=%d)", 
            self.__class__.__name__, action, self.epsilon, self.step_count
        )
        return action
    
    def reset_episode_stats(self) -> None:
        """Reset exploration statistics for a new episode."""
        stats = self._exploration_stats.copy()
        self._exploration_stats = {
            'total_actions': 0,
            'random_actions': 0,
            'exploration_rate': 0.0
        }
        
        # Update episode count for episode-based decay
        if self.decay_mode == "episode":
            self.episode_count += 1
            self._update_epsilon()
        
        return stats
    
    def get_exploration_stats(self) -> dict:
        """Get current exploration statistics."""
        # Calculate decay progress based on decay mode
        if self.decay_mode == "episode":
            current_count = self.episode_count
        else:
            current_count = self.step_count
        
        # Calculate random action rate (percentage of actions that were random)
        random_action_rate = (
            self._exploration_stats['random_actions'] / self._exploration_stats['total_actions']
            if self._exploration_stats['total_actions'] > 0 else 0.0
        )
        
        return {
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'episode_count': self.episode_count,
            'decay_mode': self.decay_mode,
            'decay_strategy': self.decay_strategy,
            'decay_progress': min(1.0, current_count / self.decay_steps),
            'decay_steps': self.decay_steps,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'random_action_rate': random_action_rate,  # New detailed metric
            **self._exploration_stats
        }
    
    def log_exploration_stats(self) -> None:
        """Log current exploration statistics."""
        stats = self.get_exploration_stats()
        self._logger.info(
            "Exploration stats - ε: %.3f, steps: %d, random actions: %d/%d (%.1f%%)",
            stats['epsilon'],
            stats['step_count'],
            stats['random_actions'],
            stats['total_actions'],
            stats['exploration_rate'] * 100
        )
    
    # Delegate other methods to wrapped agent
    def choose_team(self, observation) -> str:
        """Delegate team selection to wrapped agent."""
        return self.wrapped_agent.choose_team(observation)
    
    def __getattr__(self, name):
        """Delegate unknown attributes to wrapped agent."""
        # Special handling for methods that might be called by training loop
        if hasattr(self.wrapped_agent, name):
            return getattr(self.wrapped_agent, name)
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")


__all__ = ["EpsilonGreedyWrapper"]