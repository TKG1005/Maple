import torch
import torch.nn as nn
import gymnasium as gym


class PolicyNetwork(nn.Module):
    """Simple feedforward policy network."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, hidden_size: int = 128) -> None:
        super().__init__()
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("observation_space must be gym.spaces.Box")
        if not isinstance(action_space, gym.spaces.Discrete):
            raise TypeError("action_space must be gym.spaces.Discrete")

        obs_dim = int(observation_space.shape[0])
        action_dim = int(action_space.n)

        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

