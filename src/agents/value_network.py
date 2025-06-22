import torch
import torch.nn as nn
import gymnasium as gym

class ValueNetwork(nn.Module):
    """Simple feedforward value network returning state value."""

    def __init__(self, observation_space: gym.Space, hidden_size: int = 128) -> None:
        super().__init__()
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("observation_space must be gym.spaces.Box")

        obs_dim = int(observation_space.shape[0])

        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)
