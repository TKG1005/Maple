import torch
import torch.nn as nn
import gymnasium as gym

class ValueNetwork(nn.Module):
    """Feedforward value network with configurable layers."""

    def __init__(self, observation_space: gym.Space, hidden_size: int = 128, use_2layer: bool = True) -> None:
        super().__init__()
        if not isinstance(observation_space, gym.spaces.Box):
            raise TypeError("observation_space must be gym.spaces.Box")

        obs_dim = int(observation_space.shape[0])
        
        if use_2layer:
            # 2-layer MLP: obs_dim -> hidden_size -> hidden_size*2 -> hidden_size -> 1
            self.model = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size * 2),
                nn.ReLU(),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )
        else:
            # Original 1-layer MLP for comparison
            self.model = nn.Sequential(
                nn.Linear(obs_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x).squeeze(-1)
