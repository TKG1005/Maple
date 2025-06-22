from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .MapleAgent import MapleAgent
from src.env.pokemon_env import PokemonEnv
from src.algorithms import BaseAlgorithm, ReinforceAlgorithm


class RLAgent(MapleAgent):
    """Reinforcement learning agent holding a model and optimizer."""

    def __init__(
        self,
        env: PokemonEnv,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        algorithm: BaseAlgorithm | None = None,
    ) -> None:
        super().__init__(env)
        self.model = model
        self.optimizer = optimizer
        self.algorithm = algorithm or ReinforceAlgorithm()

    def select_action(
        self, observation: np.ndarray, action_mask: np.ndarray
    ) -> np.ndarray:
        """Return action probabilities respecting ``action_mask``."""
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32)
        logits = self.model(obs_tensor)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool)
        masked_logits = logits.clone()
        if mask_tensor.any():
            masked_logits[~mask_tensor] = -float("inf")
            probs = torch.softmax(masked_logits, dim=-1)
        else:
            probs = torch.full_like(logits, fill_value=1.0 / logits.numel())
        return probs.detach().cpu().numpy()

    def act(self, observation: np.ndarray, action_mask: np.ndarray) -> int:
        """Sample an action index according to the policy."""
        # Print the received mask to make debugging of invalid actions easier
        print(f"{self.__class__.__name__}: available mask = {action_mask}")
        probs = self.select_action(observation, action_mask)
        rng = getattr(self.env, "rng", np.random.default_rng())
        action = int(rng.choice(len(probs), p=probs))
        print(f"{self.__class__.__name__}: chosen action = {action}")
        return action

    def update(self, batch: dict[str, torch.Tensor]) -> float:
        """Delegate a training update to the underlying algorithm."""
        return self.algorithm.update(self.model, self.optimizer, batch)


__all__ = ["RLAgent"]
