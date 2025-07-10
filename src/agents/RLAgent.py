from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import logging

from .MapleAgent import MapleAgent
from src.env.pokemon_env import PokemonEnv
from src.algorithms import BaseAlgorithm, ReinforceAlgorithm


class RLAgent(MapleAgent):
    """Reinforcement learning agent holding a model and optimizer."""

    def __init__(
        self,
        env: PokemonEnv,
        policy_net: nn.Module,
        value_net: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        algorithm: BaseAlgorithm | None = None,
    ) -> None:
        super().__init__(env)
        self.policy_net = policy_net
        self.value_net = value_net
        self.optimizer = optimizer
        self.algorithm = algorithm or ReinforceAlgorithm()
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        
        # Check if networks support hidden states (LSTM/Attention)
        self.has_hidden_states = hasattr(policy_net, 'use_lstm') and (policy_net.use_lstm or (hasattr(policy_net, 'use_attention') and policy_net.use_attention))
        if value_net is not None:
            self.has_hidden_states = self.has_hidden_states and (hasattr(value_net, 'use_lstm') and (value_net.use_lstm or (hasattr(value_net, 'use_attention') and value_net.use_attention)))
        self.policy_hidden = None
        self.value_hidden = None

    def select_action(
        self, observation: np.ndarray, action_mask: np.ndarray
    ) -> np.ndarray:
        """Return action probabilities respecting ``action_mask``."""
        player_id = self._get_player_id()
        self._logger.debug("%s: %s", player_id, action_mask)
        obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=next(self.policy_net.parameters()).device)
        
        # Handle LSTM/Attention networks with hidden states
        if self.has_hidden_states:
            # Add batch dimension if needed for LSTM
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            # Use stored hidden state
            logits, self.policy_hidden = self.policy_net(obs_tensor, self.policy_hidden)
            # Remove batch dimension if we added it
            if logits.dim() == 2 and logits.size(0) == 1:
                logits = logits.squeeze(0)
        else:
            # Basic network without hidden states
            logits = self.policy_net(obs_tensor)
        
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

        # Show the received mask to make debugging of invalid actions easier
        player_id = self._get_player_id()
        self._logger.debug("%s: %s", player_id, action_mask)

        probs = self.select_action(observation, action_mask)
        rng = getattr(self.env, "rng", np.random.default_rng())
        action = int(rng.choice(len(probs), p=probs))
        self._logger.debug("%s: chosen action = %s", self.__class__.__name__, action)
        return action

    def reset_hidden_states(self) -> None:
        """Reset hidden states for LSTM/Attention networks at episode boundaries."""
        if self.has_hidden_states:
            self.policy_hidden = None
            self.value_hidden = None
            self._logger.debug("Hidden states reset for %s", self._get_player_id())

    def update(self, batch: dict[str, torch.Tensor]) -> float:
        """Delegate a training update to the underlying algorithm."""
        if self.optimizer is None:
            # This agent is frozen (e.g., self-play opponent), no learning
            return 0.0
        return self.algorithm.update(self.policy_net, self.optimizer, batch)


__all__ = ["RLAgent"]
