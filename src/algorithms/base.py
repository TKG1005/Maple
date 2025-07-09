from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

import torch
import torch.nn as nn


class BaseAlgorithm(ABC):
    """Interface for reinforcement learning algorithms."""

    @abstractmethod
    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        batch: Dict[str, torch.Tensor],
    ) -> float:
        """Update ``model`` using ``batch`` and return the training loss."""
        raise NotImplementedError
