from __future__ import annotations

from abc import ABC, abstractmethod

class RewardBase(ABC):
    """Interface for reward calculation."""

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state at the start of an episode."""
        raise NotImplementedError

    @abstractmethod
    def calc(self, *args, **kwargs):
        """Compute and return the reward."""
        raise NotImplementedError


__all__ = [
    "RewardBase",
    "HPDeltaReward",
    "CompositeReward",
]

from .hp_delta import HPDeltaReward
from .composite import CompositeReward
