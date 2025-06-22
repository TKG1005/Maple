from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import BaseAlgorithm


class DummyAlgorithm(BaseAlgorithm):
    """Algorithm placeholder that performs no updates."""

    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Dict[str, torch.Tensor],
    ) -> float:
        del model, optimizer, batch
        return 0.0
