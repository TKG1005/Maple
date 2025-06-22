from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .base import BaseAlgorithm


class ReinforceAlgorithm(BaseAlgorithm):
    """Monte Carlo policy gradient (REINFORCE) implementation."""

    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        batch: Dict[str, torch.Tensor],
    ) -> float:
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32)

        logits = model(obs)
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = log_probs[torch.arange(len(actions)), actions]
        loss = -(selected * rewards).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return float(loss.detach())
