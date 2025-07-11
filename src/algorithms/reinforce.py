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
        optimizer: torch.optim.Optimizer | None,
        batch: Dict[str, torch.Tensor],
    ) -> float:
        # Get device from model parameters
        device = next(model.parameters()).device
        
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)

        # Handle both enhanced networks (return tuple) and basic networks (return single tensor)
        net_output = model(obs)
        if isinstance(net_output, tuple):
            logits, _ = net_output  # Enhanced network returns (logits, hidden_state)
        else:
            logits = net_output  # Basic network returns logits only
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = log_probs[torch.arange(len(actions)), actions]
        loss = -(selected * rewards).mean()

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        
        return float(loss.detach())
