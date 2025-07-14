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
        # Handle different model formats - for REINFORCE we only need policy network
        if isinstance(model, (tuple, list)):
            policy_net = model[0]  # Use the first network (policy)
            device = next(policy_net.parameters()).device
        elif isinstance(model, dict):
            policy_net = model["policy"]
            device = next(policy_net.parameters()).device
        else:
            # Single model case - assume it's the policy network
            policy_net = model
            device = next(model.parameters()).device
        
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=device)

        # Handle both enhanced networks (return tuple) and basic networks (return single tensor)
        net_output = policy_net(obs)
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
            if isinstance(model, (tuple, list)):
                # Clip gradients for policy network only (REINFORCE doesn't use value network)
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        
        return float(loss.detach())
