from __future__ import annotations

from typing import Dict

import math
import torch
import torch.nn as nn

from .base import BaseAlgorithm


def compute_ppo_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    *,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    return_entropy: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, float]:
    """Return clipped PPO loss value, optionally with entropy.

    This function works with real :mod:`torch` tensors.  When ``torch`` is not
    available (as in the lightweight test environment), the inputs can be Python
    lists of floats.  In that case, operations fall back to ``math`` and list
    comprehensions.
    
    Args:
        new_log_probs: Log probabilities from current policy
        old_log_probs: Log probabilities from old policy
        advantages: Advantage estimates
        returns: Return estimates
        values: Value estimates
        clip_range: PPO clipping parameter
        value_coef: Value loss coefficient
        entropy_coef: Entropy coefficient
        return_entropy: If True, return (loss, entropy) tuple
        
    Returns:
        Loss value, or (loss, entropy) if return_entropy=True
    """
    if hasattr(torch, "exp"):
        ratio = torch.exp(new_log_probs - old_log_probs)
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        value_loss = 0.5 * (returns - values).pow(2).mean()
        entropy = -(new_log_probs.exp() * new_log_probs).sum(-1).mean()
        total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
        
        if return_entropy:
            return total_loss, float(entropy.detach())
        return total_loss

    ratio = [math.exp(n - o) for n, o in zip(new_log_probs, old_log_probs)]
    clipped_ratio = [max(min(r, 1.0 + clip_range), 1.0 - clip_range) for r in ratio]
    policy_terms = [min(r * a, c * a) for r, c, a in zip(ratio, clipped_ratio, advantages)]
    policy_loss = -sum(policy_terms) / len(policy_terms)
    value_terms = [(ret - val) ** 2 for ret, val in zip(returns, values)]
    value_loss = 0.5 * sum(value_terms) / len(value_terms)
    entropy_terms = [-math.exp(n) * n for n in new_log_probs]
    entropy = sum(entropy_terms) / len(entropy_terms)
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    
    if return_entropy:
        return total_loss, entropy
    return total_loss


class PPOAlgorithm(BaseAlgorithm):
    """Proximal Policy Optimization with clipping."""

    def __init__(
        self,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
    ) -> None:
        self.clip_range = float(clip_range)
        self.value_coef = float(value_coef)
        self.entropy_coef = float(entropy_coef)

    def update(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None,
        batch: Dict[str, torch.Tensor],
    ) -> float | tuple[float, float]:
        # Handle different model formats
        if isinstance(model, (tuple, list)) and len(model) == 2:
            policy_net, value_net = model
            device = next(policy_net.parameters()).device
        elif isinstance(model, dict):
            policy_net = model["policy"]
            value_net = model["value"]
            device = next(policy_net.parameters()).device
        else:
            # Legacy single model case - assume it's the policy network only
            policy_net = model
            value_net = None
            device = next(model.parameters()).device
        
        obs = torch.as_tensor(batch["observations"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.int64, device=device)
        old_log_probs = torch.as_tensor(batch["old_log_probs"], dtype=torch.float32, device=device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=device)

        # Handle both enhanced networks (return tuple) and basic networks (return single tensor)
        net_output = policy_net(obs)
        if isinstance(net_output, tuple):
            logits, _ = net_output  # Enhanced network returns (logits, hidden_state)
        else:
            logits = net_output  # Basic network returns logits only
        log_probs = torch.log_softmax(logits, dim=-1)
        selected = log_probs[torch.arange(len(actions)), actions]

        # Compute values using value network if available, otherwise use pre-computed values
        if value_net is not None:
            # Use value network to compute current values (enables value network learning)
            value_output = value_net(obs)
            if isinstance(value_output, tuple):
                values, _ = value_output  # Enhanced network returns (values, hidden_state)
            else:
                values = value_output  # Basic network returns values only
            # Ensure values is 1D
            if values.dim() > 1:
                values = values.squeeze(-1)
        else:
            # Fallback to pre-computed values (legacy behavior)
            values = torch.as_tensor(batch.get("values", torch.zeros_like(returns)), dtype=torch.float32, device=device)

        result = compute_ppo_loss(
            selected,
            old_log_probs,
            advantages,
            returns,
            values,
            clip_range=self.clip_range,
            value_coef=self.value_coef,
            entropy_coef=self.entropy_coef,
            return_entropy=True,
        )
        
        loss, entropy = result

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping for stability
            if isinstance(model, (tuple, list)):
                # Clip gradients for both networks
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=5.0)
                if value_net is not None:
                    torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=5.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
        
        return float(loss.detach()), float(entropy)


__all__ = ["compute_ppo_loss", "PPOAlgorithm"]
