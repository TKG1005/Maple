"""Optimizer and scheduler utilities for training."""
from __future__ import annotations

import logging
from typing import Optional, Dict, Any

import torch
from torch import optim
from torch.optim import lr_scheduler

logger = logging.getLogger(__name__)


def save_training_state(
    checkpoint_path: str,
    episode: int,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[lr_scheduler._LRScheduler] = None,
) -> None:
    """Save complete training state including networks, optimizer, and scheduler.
    
    Args:
        checkpoint_path: Path to save the checkpoint
        episode: Current episode number
        policy_net: Policy network
        value_net: Value network
        optimizer: Optimizer state
        scheduler: Optional learning rate scheduler
    """
    checkpoint = {
        "episode": episode,
        "policy": policy_net.state_dict(),
        "value": value_net.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved training state to {checkpoint_path}")


def load_training_state(
    checkpoint_path: str,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Optional[lr_scheduler._LRScheduler] = None,
    device: str = "cpu",
    reset_optimizer: bool = False,
) -> int:
    """Load training state from checkpoint with device transfer support.
    
    Args:
        checkpoint_path: Path to load checkpoint from
        policy_net: Policy network to load state into
        value_net: Value network to load state into
        optimizer: Optimizer to load state into
        scheduler: Optional scheduler to load state into
        device: Target device for loading
        reset_optimizer: If True, skip loading optimizer state
        
    Returns:
        Episode number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if "policy" in checkpoint and "value" in checkpoint:
        # New format with separate policy and value
        policy_net.load_state_dict(checkpoint["policy"])
        value_net.load_state_dict(checkpoint["value"])
        episode = checkpoint.get("episode", 0)
        
        # Load optimizer state if not resetting
        if not reset_optimizer and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            # Transfer optimizer state to correct device
            transfer_optimizer_state_to_device(optimizer, device)
        
        # Load scheduler state if present
        if scheduler is not None and "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
    else:
        # Legacy format - single state dict
        policy_net.load_state_dict(checkpoint)
        value_net.load_state_dict(checkpoint)
        episode = 0
    
    logger.info(f"Loaded training state from {checkpoint_path}, episode: {episode}")
    return episode


def transfer_optimizer_state_to_device(optimizer: optim.Optimizer, device: str) -> None:
    """Transfer optimizer state to specified device.
    
    Args:
        optimizer: Optimizer whose state needs to be transferred
        device: Target device ('cpu', 'cuda', 'mps')
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_config: Dict[str, Any]
) -> Optional[lr_scheduler._LRScheduler]:
    """Create learning rate scheduler from configuration.
    
    Args:
        optimizer: Optimizer to attach scheduler to
        scheduler_config: Scheduler configuration dict
        
    Returns:
        Scheduler instance or None if not enabled
    """
    if not scheduler_config.get("enabled", False):
        return None
    
    scheduler_type = scheduler_config.get("type", "step")
    
    if scheduler_type == "step":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get("step_size", 1000),
            gamma=scheduler_config.get("gamma", 0.9)
        )
    elif scheduler_type == "exponential":
        return lr_scheduler.ExponentialLR(
            optimizer,
            gamma=scheduler_config.get("gamma", 0.99)
        )
    elif scheduler_type == "cosine":
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=scheduler_config.get("T_max", 1000),
            eta_min=scheduler_config.get("eta_min", 0)
        )
    elif scheduler_type == "reduce_on_plateau":
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get("mode", "min"),
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 10),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")