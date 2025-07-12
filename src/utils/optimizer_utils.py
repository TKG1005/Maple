"""Utilities for optimizer and scheduler state management across devices."""

import torch
from typing import Dict, Any, Optional


def transfer_optimizer_state_to_device(
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> None:
    """Transfer optimizer state tensors to specified device.
    
    This is necessary when loading a checkpoint saved on a different device
    (e.g., GPU checkpoint loaded on CPU or vice versa).
    
    Args:
        optimizer: Optimizer whose state needs to be transferred
        device: Target device for the state tensors
    """
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def save_training_state(
    filepath: str,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    episode: int = 0,
    **kwargs
) -> None:
    """Save complete training state including models, optimizer, and scheduler.
    
    Args:
        filepath: Path to save the checkpoint
        policy_net: Policy network
        value_net: Value network
        optimizer: Optimizer
        scheduler: Learning rate scheduler (optional)
        episode: Current episode number
        **kwargs: Additional state to save
    """
    state = {
        "policy": policy_net.state_dict(),
        "value": value_net.state_dict(),
        "optimizer": optimizer.state_dict(),
        "episode": episode,
    }
    
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    
    # Add any additional state
    state.update(kwargs)
    
    torch.save(state, filepath)


def load_training_state(
    filepath: str,
    policy_net: torch.nn.Module,
    value_net: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any] = None,
    device: torch.device = torch.device("cpu"),
    strict: bool = True
) -> Dict[str, Any]:
    """Load complete training state including models, optimizer, and scheduler.
    
    Args:
        filepath: Path to load the checkpoint from
        policy_net: Policy network to load state into
        value_net: Value network to load state into
        optimizer: Optimizer to load state into
        scheduler: Learning rate scheduler to load state into (optional)
        device: Device to map the loaded tensors to
        strict: Whether to strictly enforce that the keys match
        
    Returns:
        Dictionary containing loaded state information
    """
    # Load checkpoint with map_location to handle device changes
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model states
    policy_net.load_state_dict(checkpoint["policy"], strict=strict)
    value_net.load_state_dict(checkpoint["value"], strict=strict)
    
    # Load optimizer state
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Transfer optimizer state to correct device
        transfer_optimizer_state_to_device(optimizer, device)
    
    # Load scheduler state if available
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    # Return additional state information
    result = {
        "episode": checkpoint.get("episode", 0),
        "has_optimizer": "optimizer" in checkpoint,
        "has_scheduler": "scheduler" in checkpoint,
    }
    
    # Include any additional state that was saved
    for key in checkpoint:
        if key not in ["policy", "value", "optimizer", "scheduler", "episode"]:
            result[key] = checkpoint[key]
    
    return result


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "step",
    **kwargs
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create a learning rate scheduler.
    
    Args:
        optimizer: Optimizer to create scheduler for
        scheduler_type: Type of scheduler to create
        **kwargs: Additional arguments for the scheduler
        
    Returns:
        Learning rate scheduler or None if type is "none"
    """
    if scheduler_type == "none" or scheduler_type is None:
        return None
    
    if scheduler_type == "step":
        step_size = kwargs.get("step_size", 1000)
        gamma = kwargs.get("gamma", 0.9)
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma
        )
    elif scheduler_type == "exponential":
        gamma = kwargs.get("gamma", 0.99)
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=gamma
        )
    elif scheduler_type == "cosine":
        T_max = kwargs.get("T_max", 1000)
        eta_min = kwargs.get("eta_min", 0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=T_max, eta_min=eta_min
        )
    elif scheduler_type == "reduce_on_plateau":
        mode = kwargs.get("mode", "min")
        factor = kwargs.get("factor", 0.5)
        patience = kwargs.get("patience", 10)
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")