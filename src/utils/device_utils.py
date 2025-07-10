"""Device selection utilities for GPU support."""

import torch
import logging

logger = logging.getLogger(__name__)


def get_device(prefer_gpu: bool = True, device_name: str = "auto") -> torch.device:
    """Get the best available device for PyTorch operations.
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU when available
        device_name: Specific device name ("auto", "cpu", "cuda", "mps") or device string
        
    Returns:
        torch.device: The selected device
    """
    if device_name == "cpu":
        device = torch.device("cpu")
        logger.info("Using CPU device (forced)")
        return device
    
    if device_name == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            return device
        else:
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = torch.device("cpu")
            return device
    
    if device_name == "mps":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using MPS (Apple Metal) device")
            return device
        else:
            logger.warning("MPS requested but not available, falling back to CPU")
            device = torch.device("cpu")
            return device
    
    if device_name == "auto" and prefer_gpu:
        # Auto-detect best GPU
        if torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Auto-selected CUDA device: {torch.cuda.get_device_name()}")
            return device
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Auto-selected MPS (Apple Metal) device")
            return device
        else:
            device = torch.device("cpu")
            logger.info("No GPU available, using CPU")
            return device
    
    # Try to parse as device string (e.g., "cuda:0", "cpu")
    try:
        device = torch.device(device_name)
        logger.info(f"Using specified device: {device}")
        return device
    except Exception as e:
        logger.warning(f"Invalid device specification '{device_name}': {e}, falling back to CPU")
        device = torch.device("cpu")
        return device


def transfer_to_device(tensor_or_module, device: torch.device):
    """Transfer tensor or module to the specified device.
    
    Args:
        tensor_or_module: PyTorch tensor or module to transfer
        device: Target device
        
    Returns:
        Transferred tensor or module
    """
    if tensor_or_module is None:
        return None
    
    try:
        return tensor_or_module.to(device)
    except Exception as e:
        logger.error(f"Failed to transfer to device {device}: {e}")
        raise


def get_device_info(device: torch.device) -> dict:
    """Get information about the specified device.
    
    Args:
        device: Device to get info for
        
    Returns:
        Dictionary with device information
    """
    info = {
        "device": str(device),
        "type": device.type,
    }
    
    if device.type == "cuda" and torch.cuda.is_available():
        info.update({
            "name": torch.cuda.get_device_name(device),
            "memory_total": torch.cuda.get_device_properties(device).total_memory,
            "memory_allocated": torch.cuda.memory_allocated(device),
            "memory_cached": torch.cuda.memory_reserved(device),
        })
    elif device.type == "mps" and torch.backends.mps.is_available():
        info.update({
            "name": "Apple Metal GPU",
            "driver_version": "MPS backend",
        })
    else:
        info.update({
            "name": "CPU",
        })
    
    return info