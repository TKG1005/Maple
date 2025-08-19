"""Fix for poke-env duplicate logging issue.

This module patches poke-env's logger creation to prevent duplicate handlers
when multiple players with the same username are created.
"""

import logging
from typing import Optional
from logging import Logger


def create_logger_without_duplicate_handlers(username: str, log_level: Optional[int]) -> Logger:
    """Create a logger without adding duplicate handlers.
    
    This function checks if a logger already has handlers before adding new ones,
    preventing duplicate log messages when multiple PSClient instances are created
    with the same username.
    """
    logger = logging.getLogger(username)
    
    # Only add handler if the logger doesn't already have one
    if not logger.handlers:
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    
    # Always update log level if provided
    if log_level is not None:
        logger.setLevel(log_level)
        # Also update handler levels
        for handler in logger.handlers:
            handler.setLevel(log_level)
    
    return logger


def patch_poke_env_logging():
    """Monkey patch poke-env's PSClient._create_logger method to fix duplicate logging."""
    try:
        from poke_env.ps_client.ps_client import PSClient
        
        # Store original method for potential restoration
        PSClient._original_create_logger = PSClient._create_logger
        
        # Replace with our fixed version
        def _create_logger(self, log_level: Optional[int]) -> Logger:
            return create_logger_without_duplicate_handlers(self.username, log_level)
        
        PSClient._create_logger = _create_logger
        
        print("Successfully patched poke-env logging to prevent duplicate messages")
        return True
        
    except ImportError:
        print("Failed to import poke-env, logging patch not applied")
        return False
