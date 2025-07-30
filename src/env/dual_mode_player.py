"""Dual-mode player supporting both WebSocket and IPC communication.

This module provides a player implementation that can switch between
WebSocket (online) and IPC (local) communication modes while maintaining
full compatibility with the existing poke-env Player interface.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from poke_env.ps_client.server_configuration import ServerConfiguration

from .env_player import EnvPlayer
from src.sim.battle_communicator import CommunicatorFactory, BattleCommunicator


class DualModeEnvPlayer(EnvPlayer):
    """Environment player supporting both online and local battle modes.
    
    This player extends EnvPlayer to support dual communication modes:
    - Online mode: Traditional WebSocket communication with Pokemon Showdown servers
    - Local mode: High-speed IPC communication with embedded Node.js processes
    
    The mode is determined at initialization and affects all battle communication.
    """
    
    def __init__(
        self,
        env: Any,
        player_id: str,
        mode: str = "local",  # "local" or "online"
        server_configuration: Optional[ServerConfiguration] = None,
        ipc_script_path: str = "sim/ipc-battle-server.js",
        **kwargs: Any
    ) -> None:
        """Initialize dual-mode player.
        
        Args:
            env: The PokemonEnv instance
            player_id: Unique identifier for this player
            mode: Communication mode ("local" for IPC, "online" for WebSocket)
            server_configuration: WebSocket server config (required for online mode)
            ipc_script_path: Path to Node.js IPC server script (for local mode)
            **kwargs: Additional arguments passed to parent EnvPlayer
        """
        self.mode = mode
        self.ipc_script_path = ipc_script_path
        self._communicator: Optional[BattleCommunicator] = None
        self._original_ps_client = None
        self.player_id = player_id  # Set player_id early for logging
        
        self._logger = logging.getLogger(__name__)
        
        # Initialize communicator BEFORE parent class to prevent WebSocket connection in local mode
        if mode == "local":
            self._logger.info(f"Pre-initializing IPC communicator for player {self.player_id}")
            self._initialize_communicator()
        
        # Initialize parent class
        if mode == "online":
            if server_configuration is None:
                raise ValueError("server_configuration required for online mode")
            super().__init__(
                env=env,
                player_id=player_id,
                server_configuration=server_configuration,
                **kwargs
            )
        else:
            # For local mode, we need to prevent WebSocket connection
            # Use a working server configuration but don't actually connect
            if server_configuration is None:
                # Create a valid server configuration for compatibility
                server_configuration = ServerConfiguration("localhost", 8000)
            
            # Skip parent initialization for local mode - handle manually
            self._setup_local_mode_player(env, player_id, server_configuration, **kwargs)
        
        # Initialize communicator for online mode only (local mode already done)
        if mode == "online":
            self._initialize_communicator()
    
    def _setup_local_mode_player(self, env: Any, player_id: str, server_configuration: ServerConfiguration, **kwargs: Any) -> None:
        """Set up player for local mode without WebSocket connection."""
        # For local mode, we still need to initialize the parent class properly
        # but we'll override the WebSocket communication methods
        
        try:
            # Initialize parent class normally - the override should prevent actual WebSocket usage
            super().__init__(
                env=env,
                player_id=player_id,
                server_configuration=server_configuration,
                **kwargs
            )
            self._logger.info(f"✅ Local mode player initialized for {player_id} (WebSocket methods overridden)")
            
        except Exception as e:
            self._logger.error(f"❌ Failed to initialize parent class for local mode: {e}")
            raise RuntimeError(f"Local mode player initialization failed: {e}") from e
    
    def _initialize_communicator(self) -> None:
        """Initialize the appropriate communicator based on mode."""
        if self.mode == "local":
            self._logger.info(f"Initializing local IPC communicator for player {self.player_id}")
            try:
                # Initialize IPC communicator
                self._communicator = CommunicatorFactory.create_communicator(
                    mode="ipc",
                    node_script_path=self.ipc_script_path,
                    logger=self._logger
                )
                self._logger.info(f"Successfully initialized IPC communicator for {self.player_id}")
                
                # Override WebSocket methods to use IPC
                self._override_websocket_methods()
                
            except Exception as e:
                self._logger.error(f"❌ CRITICAL: Failed to initialize IPC communicator for {self.player_id}")
                self._logger.error(f"❌ Error details: {type(e).__name__}: {e}")
                self._logger.error(f"❌ Node.js script path: {self.ipc_script_path}")
                self._logger.error(f"❌ Working directory: {os.getcwd()}")
                self._logger.error("❌ Local mode requires working IPC communicator - WebSocket fallback is disabled")
                raise RuntimeError(f"Local mode IPC initialization failed for {self.player_id}: {e}") from e
        else:
            self._logger.info(f"Using online WebSocket mode for player {self.player_id}")
            # For online mode, don't initialize custom communicator - let poke-env handle it
    
    def _override_websocket_methods(self) -> None:
        """Override WebSocket-related methods for local IPC mode.
        
        This method replaces the WebSocket client functionality with IPC
        communication while maintaining the same interface.
        """
        if self.mode != "local":
            return
        
        # Store original ps_client for potential restoration
        self._original_ps_client = getattr(self, 'ps_client', None)
        
        # Replace ps_client with IPC wrapper
        self.ps_client = IPCClientWrapper(self._communicator, self._logger)
        
        # Override key methods that would use WebSocket
        self._original_listen = getattr(self, 'listen', None)
        self.listen = self._ipc_listen
        
        self._logger.debug(f"Overridden WebSocket methods for local IPC mode (player {self.player_id})")
    
    async def _ipc_listen(self) -> None:
        """IPC-based listen method that replaces WebSocket listening."""
        self._logger.info(f"🎧 Starting IPC listen mode for player {self.player_id}")
        
        # For now, just log that we're in IPC mode and don't actually listen
        # The real battle communication will happen through the environment's queue system
        self._logger.info(f"✅ IPC listen mode active for player {self.player_id} (no WebSocket connection)")
        
        # Keep the method running to satisfy poke-env expectations
        while True:
            await asyncio.sleep(1)  # Prevent busy waiting
    
    async def _ensure_communicator_connected(self) -> None:
        """Ensure the communicator is connected before use."""
        if not self._communicator:
            raise RuntimeError("Communicator not initialized")
        
        if not await self._communicator.is_alive():
            await self._communicator.connect()
    
    async def close_connection(self) -> None:
        """Close the communicator connection."""
        if self._communicator:
            await self._communicator.disconnect()
        
        # Call parent close if in online mode
        if self.mode == "online" and hasattr(super(), 'close_connection'):
            await super().close_connection()


class IPCClientWrapper:
    """Wrapper that provides ps_client-like interface for IPC communication.
    
    This class mimics the poke_env ps_client interface but uses IPC
    communication instead of WebSocket, allowing seamless integration
    with existing EnvPlayer code.
    """
    
    def __init__(self, communicator: BattleCommunicator, logger: logging.Logger):
        self.communicator = communicator
        self.logger = logger
        self._battle_counter = 0
    
    async def send_message(self, message: str, battle_tag: str) -> None:
        """Send a battle command via IPC.
        
        Args:
            message: The battle command (e.g., "move 1", "switch 2")
            battle_tag: The battle identifier
        """
        try:
            # Ensure communicator is connected
            if not await self.communicator.is_alive():
                await self.communicator.connect()
            
            # Format message for IPC protocol
            ipc_message = {
                "type": "battle_command",
                "battle_id": battle_tag,
                "player": "p1",  # This should be dynamically determined
                "command": message
            }
            
            await self.communicator.send_message(ipc_message)
            self.logger.debug(f"Sent IPC battle command: {message} to {battle_tag}")
            
        except Exception as e:
            self.logger.error(f"Failed to send IPC message: {e}")
            raise
    
    async def create_battle(self, battle_tag: str, format_id: str, players: list) -> None:
        """Create a new battle via IPC.
        
        Args:
            battle_tag: Unique battle identifier
            format_id: Pokemon Showdown format (e.g., "gen9randombattle")
            players: List of player configurations
        """
        try:
            if not await self.communicator.is_alive():
                await self.communicator.connect()
            
            ipc_message = {
                "type": "create_battle",
                "battle_id": battle_tag,
                "format": format_id,
                "players": players,
                "seed": None  # Can be added for reproducible battles
            }
            
            await self.communicator.send_message(ipc_message)
            self.logger.debug(f"Created IPC battle: {battle_tag} ({format_id})")
            
        except Exception as e:
            self.logger.error(f"Failed to create IPC battle: {e}")
            raise
    
    async def get_battle_state(self, battle_tag: str) -> dict:
        """Get current battle state via IPC.
        
        Args:
            battle_tag: Battle identifier
            
        Returns:
            Dictionary containing current battle state
        """
        try:
            if not await self.communicator.is_alive():
                await self.communicator.connect()
            
            ipc_message = {
                "type": "get_battle_state",
                "battle_id": battle_tag
            }
            
            await self.communicator.send_message(ipc_message)
            response = await self.communicator.receive_message()
            
            if response.get("type") == "battle_state" and response.get("success"):
                return response.get("state", {})
            else:
                raise RuntimeError(f"Failed to get battle state: {response}")
                
        except Exception as e:
            self.logger.error(f"Failed to get IPC battle state: {e}")
            raise
    
    async def ping(self) -> bool:
        """Test IPC connection with ping.
        
        Returns:
            True if ping successful, False otherwise
        """
        try:
            if not await self.communicator.is_alive():
                await self.communicator.connect()
            
            ping_message = {
                "type": "ping",
                "timestamp": asyncio.get_event_loop().time()
            }
            
            await self.communicator.send_message(ping_message)
            response = await asyncio.wait_for(
                self.communicator.receive_message(), 
                timeout=5.0
            )
            
            return (response.get("type") == "pong" and 
                    response.get("success", False))
                    
        except Exception as e:
            self.logger.error(f"IPC ping failed: {e}")
            return False


# Utility functions for mode management
def create_dual_mode_players(
    env: Any,
    mode: str = "local",
    server_configuration: Optional[ServerConfiguration] = None,
    **kwargs: Any
) -> tuple[DualModeEnvPlayer, DualModeEnvPlayer]:
    """Create a pair of dual-mode players for battle.
    
    Args:
        env: The PokemonEnv instance
        mode: Communication mode ("local" or "online")
        server_configuration: Required for online mode
        **kwargs: Additional arguments passed to players
        
    Returns:
        Tuple of two DualModeEnvPlayer instances
    """
    player_0 = DualModeEnvPlayer(
        env=env,
        player_id="player_0",
        mode=mode,
        server_configuration=server_configuration,
        **kwargs
    )
    
    player_1 = DualModeEnvPlayer(
        env=env,
        player_id="player_1", 
        mode=mode,
        server_configuration=server_configuration,
        **kwargs
    )
    
    return player_0, player_1


def get_mode_from_config(config: dict) -> str:
    """Extract communication mode from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Communication mode ("local" or "online")
    """
    return config.get("battle_mode", "local")


def validate_mode_configuration(mode: str, config: dict) -> None:
    """Validate that the configuration is appropriate for the specified mode.
    
    Args:
        mode: Communication mode
        config: Configuration dictionary
        
    Raises:
        ValueError: If configuration is invalid for the mode
    """
    if mode == "online":
        if "pokemon_showdown" not in config:
            raise ValueError("Online mode requires 'pokemon_showdown' configuration")
        if "servers" not in config["pokemon_showdown"]:
            raise ValueError("Online mode requires server configuration")
    elif mode == "local":
        local_config = config.get("local_mode", {})
        if "max_processes" in local_config and local_config["max_processes"] < 1:
            raise ValueError("local_mode.max_processes must be at least 1")
    else:
        raise ValueError(f"Unsupported mode: {mode}. Use 'local' or 'online'")