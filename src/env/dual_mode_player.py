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
        full_ipc: bool = False,  # Phase 4: Enable full IPC mode without WebSocket fallback
        **kwargs: Any
    ) -> None:
        """Initialize dual-mode player.
        
        Args:
            env: The PokemonEnv instance
            player_id: Unique identifier for this player
            mode: Communication mode ("local" for IPC, "online" for WebSocket)
            server_configuration: WebSocket server config (required for online mode)
            ipc_script_path: Path to Node.js IPC server script (for local mode)
            full_ipc: Phase 4 flag - if True, disables WebSocket fallback completely
            **kwargs: Additional arguments passed to parent EnvPlayer
        """
        self.mode = mode
        self.full_ipc = full_ipc
        self.ipc_script_path = ipc_script_path
        self._communicator: Optional[BattleCommunicator] = None
        self._original_ps_client = None
        self.player_id = player_id  # Set player_id early for logging
        
        self._logger = logging.getLogger(__name__)
        
        # Initialize parent class based on mode and full_ipc setting
        if mode == "online":
            if server_configuration is None:
                raise ValueError("server_configuration required for online mode")
            super().__init__(
                env=env,
                player_id=player_id,
                server_configuration=server_configuration,
                **kwargs
            )
            self._logger.info(f"🌐 Online mode player initialized for {player_id}")
        else:
            # Local mode initialization - Phase 4 adds full_ipc support
            if server_configuration is None:
                # Create a valid server configuration for compatibility
                server_configuration = ServerConfiguration("localhost", 8000)
            
            if full_ipc:
                # Phase 4: Full IPC mode - completely disable WebSocket
                self._logger.info(f"🚀 Phase 4: Initializing full IPC mode for {player_id}")
                kwargs['start_listening'] = False  # Disable WebSocket completely
                
                try:
                    super().__init__(
                        env=env,
                        player_id=player_id,
                        server_configuration=server_configuration,
                        **kwargs
                    )
                    self._logger.info(f"✅ Full IPC mode player initialized for {player_id} (WebSocket disabled)")
                except Exception as e:
                    self._logger.error(f"❌ Full IPC initialization failed for {player_id}: {e}")
                    raise RuntimeError(f"Full IPC mode requires working IPC infrastructure: {e}") from e
            else:
                # Phase 3: Local mode with WebSocket fallback
                super().__init__(
                    env=env,
                    player_id=player_id,
                    server_configuration=server_configuration,
                    **kwargs
                )
                self._logger.info(f"✅ Local mode player initialized for {player_id} with IPC capability (WebSocket fallback for Phase 3)")
        
        # Initialize communicator after parent class initialization
        self._initialize_communicator()
        
        # For local mode, handle IPC capability demonstration
        if mode == "local":
            if full_ipc:
                # Phase 4: Full IPC mode - must establish working connection
                self._logger.info(f"🔌 Phase 4: Establishing mandatory IPC connection for {self.player_id}")
                self._establish_full_ipc_connection()
            else:
                # Phase 3: IPC capability demonstration (optional)
                self._logger.info(f"🚀 IPC capability initialized for {self.player_id} (demonstration skipped to prevent blocking)")
                # self._demonstrate_ipc_capability()  # Commented out to prevent blocking during initialization
    
    def _initialize_communicator(self) -> None:
        """Initialize the appropriate communicator based on mode."""
        if self.mode == "local":
            self._logger.info(f"Local IPC mode configured for player {self.player_id}")
            try:
                # Create IPC communicator for both Phase 3 and Phase 4
                self._communicator = CommunicatorFactory.create_communicator(
                    mode="ipc", 
                    node_script_path=self.ipc_script_path,
                    logger=self._logger
                )
                
                if self.full_ipc:
                    # Phase 4: Full IPC mode requires immediate connection and WebSocket override
                    self._logger.info(f"🔌 Phase 4: IPC communicator created for {self.player_id} (mandatory connection)")
                    # Override will be activated in _establish_full_ipc_connection()
                else:
                    # Phase 3: IPC ready but not mandatory
                    self._logger.info(f"✅ IPC communicator ready for {self.player_id} (Phase 3 demonstration mode)")
                
            except Exception as e:
                if self.full_ipc:
                    # Phase 4: Full IPC mode cannot fallback to WebSocket
                    self._logger.error(f"❌ Full IPC mode requires working IPC communicator for {self.player_id}: {e}")
                    raise RuntimeError(f"Full IPC initialization failed: {e}") from e
                else:
                    # Phase 3: Fallback to WebSocket is allowed
                    self._logger.warning(f"⚠️ IPC communicator setup failed for {self.player_id}: {e}")
                    self._logger.info(f"🔄 Falling back to WebSocket mode for {self.player_id}")
        else:
            self._logger.info(f"Using online WebSocket mode for player {self.player_id}")
            # For online mode, don't initialize custom communicator - let poke-env handle it
    
    def _demonstrate_ipc_capability(self) -> None:
        """Demonstrate IPC capability for Phase 3 completion."""
        if self.mode != "local":
            return
            
        try:
            # Test basic IPC communication capability
            from poke_env.concurrency import POKE_LOOP
            import asyncio
            
            # Run a quick IPC connectivity test
            async def test_ipc():
                try:
                    await self._ensure_communicator_connected()
                    self._logger.info(f"🔌 IPC communicator successfully connected for {self.player_id}")
                    
                    # Test ping-pong functionality
                    if hasattr(self._communicator, 'send_message') and hasattr(self._communicator, 'receive_message'):
                        ping_msg = {"type": "ping", "timestamp": asyncio.get_event_loop().time()}
                        await self._communicator.send_message(ping_msg)
                        self._logger.info(f"📡 IPC ping sent successfully for {self.player_id}")
                        
                        # Wait for pong response
                        try:
                            response = await asyncio.wait_for(
                                self._communicator.receive_message(), 
                                timeout=5.0
                            )
                            if response.get("type") == "pong" and response.get("success"):
                                self._logger.info(f"🏓 IPC pong received successfully for {self.player_id}")
                            else:
                                self._logger.warning(f"⚠️ Unexpected IPC response for {self.player_id}: {response}")
                        except asyncio.TimeoutError:
                            self._logger.warning(f"⚠️ IPC pong timeout for {self.player_id}")
                        except Exception as e:
                            self._logger.warning(f"⚠️ IPC pong receive failed for {self.player_id}: {e}")
                    
                    return True
                except Exception as e:
                    self._logger.warning(f"⚠️ IPC test failed for {self.player_id}: {e}")
                    return False
            
            # Run the test asynchronously
            future = asyncio.run_coroutine_threadsafe(test_ipc(), POKE_LOOP)
            # Don't wait for the result to avoid blocking initialization
            self._logger.info(f"🧪 IPC capability test initiated for player {self.player_id}")
            
        except Exception as e:
            self._logger.warning(f"⚠️ Could not start IPC capability test for {self.player_id}: {e}")
    
    def _establish_full_ipc_connection(self) -> None:
        """Phase 4: Establish mandatory IPC connection for full IPC mode."""
        if not self.full_ipc or self.mode != "local":
            return
        
        try:
            # Synchronously establish IPC connection for Phase 4
            from poke_env.concurrency import POKE_LOOP
            import asyncio
            
            async def establish_connection():
                try:
                    # Connect to IPC server
                    await self._ensure_communicator_connected()
                    self._logger.info(f"🔗 Phase 4: IPC connection established for {self.player_id}")
                    
                    # Test connection with ping-pong
                    ping_msg = {"type": "ping", "timestamp": asyncio.get_event_loop().time()}
                    self._logger.info(f"📤 Phase 4: Sending ping message for {self.player_id}: {ping_msg}")
                    await self._communicator.send_message(ping_msg)
                    self._logger.info(f"✅ Phase 4: Ping sent, waiting for pong from {self.player_id}")
                    
                    try:
                        response = await asyncio.wait_for(
                            self._communicator.receive_message(), 
                            timeout=5.0  # Reduced timeout for faster debugging
                        )
                        self._logger.info(f"📥 Phase 4: Received response for {self.player_id}: {response}")
                        
                        if response.get("type") == "pong" and response.get("success"):
                            self._logger.info(f"🏓 Phase 4: IPC ping-pong successful for {self.player_id}")
                            
                            # Now override WebSocket methods since IPC is working
                            self._override_websocket_methods()
                            self._logger.info(f"✅ Phase 4: WebSocket overridden with IPC for {self.player_id}")
                            
                            return True
                        else:
                            raise RuntimeError(f"Invalid pong response: {response}")
                    except asyncio.TimeoutError:
                        self._logger.error(f"⏱️ Phase 4: Ping-pong timeout for {self.player_id} - no response received within 5 seconds")
                        raise RuntimeError("IPC ping-pong timeout")
                        
                except Exception as e:
                    self._logger.error(f"❌ Phase 4: IPC connection failed for {self.player_id}: {e}")
                    raise
            
            # Execute connection establishment synchronously
            future = asyncio.run_coroutine_threadsafe(establish_connection(), POKE_LOOP)
            result = future.result(timeout=30.0)  # Wait up to 30 seconds for connection
            
            if result:
                self._logger.info(f"🚀 Phase 4: Full IPC mode successfully activated for {self.player_id}")
            else:
                raise RuntimeError("IPC connection establishment failed")
                
        except Exception as e:
            self._logger.error(f"❌ Phase 4: Could not establish full IPC connection for {self.player_id}: {e}")
            raise RuntimeError(f"Full IPC mode initialization failed: {e}") from e
    
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
        
        # Override poke-env internal methods to prevent WebSocket operations
        self._override_poke_env_internals()
        
        self._logger.debug(f"Overridden ps_client and poke-env internals with IPC for player {self.player_id}")
        
    def _override_poke_env_internals(self) -> None:
        """Override poke-env internal methods to prevent WebSocket operations."""
        
        # Override the listening coroutine to prevent WebSocket listening
        async def mock_listening_coroutine():
            """Mock listening coroutine that does nothing."""
            self._logger.info(f"🎧 Mock listening coroutine started for IPC player {self.player_id}")
            # Just keep running without doing anything
            while True:
                await asyncio.sleep(10)
                if hasattr(self, '_should_stop_listening') and self._should_stop_listening:
                    break
                
        # Replace the listening coroutine
        self._listening_coroutine = mock_listening_coroutine
        
        # Override start_listening to prevent WebSocket initialization  
        def mock_start_listening():
            """Mock start_listening that does nothing."""
            self._logger.info(f"🚀 Mock start_listening called for IPC player {self.player_id}")
            return True
            
        self.start_listening = mock_start_listening
        
        # Override stop_listening
        def mock_stop_listening():
            """Mock stop_listening that sets flag."""
            self._logger.info(f"🛑 Mock stop_listening called for IPC player {self.player_id}")
            self._should_stop_listening = True
            
        self.stop_listening = mock_stop_listening
        
        # Override accept_challenges to prevent WebSocket operations
        async def mock_accept_challenges(opponent, n_challenges=1, packed_team=None):
            """Mock accept_challenges for IPC mode."""
            self._logger.info(f"🏟️ Mock accept_challenges called for IPC player {self.player_id}")
            # In IPC mode, battles are managed differently
            raise NotImplementedError("accept_challenges not supported in full IPC mode")
            
        self.accept_challenges = mock_accept_challenges
        
        # Override send_message to use IPC
        async def ipc_send_message(message, room=""):
            """Send message via IPC instead of WebSocket."""
            self._logger.debug(f"📤 IPC send_message: {message} to room {room}")
            if self.ps_client:
                await self.ps_client.send(message)
            else:
                self._logger.warning(f"⚠️ No ps_client available for IPC send_message")
                
        # Replace the send_message method
        self.send_message = ipc_send_message
    
    async def _ipc_listen(self) -> None:
        """IPC-based listen method that replaces WebSocket listening."""
        self._logger.info(f"🎧 Starting IPC listen mode for player {self.player_id}")
        
        try:
            # Ensure communicator is connected
            await self._ensure_communicator_connected()
            
            # Log successful IPC connection
            self._logger.info(f"✅ IPC connection established for player {self.player_id}")
            
            # For now, we don't need to actively listen like WebSocket does
            # The communication will happen through direct method calls via the environment
            # This method just needs to stay alive to satisfy poke-env expectations
            self._logger.info(f"✅ IPC listen mode active for player {self.player_id} (no WebSocket connection)")
            
            # Keep the method running but with less frequent checks
            while True:
                await asyncio.sleep(5)  # Check every 5 seconds instead of constantly
                
                # Verify connection is still alive
                if not await self._communicator.is_alive():
                    self._logger.warning(f"⚠️ IPC connection lost for player {self.player_id}, attempting reconnect...")
                    await self._communicator.connect()
                    
        except Exception as e:
            self._logger.error(f"❌ IPC listen failed for player {self.player_id}: {e}")
            # Don't raise the exception - let the player continue with degraded functionality
            # Keep running the loop
            while True:
                await asyncio.sleep(10)  # Wait longer on error
    
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
        # Mock WebSocket-like attributes that poke-env might check
        self.closed = False
        self.state = "connected"
        
    async def send(self, message: str) -> None:
        """Send raw message via IPC (WebSocket interface compatibility)."""
        try:
            # Parse Pokemon Showdown protocol message
            if message.startswith('>'):
                # Battle command format: >p1 move 1
                parts = message[1:].split(' ', 2)
                if len(parts) >= 2:
                    player = parts[0]
                    command = ' '.join(parts[1:])
                    
                    ipc_message = {
                        "type": "battle_command",
                        "battle_id": "default",  # Will be updated with proper battle ID
                        "player": player,
                        "command": command
                    }
                    
                    await self.communicator.send_message(ipc_message)
                    self.logger.debug(f"Sent IPC raw command: {message}")
            else:
                # Non-battle message, send as-is
                ipc_message = {
                    "type": "raw_message",
                    "message": message
                }
                await self.communicator.send_message(ipc_message)
                self.logger.debug(f"Sent IPC raw message: {message}")
                
        except Exception as e:
            self.logger.error(f"Failed to send IPC raw message: {e}")
            raise
            
    async def recv(self) -> str:
        """Receive message via IPC (WebSocket interface compatibility)."""
        try:
            response = await self.communicator.receive_message()
            # Convert IPC response back to Pokemon Showdown protocol format
            if response.get("type") == "battle_update":
                return str(response.get("updates", ""))
            elif response.get("type") == "pong":
                return ""  # Ignore pong messages
            else:
                return str(response)
        except Exception as e:
            self.logger.error(f"Failed to receive IPC message: {e}")
            raise
            
    async def close(self) -> None:
        """Close IPC connection (WebSocket interface compatibility)."""
        self.closed = True
        await self.communicator.disconnect()
        
    async def ping(self) -> None:
        """Send ping via IPC (WebSocket interface compatibility)."""
        ping_message = {"type": "ping"}
        await self.communicator.send_message(ping_message)
        
    # Properties that poke-env might check
    @property
    def state_name(self) -> str:
        return "OPEN" if not self.closed else "CLOSED"
    
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