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
        self._env = env  # Store environment reference for battle queue access
        
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
        """Initialize the appropriate communicator and IPCClientWrapper based on mode."""
        if self.mode == "local":
            self._logger.info(f"Local IPC mode configured for player {self.player_id}")
            try:
                # Create IPC communicator for both Phase 3 and Phase 4
                self._communicator = CommunicatorFactory.create_communicator(
                    mode="ipc", 
                    node_script_path=self.ipc_script_path,
                    logger=self._logger
                )
                
                # Phase 2: Create IPCClientWrapper with PSClient-compatible interface
                account_config = getattr(self, 'account_configuration', None)
                server_config = getattr(self, 'server_configuration', None) 
                
                self.ipc_client_wrapper = IPCClientWrapper(
                    account_configuration=account_config,
                    server_configuration=server_config,
                    communicator=self._communicator,
                    logger=self._logger
                )
                
                if self.full_ipc:
                    # Phase 4: Full IPC mode requires immediate connection and WebSocket override
                    self._logger.info(f"🔌 Phase 4: IPCClientWrapper created for {self.player_id} (mandatory connection)")
                    # Override will be activated in _establish_full_ipc_connection()
                else:
                    # Phase 3: IPC ready but not mandatory
                    self._logger.info(f"✅ IPCClientWrapper ready for {self.player_id} (Phase 3 demonstration mode)")
                
            except Exception as e:
                if self.full_ipc:
                    # Phase 4: Full IPC mode cannot fallback to WebSocket
                    self._logger.error(f"❌ Full IPC mode requires working IPCClientWrapper for {self.player_id}: {e}")
                    raise RuntimeError(f"Full IPC initialization failed: {e}") from e
                else:
                    # Phase 3: Fallback to WebSocket is allowed
                    self._logger.warning(f"⚠️ IPCClientWrapper setup failed for {self.player_id}: {e}")
                    self._logger.info(f"🔄 Falling back to WebSocket mode for {self.player_id}")
        else:
            self._logger.info(f"Using online WebSocket mode for player {self.player_id}")
            # For online mode, don't initialize custom communicator - let poke-env handle it
            self.ipc_client_wrapper = None  # Phase 2: No IPC wrapper for online mode
    
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
                            # Safely handle response that might be string or dict
                            response_type = None
                            response_success = False
                            
                            if isinstance(response, dict):
                                response_type = response.get("type")
                                response_success = response.get("success", False)
                            elif isinstance(response, str):
                                try:
                                    import json
                                    parsed_response = json.loads(response)
                                    response_type = parsed_response.get("type")
                                    response_success = parsed_response.get("success", False)
                                except json.JSONDecodeError:
                                    self._logger.warning(f"⚠️ IPC response is not valid JSON for {self.player_id}: {response}")
                                    response_type = "unknown"
                            
                            if response_type == "pong" and response_success:
                                self._logger.info(f"🏓 IPC pong received successfully for {self.player_id}")
                            else:
                                self._logger.warning(f"⚠️ Unexpected IPC response for {self.player_id}: type={response_type}, success={response_success}, response={response}")
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
        if not self.full_ipc or self.mode != "local" or not hasattr(self, 'ipc_client_wrapper'):
            return
        
        try:
            # Synchronously establish IPC connection for Phase 4
            from poke_env.concurrency import POKE_LOOP
            import asyncio
            
            async def establish_connection():
                try:
                    # Phase 4: Ensure IPC communicator is connected
                    self._logger.info(f"🔌 Phase 4: Establishing mandatory IPC connection for {self.player_id}")
                    if not await self.ipc_client_wrapper.is_alive():
                        await self.ipc_client_wrapper.communicator.connect()

                    # Phase 4: Simulate login (authentication bypassed)
                    await self.ipc_client_wrapper.log_in([])
                    self._logger.info(f"🔐 Phase 4: Authentication completed for {self.player_id}")

                    # Phase 4: Test IPC connection with ping-pong
                    self._logger.info(f"📤 Phase 4: Testing IPC connection for {self.player_id}")
                    ping_success = await self.ipc_client_wrapper.ping()
                    if not ping_success:
                        raise RuntimeError("IPC ping-pong failed")
                    self._logger.info(f"🏓 Phase 4: IPC ping-pong successful for {self.player_id}")

                    # Phase 4: Start listening for IPC messages
                    self._logger.info(f"🔗 Phase 4: Starting IPCClientWrapper.listen() for {self.player_id}")
                    listen_task = asyncio.create_task(self.ipc_client_wrapper.listen())

                    # Phase 4: Override WebSocket methods now that IPC is verified
                    self._override_websocket_methods()
                    self._logger.info(f"✅ Phase 4: WebSocket overridden with IPC for {self.player_id}")

                    return True
                except Exception as e:
                    self._logger.error(f"❌ Phase 4: Could not establish full IPC connection for {self.player_id}: {e}")
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
        if self.mode != "local" or not hasattr(self, 'ipc_client_wrapper'):
            return
        
        # Store original ps_client for potential restoration
        self._original_ps_client = getattr(self, 'ps_client', None)
        
        # Phase 2: Replace ps_client with IPCClientWrapper (created in _initialize_communicator)
        self.ps_client = self.ipc_client_wrapper
        
        # Set parent player reference for poke-env integration
        self.ipc_client_wrapper.set_parent_player(self)
        
        # Override poke-env internal methods to prevent WebSocket operations
        self._override_poke_env_internals()
        
        self._logger.debug(f"Overridden ps_client with IPCClientWrapper for player {self.player_id}")
        
    def _override_poke_env_internals(self) -> None:
        """Override poke-env internal methods to prevent WebSocket operations."""
        
        # Phase 2: Override the listening coroutine to use IPCClientWrapper.listen()
        async def ipc_listening_coroutine():
            """IPC listening coroutine using IPCClientWrapper."""
            self._logger.info(f"🎧 IPC listening coroutine started for player {self.player_id}")
            try:
                # Use IPCClientWrapper.listen() instead of WebSocket listening
                if hasattr(self, 'ipc_client_wrapper'):
                    await self.ipc_client_wrapper.listen()
                else:
                    self._logger.error(f"❌ No IPCClientWrapper available for {self.player_id}")
                    # Fallback to mock behavior
                    while True:
                        await asyncio.sleep(10)
                        if hasattr(self, '_should_stop_listening') and self._should_stop_listening:
                            break
            except Exception as e:
                self._logger.error(f"❌ IPC listening coroutine failed for {self.player_id}: {e}")
                
        # Replace the listening coroutine
        self._listening_coroutine = ipc_listening_coroutine
        
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
        
        # Override accept_challenges for IPC mode
        async def ipc_accept_challenges(opponent, n_challenges=1, packed_team=None):
            """IPC version of accept_challenges."""
            self._logger.info(f"🏟️ IPC accept_challenges called for {self.player_id}")
            # In IPC mode, we don't need to accept challenges - battles are created directly
            # This is just a placeholder to prevent errors
            return
            
        self.accept_challenges = ipc_accept_challenges
        
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
    
    async def battle_against(self, opponent, n_battles=1):
        """Override battle_against for IPC mode.
        
        In IPC mode, we create battles directly through IPC instead of using
        the WebSocket challenge/accept mechanism.
        """
        if self.mode != "local" or not self.full_ipc:
            # Use parent class implementation for WebSocket mode
            return await super().battle_against(opponent, n_battles)
        
        self._logger.info(f"🎮 IPC battle_against called for {self.player_id} vs {opponent.username}")
        
        for i in range(n_battles):
            # Generate unique battle ID
            battle_id = f"battle-{self.player_id}-{opponent.username}-{i}-{asyncio.get_event_loop().time()}"
            
            # Create battle via IPC
            await self._create_ipc_battle(battle_id, opponent)
            
            # Wait for battle to complete
            await self._wait_for_battle_completion(battle_id)
    
    async def _create_ipc_battle(self, battle_id: str, opponent):
        """Create a battle through IPC."""
        try:
            # Get battle format
            battle_format = self._format if hasattr(self, '_format') else "gen9randombattle"
            
            # Prepare player configurations
            # Convert team objects to strings if needed
            p1_team = None
            if hasattr(self, '_team') and self._team is not None:
                # Get the packed team string
                if hasattr(self._team, 'yield_team'):
                    p1_team = self._team.yield_team()
                else:
                    p1_team = str(self._team)
            
            p2_team = None
            if hasattr(opponent, '_team') and opponent._team is not None:
                # Get the packed team string
                if hasattr(opponent._team, 'yield_team'):
                    p2_team = opponent._team.yield_team()
                else:
                    p2_team = str(opponent._team)
            
            players = [
                {
                    "id": "p1",
                    "name": self.username,
                    "team": p1_team
                },
                {
                    "id": "p2", 
                    "name": opponent.username,
                    "team": p2_team
                }
            ]
            
            # Create battle via IPC
            await self.ipc_client_wrapper.create_battle(battle_id, battle_format, players)
            
            # Register battle with both players
            self._battles[battle_id] = None  # Will be populated when battle starts
            opponent._battles[battle_id] = None
            
            # Notify both players about the battle
            await self._handle_battle_start(battle_id)
            await opponent._handle_battle_start(battle_id)
            
            self._logger.info(f"✅ IPC battle created: {battle_id}")
            
        except Exception as e:
            self._logger.error(f"❌ Failed to create IPC battle: {e}")
            raise
    
    async def _handle_battle_start(self, battle_id: str):
        """Handle battle start notification."""
        try:
            # Create battle object
            from poke_env.environment.abstract_battle import AbstractBattle
            from poke_env.environment.battle import Battle
            
            # Get the generation from the format
            gen = 9  # Default to gen 9
            if hasattr(self, '_format'):
                # Extract generation number from format string (e.g., "gen9bssregi" -> 9)
                import re
                match = re.search(r'gen(\d+)', self._format)
                if match:
                    gen = int(match.group(1))
            
            # Initialize battle with the given ID and generation
            battle = Battle(battle_id, self.username, self._logger, gen=gen)
            self._battles[battle_id] = battle
            
            # Queue battle for environment processing
            if hasattr(self, '_env') and hasattr(self._env, '_battle_queues'):
                await self._env._battle_queues[self.player_id].put(battle)
                self._logger.debug(f"📥 Battle {battle_id} queued for {self.player_id}")
            
        except Exception as e:
            self._logger.error(f"❌ Failed to handle battle start: {e}")
            raise
    
    async def _wait_for_battle_completion(self, battle_id: str):
        """Wait for a battle to complete."""
        while battle_id in self._battles:
            battle = self._battles[battle_id]
            if battle and battle.finished:
                break
            await asyncio.sleep(0.1)
        
        self._logger.info(f"🏁 Battle {battle_id} completed")


class IPCClientWrapper:
    """Wrapper that provides ps_client-like interface for IPC communication.
    
    This class mimics the poke_env ps_client interface but uses IPC
    communication instead of WebSocket, allowing seamless integration
    with existing EnvPlayer code. Enhanced in Phase 1 to support PSClient
    compatibility with AccountConfiguration and authentication.
    """
    
    def __init__(self, 
                 account_configuration=None,
                 *,
                 server_configuration=None,
                 communicator: BattleCommunicator = None, 
                 logger: logging.Logger = None,
                 # Legacy support
                 **kwargs):
        """Initialize IPCClientWrapper with PSClient-compatible interface.
        
        Args:
            account_configuration: AccountConfiguration object (PSClient compatibility)
            server_configuration: ServerConfiguration object (PSClient compatibility)
            communicator: BattleCommunicator instance for IPC
            logger: Logger instance
            **kwargs: Additional arguments for compatibility
        """
        # PSClient-compatible attributes
        if account_configuration is not None:
            # New PSClient-compatible initialization
            from poke_env.ps_client.account_configuration import AccountConfiguration
            if not isinstance(account_configuration, AccountConfiguration):
                raise TypeError(f"account_configuration must be AccountConfiguration, got {type(account_configuration)}")
            self.account_configuration = account_configuration
        else:
            # Legacy initialization or auto-generate
            if communicator is not None and logger is not None:
                # Legacy mode: IPCClientWrapper(communicator, logger)
                from poke_env.ps_client.account_configuration import AccountConfiguration
                self.account_configuration = AccountConfiguration("IPCPlayer", None)
            else:
                raise ValueError("Either account_configuration or (communicator, logger) must be provided")
        
        self.server_configuration = server_configuration
        self.communicator = communicator
        self.logger = logger or logging.getLogger(__name__)
        
        # PSClient-compatible attributes
        self.closed = False
        self.state = "connected"
        self.logged_in = asyncio.Event()  # PSClient compatibility
        self._battle_counter = 0
        
        # IPC-specific attributes
        self._message_queue = asyncio.Queue()
        self._listen_task = None
        self._parent_player = None  # Will be set by DualModeEnvPlayer
        
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
    
    # Phase 1: PSClient-compatible methods
    
    async def listen(self) -> None:
        """PSClient.listen() compatible method for IPC communication.
        
        This method mimics PSClient.listen() by establishing IPC connection
        and starting the message processing loop, similar to WebSocket listening.
        """
        try:
            self.logger.info(f"🎧 Starting IPC listen mode for {self.username}")
            
            # Ensure communicator is connected
            if not await self.communicator.is_alive():
                await self.communicator.connect()
                self.logger.info(f"✅ IPC connection established for {self.username}")
            
            # Trigger login sequence (IPC doesn't need challenge string)
            await self.log_in([])
            
            # Start message processing loop (similar to PSClient WebSocket loop)
            self._listen_task = asyncio.create_task(self._message_loop())
            
            # Keep method running like PSClient.listen()
            await self._listen_task
            
        except Exception as e:
            self.logger.error(f"❌ IPC listen failed for {self.username}: {e}")
            raise
    
    async def _message_loop(self) -> None:
        """Main message processing loop (similar to PSClient WebSocket loop)."""
        try:
            while not self.closed:
                try:
                    # Receive message from IPC communicator
                    message = await self.communicator.receive_message()
                    
                    # DEBUG_TEAMPREVIEW: Check for teampreview messages
                    message_str = str(message)
                    if "teampreview" in message_str.lower() or "poke|" in message_str:
                        self.logger.info(f"🔍 [DEBUG_TEAMPREVIEW] IPCClientWrapper received: {message}")
                    else:
                        self.logger.debug(f"📥 IPC received message: {message}")
                    
                    # Process message (IPC message routing based on type)
                    # message is a dict with keys {type, data, battle_id, …}
                    await self._route_ipc_message(message)
                    
                except asyncio.CancelledError:
                    self.logger.info(f"🛑 IPC message loop cancelled for {self.username}")
                    break
                except Exception as e:
                    self.logger.error(f"❌ Error in IPC message loop: {e}")
                    await asyncio.sleep(1)  # Brief pause before retrying
                    
        except Exception as e:
            self.logger.error(f"❌ IPC message loop failed: {e}")
        finally:
            self.logger.info(f"🔚 IPC message loop ended for {self.username}")
    
    async def log_in(self, split_message=None) -> None:
        """PSClient.log_in() compatible method for IPC authentication.
        
        Args:
            split_message: Challenge string (not used in IPC, kept for compatibility)
        """
        try:
            # IPC environment doesn't need authentication like online Showdown
            # but we simulate the login process for PSClient compatibility
            self.logger.info(f"🔐 IPC login for {self.username} (authentication bypassed)")
            
            # Set logged_in event to signal successful "authentication"
            self.logged_in.set()
            self.logger.info(f"✅ IPC login successful for {self.username}")
            
        except Exception as e:
            self.logger.error(f"❌ IPC login failed for {self.username}: {e}")
            raise
    
    async def wait_for_login(self) -> None:
        """PSClient.wait_for_login() compatible method."""
        await self.logged_in.wait()
    
    @property
    def username(self) -> str:
        """Get username from account configuration (PSClient compatibility)."""
        return self.account_configuration.username
    
    async def is_alive(self) -> bool:
        """Check if IPC connection is alive (PSClient compatibility)."""
        return await self.communicator.is_alive()
    
    def set_parent_player(self, player) -> None:
        """Set parent player reference for poke-env integration."""
        self._parent_player = player
    
    async def _route_ipc_message(self, message) -> None:
        """Route IPC message based on type (protocol vs control).
        
        This method determines whether the message is a Showdown protocol message
        that should be forwarded to poke-env, or an IPC control message that
        should be handled internally.
        
        Args:
            message: Raw IPC message from communicator
        """
        try:
            # Parse message type
            is_showdown, parsed_message = self._parse_message_type(message)
            
            # DEBUG_TEAMPREVIEW: Log routing decision for teampreview
            message_str = str(message)
            if "teampreview" in message_str.lower() or "poke|" in message_str:
                self.logger.info(f"🔍 [DEBUG_TEAMPREVIEW] Routing decision - is_showdown: {is_showdown}, message_type: {parsed_message.get('type', 'unknown')}")
            
            if is_showdown:
                # Handle Showdown protocol messages (including battle tag and request lines)
                await self._handle_showdown_message(parsed_message)
            else:
                # Handle IPC control message
                await self._handle_ipc_control_message(parsed_message)
                
        except Exception as e:
            self.logger.error(f"❌ Error routing IPC message: {e}")
    
    def _parse_message_type(self, message) -> tuple[bool, dict]:
        """Parse message and determine if it's Showdown protocol or IPC control.
        
        Args:
            message: Raw message from IPC communicator
            
        Returns:
            Tuple of (is_showdown_protocol, parsed_message)
        """
        if not isinstance(message, dict):
            return False, {}
        
        msg_type = message.get("type")
        
        # Showdown protocol messages
        if msg_type == "protocol":
            return True, message
        
        # IPC control messages
        elif msg_type in ["battle_created", "battle_update", "player_registered", "battle_end", "error", "pong"]:
            return False, message
        
        else:
            # Unknown message type - treat as IPC control
            self.logger.warning(f"⚠️ Unknown message type: {msg_type}")
            return False, message
    
    async def _handle_showdown_message(self, message: dict) -> None:
        """Handle Showdown protocol message by forwarding to poke-env.
        
        Args:
            message: Parsed protocol message dict
        """
        try:
            protocol_data = message.get("data", "")
            # Debug: show raw protocol_data forwarded from IPC (should include >battle tag)
            self.logger.debug(f"[IPC_DEBUG_PY] Forwarding raw protocol data to poke-env:\n{protocol_data}")
            
            # DEBUG_TEAMPREVIEW: Check for teampreview in protocol data
            if "teampreview" in str(protocol_data).lower() or "poke|" in str(protocol_data):
                self.logger.info(f"🔍 [DEBUG_TEAMPREVIEW] About to forward to poke-env: {protocol_data}")
            
            if protocol_data and self._parent_player:
                # Always forward full protocol_data (with tag and body) to poke-env
                self.logger.debug(f"[IPC_DEBUG_PY] Sending to poke-env._handle_message: {protocol_data}")
                await self._parent_player._handle_message(protocol_data)
                
            else:
                self.logger.warning(f"⚠️ No protocol data or parent player: {message}")
                
        except Exception as e:
            self.logger.error(f"❌ Error forwarding to poke-env: {e}")
    
    async def _handle_ipc_control_message(self, message: dict) -> None:
        """Handle IPC control message.
        
        Args:
            message: Parsed IPC control message dict
        """
        msg_type = message.get("type")
        
        if msg_type == "battle_created":
            battle_id = message.get('battle_id')
            self.logger.info(f"🏟️ Battle created: {battle_id}")
            # Notify parent player about battle creation
            if self._parent_player and hasattr(self._parent_player, '_handle_battle_start'):
                # Run in the event loop to avoid blocking
                asyncio.create_task(self._parent_player._handle_battle_start(battle_id))
        elif msg_type == "battle_update":
            # Handle battle updates (teampreview, turn requests, etc.)
            battle_id = message.get('battle_id')
            updates = message.get('updates', '')
            
            # DEBUG_TEAMPREVIEW: Check for teampreview in battle updates
            if "teampreview" in str(updates).lower() or "poke|" in str(updates):
                self.logger.info(f"🔍 [DEBUG_TEAMPREVIEW] Battle update contains teampreview: battle_id={battle_id}, updates={updates}")
                
            if updates and self._parent_player:
                # Forward as Showdown protocol message
                if "teampreview" in str(updates).lower() or "poke|" in str(updates):
                    self.logger.info(f"📤 [DEBUG_TEAMPREVIEW] Forwarding battle update to poke-env._handle_message(): {updates}")
                await self._parent_player._handle_message(updates)
        elif msg_type == "player_registered":
            self.logger.info(f"👤 Player registered: {message.get('player_id')}")
        elif msg_type == "battle_end":
            result = message.get("result")
            winner = message.get("winner")
            self.logger.info(f"🏁 Battle ended: result={result}, winner={winner}")
        elif msg_type == "error":
            error_msg = message.get("error_message", "Unknown error")
            self.logger.error(f"❌ IPC error: {error_msg}")
        elif msg_type == "pong":
            self.logger.debug(f"🏓 Pong received")
        else:
            self.logger.debug(f"🔍 Unhandled IPC control message: {msg_type}")
    
    async def close(self) -> None:
        """Close IPC connection (PSClient compatibility)."""
        self.closed = True
        
        # Cancel listen task
        if self._listen_task and not self._listen_task.done():
            self._listen_task.cancel()
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
        
        # Disconnect communicator
        if self.communicator:
            await self.communicator.disconnect()
        
        self.logger.info(f"🔌 IPC connection closed for {self.username}")


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