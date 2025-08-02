"""IPC-based battle implementation for direct Pokemon Showdown integration."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Union

from poke_env.environment import Pokemon, Move
from src.env.custom_battle import CustomBattle
import asyncio
from src.sim.battle_communicator import BattleCommunicator


class IPCBattle(CustomBattle):
    """IPC-based battle that communicates directly with Node.js Pokemon Showdown process."""
    
    def __init__(
        self,
        battle_id: str,
        username: str,
        logger: logging.Logger,
        communicator: BattleCommunicator,
        player_id: str,
        format_id: str = 'gen9randombattle',
        gen: int = 9,
        save_replays: Union[str, bool] = False,
        env_player: Any = None,
    ) -> None:
        """Initialize IPC battle with minimal required state.
        
        Args:
            battle_id: Unique battle identifier
            username: Player username
            logger: Logger instance
            communicator: IPC communicator for Node.js process
            player_id: Player identifier ('p1' or 'p2') for message filtering
            gen: Pokemon generation (default 9)
            save_replays: Whether to save replays
            env_player: EnvPlayer reference for teampreview integration
        """
        # Initialize battle tag with format name
        battle_tag = f"battle-{format_id}-{battle_id}"
        
        # Call parent constructor with required parameters
        super().__init__(
            battle_tag=battle_tag,
            username=username,
            logger=logger,
            gen=gen,
            save_replays=save_replays
        )
        
        # IPC-specific attributes
        self._communicator = communicator
        self._battle_id = battle_id
        self._player_id = player_id  # 'p1' or 'p2' for message filtering
        self._ipc_ready = False
        self._env_player = env_player  # Store EnvPlayer reference for teampreview handling
        
        # Validate player_id
        if player_id not in ['p1', 'p2']:
            raise ValueError(f"Invalid player_id: {player_id}. Must be 'p1' or 'p2'")
        
        # Mark as ready for IPC communication
        self._ipc_ready = True
        self.logger.info(f"IPCBattle initialized: {self.battle_tag} (player: {self._player_id})")
        
        # Register this player with MapleShowdownCore for filtered messages
        if hasattr(self._communicator, 'process'):
            self.logger.debug(f"[IPC_DEBUG] scheduling player registration and IPC listen tasks for battle_id={self._battle_id}")
            from poke_env.concurrency import POKE_LOOP
            asyncio.run_coroutine_threadsafe(self._register_player(), POKE_LOOP)
            asyncio.run_coroutine_threadsafe(self._ipc_listen(), POKE_LOOP)
    
    async def _register_player(self) -> None:
        """Register this player with MapleShowdownCore for message filtering."""
        if not self._ipc_ready:
            self.logger.error("IPCBattle not ready for player registration")
            return
            
        message = {
            "type": "register_player",
            "battle_id": self._battle_id,
            "player_id": self._player_id
        }
        
        try:
            # Ensure IPC process is connected
            alive = await self._communicator.is_alive()
            if not alive:
                await self._communicator.connect()
            
            await self._communicator.send_message(message)
            self.logger.info(f"Registered player {self._player_id} for battle {self._battle_id}")
        except Exception as e:
            self.logger.error(f"Failed to register player {self._player_id}: {e}")
    
    async def send_battle_command(self, command: str) -> None:
        """Send a battle command via IPC.
        
        Args:
            command: Battle command string (e.g., "move 1", "switch 2")
        """
        if not self._ipc_ready:
            self.logger.error("IPCBattle not ready for commands")
            return
            
        message = {
            "type": "battle_command",
            "battle_id": self._battle_id,
            "player_id": self._player_id,
            "command": command
        }
        
        # Ensure IPC process is connected
        try:
            # is_alive() is async: check and connect if needed
            alive = await self._communicator.is_alive()
            if not alive:
                await self._communicator.connect()
        except Exception as e:
            self.logger.error(f"Error ensuring IPC connection before sending command: {e}")
        # Send command message
        try:
            await self._communicator.send_message(message)
            self.logger.debug(f"Sent IPC command: {command}")
        except Exception as e:
            self.logger.error(f"Failed to send IPC command {command}: {e}")
    
    def _is_json_message(self, message: str) -> bool:
        """Check if a message is a JSON control message.
        
        Args:
            message: Raw message string
            
        Returns:
            True if message is valid JSON, False otherwise
        """
        if not isinstance(message, str):
            return False
        
        try:
            json.loads(message)
            return True
        except json.JSONDecodeError:
            return False
    
    def _parse_message_safely(self, message: Any) -> tuple[bool, Dict[str, Any] | str]:
        """Safely parse a message and determine its type.
        
        Args:
            message: Message from BattleCommunicator (could be string or dict)
            
        Returns:
            Tuple of (is_json_control, parsed_message)
            - is_json_control: True if JSON control message, False if raw protocol
            - parsed_message: Parsed dict for JSON, original string for raw protocol
        """
        # If already a dict, it's a JSON control message
        if isinstance(message, dict):
            return True, message
        
        # If string, check if it's JSON or raw protocol
        if isinstance(message, str):
            if self._is_json_message(message):
                try:
                    parsed = json.loads(message)
                    return True, parsed
                except json.JSONDecodeError:
                    # Fallback to raw protocol if parsing fails
                    return False, message
            else:
                # Raw protocol string
                return False, message
        
        # Unknown type - convert to string and treat as raw protocol
        return False, str(message)

    async def _ipc_listen(self) -> None:
        """Listen for messages from IPC communicator and batch them for standard poke-env processing."""
        self.logger.info("[IPC_DEBUG_PY] Enter IPCBattle._ipc_listen // start listening for IPC messages")
        current_batch = []
        battle_tag = None
        
        while True:
            self.logger.info("[IPC_DEBUG_PY] IPCBattle._ipc_listen waiting for message")
            try:
                self.logger.info("[IPC_DEBUG_PY] Calling BattleCommunicator.receive_message")
                raw_msg = await self._communicator.receive_message()
                self.logger.info(f"[IPC_DEBUG_PY] IPCBattle._ipc_listen received raw_msg: {raw_msg} (type={type(raw_msg)})")
                
                # Safely parse message and determine type
                is_json_control, parsed_msg = self._parse_message_safely(raw_msg)
                
                if is_json_control:
                    self.logger.info(f"[IPC_DEBUG_PY] JSON control message detected: type={parsed_msg.get('type')} player_id={parsed_msg.get('player_id')}")
                else:
                    line = parsed_msg.strip() if isinstance(parsed_msg, str) else str(parsed_msg)
                    self.logger.info(f"[IPC_DEBUG_PY] Raw protocol message detected: {line}")
                
                self.logger.error(f"[IPC_DEBUG_PY] Received IPC message - is_json_control={is_json_control}, parsed_msg={parsed_msg}")
            except Exception as e:
                self.logger.error(f"IPC listen receive failed: {e}")
                await asyncio.sleep(0.1)
                continue
                
            # Handle JSON control messages
            if is_json_control and isinstance(parsed_msg, dict):
                # Safe access to dict attributes
                mtype = parsed_msg.get("type")
                msg_player_id = parsed_msg.get("player_id")
                
                # Filter messages by player_id if specified
                if msg_player_id and msg_player_id != self._player_id:
                    continue  # Skip messages not for this player
                
                if mtype == "battle_created":
                    self.logger.info(f"Battle created (IPC): {self._battle_id}")
                elif mtype == "player_registered":
                    self.logger.info(f"Player {self._player_id} registered successfully")
                elif mtype == "pong":
                    # Handle pong response safely
                    original_msg = parsed_msg.get("original_message", {})
                    success = parsed_msg.get("success", False)
                    self.logger.info(f"Received pong response: success={success}, original={original_msg}")
                elif mtype == "battle_update":
                    # Batch raw Showdown protocol lines and forward as multiline payloads
                    if msg_player_id == self._player_id and self._env_player:
                        log_lines = parsed_msg.get("log", [])
                        self.logger.debug(f"[IPC_DEBUG] battle_update for battle_id={self._battle_id}, player={self._player_id}, lines={len(log_lines)}")
                        for line in log_lines:
                            self.logger.debug(f"[IPC_DEBUG] processing battle_update line: {line}")
                            if not isinstance(line, str):
                                continue
                            # Start of new batch with battle tag
                            if line.startswith(">battle-"):
                                # Flush previous batch
                                if battle_tag is not None and current_batch:
                                    self.logger.debug(f"[IPC_DEBUG] flushing batch tag={battle_tag}, size={len(current_batch)}")
                                    payload = battle_tag + "\n" + "\n".join(current_batch)
                                    self.logger.debug(f"[IPC_DEBUG] forwarding payload[0:200]: {payload[:200].replace(chr(10), ' ')}...")
                                    asyncio.create_task(self._env_player.ps_client._handle_message(payload))
                                battle_tag = line
                                current_batch = []
                                continue
                            # Accumulate protocol lines
                            current_batch.append(line)
                            # On trigger messages, flush batch
                            parts = line.split("|")
                            if len(parts) >= 2 and parts[1] in ["request", "turn", "win", "tie", "teampreview"]:
                                self.logger.debug(f"[IPC_DEBUG] trigger '{parts[1]}' detected, flushing batch size={len(current_batch)}")
                                payload = battle_tag + "\n" + "\n".join(current_batch)
                                self.logger.debug(f"[IPC_DEBUG] forwarding triggered payload[0:200]: {payload[:200].replace(chr(10), ' ')}...")
                                asyncio.create_task(self._env_player.ps_client._handle_message(payload))
                                current_batch = []
                elif mtype == "error":
                    error_message = parsed_msg.get('error_message', 'Unknown error')
                    self.logger.error(f"IPC error: {error_message}")
                else:
                    self.logger.debug(f"[IPC_DEBUG] Ignoring control message type: {mtype}, msg={parsed_msg}")
                # ignore other control messages
                continue
                
            # Handle raw Showdown protocol lines
            if not is_json_control and isinstance(parsed_msg, str):
                line = parsed_msg.strip()
                self.logger.debug(f"[IPC_DEBUG] battle_id={self._battle_id}, player={self._player_id}, raw protocol line: {line}")
                self.logger.error(f"[IPC_DEBUG_PY] Raw protocol line: {line}")
                if not line.startswith("|") and not line.startswith(">battle-"):
                    continue
                
                # Detect battle tag line (start of new batch)
                if line.startswith(">battle-"):
                    # Process previous batch if exists
                    if current_batch:
                        await self._process_battle_batch(current_batch, battle_tag)
                    
                    # Start new batch
                    battle_tag = line
                    current_batch = []
                    self.logger.debug(f"Started new battle batch: {battle_tag}")
                    
                elif line.startswith("|"):
                    # Add to current batch
                    current_batch.append(line)
                    
                    # Process batch on completion triggers
                    split_line = line.split("|")
                    if len(split_line) >= 2 and split_line[1] in ["win", "tie", "turn", "request", "teampreview"]:
                        self.logger.debug(f"Processing batch trigger: {split_line[1]}, batch size: {len(current_batch)}")
                        await self._process_battle_batch(current_batch, battle_tag)
                        current_batch = []
                        battle_tag = None
    
    async def _process_battle_batch(self, lines: list[str], battle_tag: str | None) -> None:
        """Process batched lines using standard poke-env message handling."""
        if not lines:
            return
            
        self.logger.info(f"[IPC_DEBUG_PY] _process_battle_batch called for battle_tag={battle_tag}, lines_count={len(lines)}")
        
        # Reconstruct multi-line message like WebSocket format
        if battle_tag:
            message = battle_tag + "\n" + "\n".join(lines)
        else:
            message = "\n".join(lines)
        
        # Use standard poke-env pipeline
        await self._handle_message_like_websocket(message)
    
    async def _handle_message_like_websocket(self, message: str) -> None:
        """Handle message using WebSocket-style processing."""
        self.logger.info(f"[IPC_DEBUG_PY] _handle_message_like_websocket entry; message preview: {message[:200].replace(chr(10), ' ')}...")
        split_messages = [m.split("|") for m in message.split("\n")]
        self.logger.info(f"[IPC_DEBUG_PY] _handle_message_like_websocket split into {len(split_messages)} messages")
        
        if split_messages[0][0].startswith(">battle"):
            # Use the same logic as Player._handle_battle_message
            await self._handle_battle_message_ipc(split_messages)
    
    async def _handle_battle_message_ipc(self, split_messages: list[list[str]]) -> None:
        """Handle battle messages like poke-env Player class."""
        self.logger.info(f"[IPC_DEBUG_PY] _handle_battle_message_ipc entry; total_messages={len(split_messages)}")
        if split_messages and split_messages[0]:
            self.logger.info(f"[IPC_DEBUG_PY] battle_tag in _handle_battle_message_ipc: {split_messages[0][0]}")
        
        for split_message in split_messages[1:]:  # Skip battle tag line
            if len(split_message) <= 1:
                continue
                
            message_type = split_message[1] if len(split_message) > 1 else ""
            self.logger.debug(f"Processing message type: {message_type}")
            
            if message_type == "teampreview":
                # Now teampreview handling works the same as WebSocket!
                self.logger.info("Processing teampreview message via standard pipeline")
                super().parse_message(split_message)
                await self._handle_battle_request_ipc(from_teampreview_request=True)
                
            elif message_type == "request":
                if len(split_message) >= 3:
                    try:
                        request_data = json.loads(split_message[2])
                        self._last_request = request_data
                        
                        # Teampreview request detection - same as online mode
                        if request_data.get("teamPreview", False):
                            self.logger.info("Teampreview request detected in IPC via request message")
                            
                            # Set online mode compatible attributes
                            self._teampreview = True
                            number_of_mons = len(request_data.get("side", {}).get("pokemon", []))
                            self._max_team_size = request_data.get("maxTeamSize", number_of_mons)
                            
                            # Update teampreview opponent team data
                            self._update_teampreview_opponent_team(request_data)
                            
                            # Trigger teampreview request handling - same as online mode
                            await self._handle_battle_request_ipc(from_teampreview_request=True)
                        else:
                            self._teampreview = False
                            
                    except json.JSONDecodeError as e:
                        self.logger.error(f"Failed to parse request JSON: {e}")
                else:
                    # Parse message normally
                    super().parse_message(split_message)
                    
            else:
                # Parse all other message types normally
                super().parse_message(split_message)
    
    async def _handle_battle_request_ipc(self, from_teampreview_request: bool = False) -> None:
        """Handle battle requests like EnvPlayer._handle_battle_request."""
        self.logger.info(f"[IPC_DEBUG] _handle_battle_request_ipc called; from_teampreview_request={from_teampreview_request}")
        self.logger.debug(f"Handling battle request, teampreview: {from_teampreview_request}")
        
        if from_teampreview_request and hasattr(self, '_env_player') and self._env_player:
            # Trigger EnvPlayer teampreview handling - same as WebSocket mode
            self.logger.info("Triggering teampreview request via EnvPlayer")
            try:
                await self._env_player._handle_battle_request(
                    self, from_teampreview_request=True
                )
            except Exception as e:
                self.logger.error(f"Error in EnvPlayer teampreview handling: {e}")
        else:
            self.logger.debug("No EnvPlayer reference or not teampreview request")
    
    async def get_battle_state(self) -> Dict[str, Any]:
        """Get current battle state via IPC.
        
        Returns:
            Current battle state as dictionary
        """
        if not self._ipc_ready:
            return {}
            
        message = {
            "type": "get_battle_state",
            "battle_id": self._battle_id,
            "player_id": self._player_id
        }
        
        # Request and retrieve latest battle state
        try:
            # Ensure IPC connection
            alive = await self._communicator.is_alive()
            if not alive:
                await self._communicator.connect()
            # Delegate to communicator get_battle_state if available
            if hasattr(self._communicator, 'get_battle_state'):
                state = await self._communicator.get_battle_state(self._battle_id)
            else:
                # Fallback: manual send/receive
                await self._communicator.send_message(message)
                resp = await self._communicator.receive_message()
                state = resp.get('battle_state') or resp.get('state') or {}
            self.logger.debug(f"Received IPC battle state: {state}")
            self.logger.info(f"[IPC_DEBUG] get_battle_state for battle_id={self._battle_id} returned state keys: {list(state.keys())}")
            return state
        except Exception as e:
            self.logger.error(f"Failed to get battle state: {e}")
            return {}
    
    # parse_message and parse_request overrides removed
    # All message and request parsing is delegated to the original poke-env processing in CustomBattle.
    
    def _update_teampreview_opponent_team(self, request_data: Dict[str, Any]) -> None:
        """Update teampreview_opponent_team from request data - same as online mode.
        
        Args:
            request_data: The teampreview request data containing opponent Pokemon info
        """
        try:
            self.logger.info(f"[IPC_DEBUG] _update_teampreview_opponent_team called for battle_id={self._battle_id}")
            # Initialize teampreview_opponent_team if not exists
            if not hasattr(self, '_teampreview_opponent_team'):
                self._teampreview_opponent_team = set()
            else:
                self._teampreview_opponent_team.clear()
            
            # Extract opponent Pokemon from request (this contains visible team info)
            side_data = request_data.get("side", {})
            opponent_pokemon_data = side_data.get("pokemon", [])
            
            for poke_data in opponent_pokemon_data:
                try:
                    # Create Pokemon object from teampreview data
                    details = poke_data.get("details", "")
                    if details:
                        # Parse details: "Species, L50, M" or "Species, L50, F"
                        pokemon = Pokemon(gen=self._gen)
                        pokemon._species = details.split(",")[0].strip()
                        
                        # Set level if available
                        level_part = [part.strip() for part in details.split(",") if part.strip().startswith("L")]
                        if level_part:
                            try:
                                pokemon._level = int(level_part[0][1:])  # Remove 'L' prefix
                            except ValueError:
                                pokemon._level = 50  # Default level
                        
                        # Set gender if available
                        gender_part = [part.strip() for part in details.split(",") if part.strip() in ["M", "F"]]
                        if gender_part:
                            pokemon._gender = gender_part[0]
                        
                        self._teampreview_opponent_team.add(pokemon)
                        
                except Exception as e:
                    self.logger.error(f"Failed to create teampreview Pokemon from {poke_data}: {e}")
            
            self.logger.debug(f"Updated teampreview_opponent_team with {len(self._teampreview_opponent_team)} Pokemon")
            
        except Exception as e:
            self.logger.error(f"Failed to update teampreview_opponent_team: {e}")
    
    # Properties for environment compatibility
    @property
    def battle_id(self) -> str:
        """Get battle ID."""
        return self._battle_id
    
    @property
    def player_id(self) -> str:
        """Get player ID ('p1' or 'p2')."""
        return self._player_id
    
    @property
    def ipc_ready(self) -> bool:
        """Check if IPC communication is ready."""
        return self._ipc_ready
    
    @property
    def teampreview(self) -> bool:
        """Check if battle is in teampreview phase - same as online mode."""
        return getattr(self, '_teampreview', False)
    
    @property
    def teampreview_opponent_team(self):
        """Get opponent team visible during teampreview - same as online mode."""
        return getattr(self, '_teampreview_opponent_team', set())
    
    @property
    def max_team_size(self) -> int:
        """Get maximum allowed team size - same as online mode."""
        return getattr(self, '_max_team_size', 6)
    
    def clear_all_boosts(self) -> None:
        """Clear all stat boosts on active Pokemon (required by poke-env)."""
        # IPC implementation - send clear boosts command
        if self._active_pokemon:
            # This would typically be handled by Pokemon Showdown automatically
            pass
    
    def get_pokemon(self, 
                   identifier: str,
                   force_self_team: bool = False,
                   details: str = "",
                   request: Optional[Dict[str, Any]] = None) -> Pokemon:
        """Get or create Pokemon by identifier.
        
        Args:
            identifier: Pokemon identifier
            force_self_team: Force Pokemon to be on player's team
            details: Pokemon details string
            request: Current battle request
            
        Returns:
            Pokemon instance
        """
        # Try to get existing Pokemon from teams
        if not force_self_team and identifier in self._opponent_team:
            return self._opponent_team[identifier]
        elif identifier in self._team:
            return self._team[identifier]
        
        # Create new Pokemon if not found
        # This is a simplified implementation - full implementation would
        # parse the details string and create proper Pokemon objects
        try:
            pokemon = Pokemon(gen=self._gen)
            
            # Add to appropriate team
            if force_self_team or identifier.startswith("p1"):
                self._team[identifier] = pokemon
            else:
                self._opponent_team[identifier] = pokemon
                
            return pokemon
            
        except Exception as e:
            self.logger.error(f"Failed to create Pokemon {identifier}: {e}")
            # Return a placeholder Pokemon to prevent crashes
            return Pokemon(gen=self._gen)