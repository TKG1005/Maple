"""Factory for creating IPC-based battles."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Dict, List, Optional, Any

from src.sim.battle_communicator import BattleCommunicator
from src.sim.ipc_battle import IPCBattle


class IPCBattleFactory:
    """Factory for creating and managing IPC-based battles."""
    
    def __init__(self, communicator: BattleCommunicator, logger: logging.Logger):
        """Initialize the IPC battle factory.
        
        Args:
            communicator: IPC communicator for Node.js process
            logger: Logger instance
        """
        self._communicator = communicator
        self._logger = logger
        self._active_battles: Dict[str, IPCBattle] = {}
        self._battle_counter = 0
    
    async def create_battle_for_player(self, 
                          player_id: str,
                          format_id: str = "gen9randombattle",
                          player_names: List[str] = None,
                          teams: Optional[List[str]] = None,
                          env_player: Any = None) -> IPCBattle:
        """Create a new IPC battle for a specific player.
        
        Args:
            player_id: Player identifier ('p1' or 'p2')
            format_id: Battle format (e.g., "gen9randombattle")
            player_names: List of player names [player1, player2]
            teams: Optional team data for players
            env_player: EnvPlayer reference for teampreview integration
            
        Returns:
            IPCBattle instance for the specified player
        """
        if player_names is None:
            player_names = ["Player1", "Player2"]
        
        # Validate player_id
        if player_id not in ['p1', 'p2']:
            raise ValueError(f"Invalid player_id: {player_id}. Must be 'p1' or 'p2'")
        
        # Generate unique battle ID (shared between both players)
        self._battle_counter += 1
        battle_id = f"{self._battle_counter}-{uuid.uuid4().hex[:8]}"
        
        # Create battle creation message (only create once per battle_id)
        battle_key = f"{battle_id}_{player_id}"
        if battle_key not in self._active_battles:
            create_message = {
                "type": "create_battle",
                "battle_id": battle_id,
                "format": format_id,
                "players": [
                    {"name": player_names[0], "team": teams[0] if teams and len(teams) > 0 else None, "player_id": "p1"},
                    {"name": player_names[1], "team": teams[1] if teams and len(teams) > 1 else None, "player_id": "p2"}
                ]
            }
        # Log battle request payload for debugging
        try:
            import json as _json
            _payload = _json.dumps(create_message)
        except Exception:
            _payload = str(create_message)
        self._logger.info(f"IPC create_battle payload: {_payload}")
        
        try:
            # Send battle creation request to Node.js (only once per battle)
            if battle_key not in self._active_battles:
                await self._communicator.send_message(create_message)
                self._logger.info(f"Creating IPC battle: {battle_id}")
                
                # Wait for battle creation confirmation
                response = await self._wait_for_battle_creation(battle_id)
                
                if not response.get("success"):
                    error_msg = response.get("error", "Unknown error")
                    raise RuntimeError(f"Failed to create battle: {error_msg}")
            
            # Create player-specific IPCBattle instance
            player_index = 0 if player_id == 'p1' else 1
            username = player_names[player_index] if player_names else f"Player{player_index + 1}"
            
            battle = IPCBattle(
                battle_id=battle_id,
                username=username,
                logger=self._logger,
                communicator=self._communicator,
                player_id=player_id,  # Pass player_id for message filtering
                format_id=format_id,
                env_player=env_player  # Pass EnvPlayer reference
            )
            
            # Store battle reference with player-specific key
            self._active_battles[battle_key] = battle
            
            self._logger.info(f"Successfully created IPC battle for {player_id}: {battle_id}")
            return battle
                
        except Exception as e:
            self._logger.error(f"Error creating IPC battle: {e}")
            raise
    
    async def _wait_for_battle_creation(self, battle_id: str, timeout: float = 10.0) -> Dict[str, Any]:
        """Wait for battle creation confirmation from Node.js.
        
        Args:
            battle_id: Battle ID to wait for
            timeout: Timeout in seconds
            
        Returns:
            Response from Node.js server
        """
        try:
            # Wait for response with timeout
            start_time = asyncio.get_event_loop().time()
            
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                try:
                    response = await asyncio.wait_for(
                        self._communicator.receive_message(), 
                        timeout=1.0
                    )
                    
                    # Check if this is our battle creation response
                    if (response.get("type") == "battle_created" and 
                        response.get("battle_id") == battle_id):
                        return response
                        
                    # Log other messages for debugging
                    self._logger.debug(f"Received other message while waiting: {response}")
                    
                except asyncio.TimeoutError:
                    # Continue waiting
                    continue
            
            # Timeout reached
            return {"success": False, "error": "Timeout waiting for battle creation"}
            
        except Exception as e:
            self._logger.error(f"Error waiting for battle creation: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_battle(self, battle_id: str, player_id: str = None) -> Optional[IPCBattle]:
        """Get an existing battle by ID and player.
        
        Args:
            battle_id: Battle ID
            player_id: Player identifier ('p1' or 'p2')
            
        Returns:
            IPCBattle instance or None if not found
        """
        if player_id:
            battle_key = f"{battle_id}_{player_id}"
            return self._active_battles.get(battle_key)
        else:
            # Legacy support - return first found battle with this ID
            for key, battle in self._active_battles.items():
                if key.startswith(f"{battle_id}_"):
                    return battle
            return None
    
    async def create_battle(self, 
                          format_id: str = "gen9randombattle",
                          player_names: List[str] = None,
                          teams: Optional[List[str]] = None,
                          env_player: Any = None) -> IPCBattle:
        """Legacy method: Create a new IPC battle for player 1.
        
        This method creates a battle for 'p1' for backward compatibility.
        For new code, use create_battle_for_player() instead.
        
        Args:
            format_id: Battle format (e.g., "gen9randombattle")
            player_names: List of player names [player1, player2]
            teams: Optional team data for players
            env_player: EnvPlayer reference for teampreview integration
            
        Returns:
            IPCBattle instance for player 1
        """
        return await self.create_battle_for_player(
            player_id="p1",
            format_id=format_id,
            player_names=player_names,
            teams=teams,
            env_player=env_player
        )
    
    async def destroy_battle(self, battle_id: str, player_id: str = None) -> bool:
        """Destroy a battle and clean up resources.
        
        Args:
            battle_id: Battle ID to destroy
            player_id: Optional player identifier for player-specific cleanup
            
        Returns:
            True if successful, False if battle not found
        """
        # Find battles to destroy
        battles_to_destroy = []
        if player_id:
            battle_key = f"{battle_id}_{player_id}"
            if battle_key in self._active_battles:
                battles_to_destroy.append(battle_key)
        else:
            # Destroy all player instances of this battle
            battles_to_destroy = [key for key in self._active_battles.keys() 
                                if key.startswith(f"{battle_id}_")]
        
        if not battles_to_destroy:
            self._logger.warning(f"Battle {battle_id} not found for destruction")
            return False
        
        try:
            # Send destroy message to Node.js (only once per battle_id)
            if not player_id or player_id == "p1":  # Send destroy only once
                destroy_message = {
                    "type": "destroy_battle",
                    "battle_id": battle_id
                }
                await self._communicator.send_message(destroy_message)
            
            # Remove from active battles
            for battle_key in battles_to_destroy:
                del self._active_battles[battle_key]
            
            self._logger.info(f"Destroyed IPC battle: {battle_id} (removed {len(battles_to_destroy)} instances)")
            return True
            
        except Exception as e:
            self._logger.error(f"Error destroying battle {battle_id}: {e}")
            return False
    
    async def list_active_battles(self) -> List[str]:
        """Get list of active battle IDs.
        
        Returns:
            List of active battle IDs
        """
        return list(self._active_battles.keys())
    
    async def cleanup_all_battles(self) -> None:
        """Clean up all active battles."""
        battle_ids = list(self._active_battles.keys())
        
        for battle_id in battle_ids:
            try:
                await self.destroy_battle(battle_id)
            except Exception as e:
                self._logger.error(f"Error cleaning up battle {battle_id}: {e}")
        
        self._logger.info("Cleaned up all IPC battles")
    
    def get_battle_count(self) -> int:
        """Get number of active battles.
        
        Returns:
            Number of active battles
        """
        return len(self._active_battles)