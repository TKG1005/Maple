"""Factory for creating IPC-based battles."""

from __future__ import annotations

import asyncio
import logging
import uuid
import random
from typing import Dict, List, Optional, Any

from src.sim.battle_communicator import BattleCommunicator
from src.sim.ipc_battle import IPCBattle
from src.teams.team_cache import TeamCacheManager


class IPCBattleFactory:
    """Factory for creating and managing IPC-based battles."""
    
    def __init__(self, communicator: BattleCommunicator, logger: logging.Logger, teams_dir: Optional[str] = None):
        """Initialize the IPC battle factory.
        
        Args:
            communicator: IPC communicator for Node.js process
            logger: Logger instance
            teams_dir: Directory containing team files (optional)
        """
        self._communicator = communicator
        self._logger = logger
        self._active_battles: Dict[str, IPCBattle] = {}
        self._battle_counter = 0
        self._teams_dir = teams_dir
        self._cached_teams = None
        self._default_team_used_count = 0
    
    async def create_battle(self, 
                          format_id: str = "gen9randombattle",
                          player_names: List[str] = None,
                          teams: Optional[List[str]] = None) -> IPCBattle:
        """Create a new IPC battle.
        
        Args:
            format_id: Battle format (e.g., "gen9randombattle")
            player_names: List of player names [player1, player2]
            teams: Optional team data for players
            
        Returns:
            IPCBattle instance
        """
        if player_names is None:
            player_names = ["Player1", "Player2"]
        
        # Generate unique battle ID
        self._battle_counter += 1
        battle_id = f"{self._battle_counter}-{uuid.uuid4().hex[:8]}"
        
        # Ensure teams are provided (B1 strategy)
        final_teams = await self._ensure_teams_provided(teams, format_id)
        
        # Create battle creation message
        create_message = {
            "type": "create_battle",
            "battle_id": battle_id,
            "format": format_id,
            "players": [
                {"name": player_names[0], "team": final_teams[0]},
                {"name": player_names[1], "team": final_teams[1]}
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
            # Send battle creation request to Node.js
            await self._communicator.send_message(create_message)
            self._logger.info(f"Creating IPC battle: {battle_id}")
            
            # Wait for battle creation confirmation
            response = await self._wait_for_battle_creation(battle_id)
            
            if response.get("success"):
                # Create IPCBattle instance
                battle = IPCBattle(
                    battle_id=battle_id,
                    username=player_names[0],  # Use first player as main player
                    logger=self._logger,
                    communicator=self._communicator,
                    format_id=format_id
                )
                
                # Store battle reference
                self._active_battles[battle_id] = battle
                
                self._logger.info(f"Successfully created IPC battle: {battle_id}")
                return battle
            else:
                error_msg = response.get("error", "Unknown error")
                raise RuntimeError(f"Failed to create battle: {error_msg}")
                
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
    
    async def get_battle(self, battle_id: str) -> Optional[IPCBattle]:
        """Get an existing battle by ID.
        
        Args:
            battle_id: Battle ID
            
        Returns:
            IPCBattle instance or None if not found
        """
        return self._active_battles.get(battle_id)
    
    async def destroy_battle(self, battle_id: str) -> bool:
        """Destroy a battle and clean up resources.
        
        Args:
            battle_id: Battle ID to destroy
            
        Returns:
            True if successful, False if battle not found
        """
        if battle_id not in self._active_battles:
            self._logger.warning(f"Battle {battle_id} not found for destruction")
            return False
        
        try:
            # Send destroy message to Node.js
            destroy_message = {
                "type": "destroy_battle",
                "battle_id": battle_id
            }
            
            await self._communicator.send_message(destroy_message)
            
            # Remove from active battles
            del self._active_battles[battle_id]
            
            self._logger.info(f"Destroyed IPC battle: {battle_id}")
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
    
    async def _ensure_teams_provided(self, teams: Optional[List[str]], format_id: str) -> List[str]:
        """Ensure both players have teams, providing defaults if necessary.
        
        Args:
            teams: Provided teams or None
            format_id: Battle format
            
        Returns:
            List of two team strings
        """
        # Initialize result with provided teams
        result_teams = [None, None]
        if teams:
            if len(teams) > 0 and teams[0]:
                result_teams[0] = teams[0]
            if len(teams) > 1 and teams[1]:
                result_teams[1] = teams[1]
        
        # Load cached teams if not already loaded
        if self._cached_teams is None and self._teams_dir:
            try:
                self._cached_teams, _ = TeamCacheManager.get_teams(self._teams_dir)
                self._logger.info(f"Loaded {len(self._cached_teams)} teams from cache")
            except Exception as e:
                self._logger.error(f"Failed to load teams from {self._teams_dir}: {e}")
                self._cached_teams = []
        
        # Provide default teams for any missing teams
        for i in range(2):
            if result_teams[i] is None:
                if self._cached_teams and len(self._cached_teams) > 0:
                    # Select random team from cache
                    result_teams[i] = random.choice(self._cached_teams)
                    self._logger.info(f"Selected random team from cache for player {i+1}")
                else:
                    # Use hardcoded default team as last resort
                    result_teams[i] = self._get_default_team()
                    self._default_team_used_count += 1
                    self._logger.warning(
                        f"No cached teams available, using default team for player {i+1}. "
                        f"Default team has been used {self._default_team_used_count} times."
                    )
        
        return result_teams
    
    def _get_default_team(self) -> str:
        """Get a hardcoded default team for BSS Reg I format.
        
        Returns:
            Default team string
        """
        # Basic BSS Reg I team that avoids custom ban issues
        return """Miraidon @ Choice Specs
Ability: Hadron Engine
Level: 50
Tera Type: Electric
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Electro Drift
- Draco Meteor
- Volt Switch
- Overheat

Calyrex-Shadow @ Focus Sash
Ability: As One (Spectrier)
Level: 50
Tera Type: Fighting
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Astral Barrage
- Psyshock
- Draining Kiss
- Nasty Plot

Flutter Mane @ Booster Energy
Ability: Protosynthesis
Level: 50
Tera Type: Fairy
EVs: 4 HP / 252 SpA / 252 Spe
Modest Nature
- Moonblast
- Shadow Ball
- Mystical Fire
- Protect

Urshifu @ Choice Band
Ability: Unseen Fist
Level: 50
Tera Type: Water
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Surging Strikes
- Close Combat
- Aqua Jet
- U-turn

Ogerpon-Wellspring @ Wellspring Mask
Ability: Water Absorb
Level: 50
Tera Type: Water
EVs: 4 HP / 252 Atk / 252 Spe
Jolly Nature
- Ivy Cudgel
- Horn Leech
- Superpower
- Follow Me

Incineroar @ Sitrus Berry
Ability: Intimidate
Level: 50
Tera Type: Ghost
EVs: 252 HP / 252 Atk / 4 Def
Adamant Nature
- Flare Blitz
- Knock Off
- Fake Out
- Parting Shot"""