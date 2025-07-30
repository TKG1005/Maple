"""IPC-based battle implementation for direct Pokemon Showdown integration."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional, Set, Union

from poke_env.environment import Pokemon, Move
from src.env.custom_battle import CustomBattle
from src.sim.battle_communicator import BattleCommunicator


class IPCBattle(CustomBattle):
    """IPC-based battle that communicates directly with Node.js Pokemon Showdown process."""
    
    def __init__(self, 
                 battle_id: str,
                 username: str, 
                 logger: logging.Logger,
                 communicator: BattleCommunicator,
                 gen: int = 9,
                 save_replays: Union[str, bool] = False) -> None:
        """Initialize IPC battle with minimal required state.
        
        Args:
            battle_id: Unique battle identifier
            username: Player username
            logger: Logger instance
            communicator: IPC communicator for Node.js process
            gen: Pokemon generation (default 9)
            save_replays: Whether to save replays
        """
        # Initialize with minimal battle tag format
        battle_tag = f"battle-gen{gen}randombattle-{battle_id}"
        
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
        self._ipc_ready = False
        
        # Initialize battle state
        self._initialize_battle_state()
    
    def _initialize_battle_state(self) -> None:
        """Initialize basic battle state for environment compatibility."""
        # Critical state for environment synchronization
        self._last_request = None
        self._turn = 0
        self._finished = False
        self._trapped = False
        
        # Pokemon teams (will be populated by IPC messages)
        self._team: Dict[str, Pokemon] = {}
        self._opponent_team: Dict[str, Pokemon] = {}
        self._active_pokemon: Optional[Pokemon] = None
        self._opponent_active_pokemon: Optional[Pokemon] = None
        self._teampreview_opponent_team: Set[Pokemon] = set()
        
        # Action constraints
        self._available_moves: List[Move] = []
        self._available_switches: List[Pokemon] = []
        
        # Create minimal Pokemon teams for initial compatibility
        self._create_minimal_teams()
        
        # Mark as ready for IPC communication
        self._ipc_ready = True
        self.logger.info(f"IPCBattle initialized: {self.battle_tag}")
    
    def _create_minimal_teams(self) -> None:
        """Create minimal Pokemon teams for initial state observer compatibility."""
        try:
            from poke_env.environment import Pokemon, Move
            
            # Create a minimal Pokemon for each player to prevent None errors
            # This is a temporary solution - in a full implementation, teams would
            # be populated from actual battle data
            
            # Create teams with multiple Pokemon (6 per team for full team)
            team_configs = [
                ("_team", ["p1a", "p1b", "p1c", "p1d", "p1e", "p1f"]),
                ("_opponent_team", ["p2a", "p2b", "p2c", "p2d", "p2e", "p2f"])
            ]
            
            for team_key, pokemon_keys in team_configs:
                team_dict = getattr(self, team_key)
                first_pokemon = None
                
                for i, pokemon_key in enumerate(pokemon_keys):
                    try:
                        # Create Pokemon using proper poke-env constructor with species parameter
                        # This will automatically load species data, base stats, types, etc. from pokedex
                        pokemon_species = "ditto"  # Use lowercase for poke-env species lookup
                        pokemon = Pokemon(gen=self._gen, species=pokemon_species)
                        
                        # Set level and HP
                        pokemon._level = 50
                        pokemon._max_hp = 100
                        pokemon._current_hp = 100
                        
                        # Calculate and set actual stats based on base stats
                        # Use simplified stat calculation for level 50 with neutral nature
                        if hasattr(pokemon, 'base_stats') and pokemon.base_stats:
                            # Calculate actual stats using Pokemon's base stats
                            # Simplified formula: ((base_stat * 2 + 31 + 252/4) * level / 100) + 5
                            # For HP: ((base_stat * 2 + 31 + 252/4) * level / 100) + level + 10
                            level = 50
                            pokemon._stats = {
                                'hp': int(((pokemon.base_stats['hp'] * 2 + 31 + 252/4) * level / 100) + level + 10),
                                'atk': int(((pokemon.base_stats['atk'] * 2 + 31 + 252/4) * level / 100) + 5),
                                'def': int(((pokemon.base_stats['def'] * 2 + 31 + 252/4) * level / 100) + 5),
                                'spa': int(((pokemon.base_stats['spa'] * 2 + 31 + 252/4) * level / 100) + 5),
                                'spd': int(((pokemon.base_stats['spd'] * 2 + 31 + 252/4) * level / 100) + 5),
                                'spe': int(((pokemon.base_stats['spe'] * 2 + 31 + 252/4) * level / 100) + 5)
                            }
                            # Update max_hp to match calculated HP stat
                            pokemon._max_hp = pokemon._stats['hp']
                            pokemon._current_hp = pokemon._stats['hp']
                        
                        # Set as active Pokemon only for the first Pokemon in each team
                        pokemon._active = (i == 0)  # Only first Pokemon is active
                        
                        # Create moves using poke-env's proper data loading
                        from poke_env.environment.move import Move
                        
                        basic_moves = {}
                        # Use actual move names that poke-env can recognize and load data for 
                        move_names = ["tackle", "rest", "protect", "struggle"]
                        
                        for move_name in move_names:
                            # Let poke-env load the move data automatically
                            move = Move(move_name, gen=self._gen)
                            basic_moves[move_name] = move
                            
                        pokemon._moves = basic_moves
                        
                        # Add to appropriate team
                        team_dict[pokemon_key] = pokemon
                        
                        # Keep reference to first Pokemon for setting as active
                        if i == 0:
                            first_pokemon = pokemon
                            
                    except Exception as e:
                        error_context = {
                            'pokemon_key': pokemon_key,
                            'team_key': team_key,
                            'pokemon_species': pokemon_species,
                            'gen': self._gen,
                            'team_index': i,
                            'exception_type': type(e).__name__,
                            'exception_details': str(e)
                        }
                        raise RuntimeError(f"POKEMON_CREATION_ERROR: Failed to create Pokemon {pokemon_key} for {team_key}. "
                                         f"Context: {error_context}. "
                                         f"Root cause: poke-env Pokemon initialization failure - check species name, generation compatibility, or data loading issues.") from e
                
                # Set active Pokemon references
                if first_pokemon and team_key == "_team":
                    self._active_pokemon = first_pokemon
                elif first_pokemon and team_key == "_opponent_team":
                    self._opponent_active_pokemon = first_pokemon
                    
        except Exception as e:
            self.logger.error(f"Failed to create minimal teams: {e}")
            # Continue without minimal teams - might cause errors but won't crash initialization
    
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
            "command": command
        }
        
        try:
            await self._communicator.send_message(message)
            self.logger.debug(f"Sent IPC command: {command}")
        except Exception as e:
            self.logger.error(f"Failed to send IPC command {command}: {e}")
    
    async def get_battle_state(self) -> Dict[str, Any]:
        """Get current battle state via IPC.
        
        Returns:
            Current battle state as dictionary
        """
        if not self._ipc_ready:
            return {}
            
        message = {
            "type": "get_battle_state",
            "battle_id": self._battle_id
        }
        
        try:
            await self._communicator.send_message(message)
            # TODO: Implement response handling
            response = await self._communicator.receive_message()
            return response.get("battle_state", {})
        except Exception as e:
            self.logger.error(f"Failed to get battle state: {e}")
            return {}
    
    def parse_message(self, split_message: List[str]) -> None:
        """Parse battle message from IPC stream.
        
        Args:
            split_message: Split battle message from Pokemon Showdown
        """
        try:
            # Let parent handle standard message parsing
            super().parse_message(split_message)
            
            # IPC-specific message handling
            if len(split_message) >= 2:
                message_type = split_message[1]
                
                # Update turn counter
                if message_type == "turn":
                    try:
                        self._turn = int(split_message[2])
                        self.logger.debug(f"Battle turn: {self._turn}")
                    except (IndexError, ValueError):
                        pass
                
                # Check for battle end
                elif message_type in ["win", "tie"]:
                    self._finished = True
                    self.logger.info(f"Battle finished: {message_type}")
                
                # Handle request messages (critical for environment sync)
                elif message_type == "request":
                    if len(split_message) >= 3:
                        try:
                            request_data = json.loads(split_message[2])
                            self._last_request = request_data
                            self.logger.debug("Updated last_request from IPC")
                        except json.JSONDecodeError as e:
                            self.logger.error(f"Failed to parse request JSON: {e}")
            
        except Exception as e:
            self.logger.error(f"Error parsing IPC message: {e}")
            self.logger.error(f"Message: {split_message}")
    
    def parse_request(self, request: Dict[str, Any]) -> None:
        """Parse request data from Pokemon Showdown.
        
        Args:
            request: Request dictionary from server
        """
        try:
            self._last_request = request
            
            # Extract available actions
            if "moves" in request:
                # TODO: Convert to Move objects
                pass
            
            if "active" in request:
                # Update active Pokemon state
                # TODO: Update trapped status, available moves
                pass
                
            self.logger.debug("Parsed IPC request")
            
        except Exception as e:
            self.logger.error(f"Error parsing IPC request: {e}")
    
    # Properties for environment compatibility
    @property
    def battle_id(self) -> str:
        """Get battle ID."""
        return self._battle_id
    
    @property
    def ipc_ready(self) -> bool:
        """Check if IPC communication is ready."""
        return self._ipc_ready
    
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