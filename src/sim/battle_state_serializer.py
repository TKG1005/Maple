"""Battle state serialization system for Pokemon battles.

This module provides comprehensive serialization and deserialization of Pokemon
battle states to/from JSON format, enabling save/restore functionality for both
local and online battle modes.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class PokemonState:
    """Serializable representation of a Pokemon's battle state."""
    species: str
    nickname: Optional[str]
    level: int
    gender: Optional[str]
    hp: int
    max_hp: int
    status: Optional[str]
    stats: Dict[str, int]  # Current battle stats
    base_stats: Dict[str, int]  # Base species stats
    moves: List[Dict[str, Any]]  # Move data with PP
    ability: Optional[str]
    item: Optional[str]
    types: List[str]
    boosts: Dict[str, int]  # Stat boosts/drops
    volatile_status: List[str]  # Temporary status effects
    position: Optional[int]  # Position in team (0-5)
    active: bool  # Whether currently active in battle


@dataclass
class PlayerState:
    """Serializable representation of a player's battle state."""
    player_id: str
    username: str
    team: List[PokemonState]
    active_pokemon: Optional[int]  # Index of active Pokemon
    side_conditions: Dict[str, Any]  # Field effects on this side
    last_move: Optional[str]
    can_switch: List[bool]  # Which Pokemon can be switched to
    can_dynamax: bool
    dynamax_turns_left: int


@dataclass 
class BattleState:
    """Complete serializable battle state."""
    battle_id: str
    format_id: str
    turn: int
    weather: Optional[str]
    weather_turns_left: int
    terrain: Optional[str]
    terrain_turns_left: int
    field_effects: Dict[str, Any]  # Global field effects
    players: List[PlayerState]
    battle_log: List[str]  # Recent battle messages
    timestamp: str
    metadata: Dict[str, Any]  # Additional battle metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BattleState':
        """Create BattleState from dictionary."""
        # Convert player data
        players = []
        for player_data in data['players']:
            # Convert Pokemon states
            team = []
            for pokemon_data in player_data['team']:
                pokemon_state = PokemonState(**pokemon_data)
                team.append(pokemon_state)
            
            player_state = PlayerState(
                player_id=player_data['player_id'],
                username=player_data['username'],
                team=team,
                active_pokemon=player_data['active_pokemon'],
                side_conditions=player_data['side_conditions'],
                last_move=player_data['last_move'],
                can_switch=player_data['can_switch'],
                can_dynamax=player_data['can_dynamax'],
                dynamax_turns_left=player_data['dynamax_turns_left']
            )
            players.append(player_state)
        
        return cls(
            battle_id=data['battle_id'],
            format_id=data['format_id'],
            turn=data['turn'],
            weather=data['weather'],
            weather_turns_left=data['weather_turns_left'],
            terrain=data['terrain'],
            terrain_turns_left=data['terrain_turns_left'],
            field_effects=data['field_effects'],
            players=players,
            battle_log=data['battle_log'],
            timestamp=data['timestamp'],
            metadata=data['metadata']
        )


class BattleStateSerializer(ABC):
    """Abstract interface for battle state serialization."""
    
    @abstractmethod
    def serialize_state(self, battle: Any) -> BattleState:
        """Extract and serialize battle state from a battle object.
        
        Args:
            battle: Pokemon battle object (varies by implementation)
            
        Returns:
            Serialized battle state
        """
        pass
    
    @abstractmethod
    def deserialize_state(self, state: BattleState) -> Any:
        """Reconstruct battle object from serialized state.
        
        Args:
            state: Serialized battle state
            
        Returns:
            Reconstructed battle object
        """
        pass
    
    @abstractmethod
    def validate_state(self, state: BattleState) -> bool:
        """Validate that a battle state is consistent and complete.
        
        Args:
            state: Battle state to validate
            
        Returns:
            True if state is valid, False otherwise
        """
        pass


class PokeEnvBattleSerializer(BattleStateSerializer):
    """Battle state serializer for poke-env Battle objects."""
    
    def __init__(self):
        self._logger = logging.getLogger(__name__)
    
    def serialize_state(self, battle: Any) -> BattleState:
        """Extract state from poke-env Battle object."""
        try:
            # Extract player states
            players = []
            
            for player_role in ['p1', 'p2']:
                if not hasattr(battle, player_role):
                    continue
                
                player = getattr(battle, player_role)
                
                # Extract team information
                team = []
                for i, pokemon in enumerate(player.team.values()):
                    pokemon_state = self._serialize_pokemon(pokemon, i)
                    team.append(pokemon_state)
                
                # Extract active Pokemon index
                active_pokemon = None
                if player.active_pokemon:
                    for i, team_pokemon in enumerate(player.team.values()):
                        if team_pokemon == player.active_pokemon:
                            active_pokemon = i
                            break
                
                player_state = PlayerState(
                    player_id=player_role,
                    username=getattr(player, 'username', f'Player{player_role[-1]}'),
                    team=team,
                    active_pokemon=active_pokemon,
                    side_conditions=dict(getattr(player, 'side_conditions', {})),
                    last_move=getattr(player, 'last_move', None),
                    can_switch=self._get_switch_availability(player),
                    can_dynamax=getattr(player, 'can_dynamax', True),
                    dynamax_turns_left=getattr(player, 'dynamax_turns_left', 0)
                )
                players.append(player_state)
            
            # Extract battle-wide state
            battle_state = BattleState(
                battle_id=battle.battle_tag,
                format_id=getattr(battle, 'format_id', 'gen9randombattle'),
                turn=battle.turn,
                weather=battle.weather.name if battle.weather else None,
                weather_turns_left=getattr(battle, 'weather_turns_left', 0),
                terrain=battle.terrain.name if battle.terrain else None,
                terrain_turns_left=getattr(battle, 'terrain_turns_left', 0),
                field_effects=dict(getattr(battle, 'fields', {})),
                players=players,
                battle_log=list(getattr(battle, 'history', [])[-50:]),  # Last 50 messages
                timestamp=datetime.now().isoformat(),
                metadata={
                    'rng_seed': getattr(battle, 'seed', None),
                    'battle_started': getattr(battle, 'started', False),
                    'battle_finished': getattr(battle, 'finished', False),
                    'winner': getattr(battle, 'winner', None)
                }
            )
            
            return battle_state
            
        except Exception as e:
            self._logger.error(f"Failed to serialize battle state: {e}")
            raise
    
    def _serialize_pokemon(self, pokemon: Any, position: int) -> PokemonState:
        """Extract Pokemon state from poke-env Pokemon object."""
        try:
            # Extract moves with PP information
            moves = []
            for move in getattr(pokemon, 'moves', []):
                move_data = {
                    'id': move.id,
                    'name': move.name if hasattr(move, 'name') else move.id,
                    'type': move.type.name if hasattr(move, 'type') else None,
                    'category': move.category.name if hasattr(move, 'category') else None,
                    'power': getattr(move, 'base_power', 0),
                    'accuracy': getattr(move, 'accuracy', 100),
                    'pp': getattr(move, 'current_pp', getattr(move, 'pp', 0)),
                    'max_pp': getattr(move, 'pp', 0)
                }
                moves.append(move_data)
            
            return PokemonState(
                species=pokemon.species,
                nickname=getattr(pokemon, 'nickname', pokemon.species),
                level=getattr(pokemon, 'level', 50),
                gender=pokemon.gender.name if hasattr(pokemon, 'gender') and pokemon.gender else None,
                hp=pokemon.current_hp_fraction * pokemon.max_hp if hasattr(pokemon, 'current_hp_fraction') else 100,
                max_hp=getattr(pokemon, 'max_hp', 100),
                status=pokemon.status.name if pokemon.status else None,
                stats=dict(getattr(pokemon, 'stats', {})),
                base_stats=dict(getattr(pokemon, 'base_stats', {})),
                moves=moves,
                ability=pokemon.ability.name if hasattr(pokemon, 'ability') and pokemon.ability else None,
                item=pokemon.item.name if hasattr(pokemon, 'item') and pokemon.item else None,
                types=[t.name for t in pokemon.types] if hasattr(pokemon, 'types') else [],
                boosts=dict(getattr(pokemon, 'boosts', {})),
                volatile_status=list(getattr(pokemon, 'effects', {}).keys()),
                position=position,
                active=getattr(pokemon, 'active', False)
            )
            
        except Exception as e:
            self._logger.error(f"Failed to serialize Pokemon {pokemon}: {e}")
            # Return minimal valid state
            return PokemonState(
                species=getattr(pokemon, 'species', 'unknown'),
                nickname=None,
                level=50,
                gender=None,
                hp=100,
                max_hp=100,
                status=None,
                stats={},
                base_stats={},
                moves=[],
                ability=None,
                item=None,
                types=[],
                boosts={},
                volatile_status=[],
                position=position,
                active=False
            )
    
    def _get_switch_availability(self, player: Any) -> List[bool]:
        """Determine which Pokemon can be switched to."""
        try:
            can_switch = []
            for pokemon in player.team.values():
                # Pokemon can be switched if it's not active, not fainted, and no trapping effects
                can_switch.append(
                    not getattr(pokemon, 'active', False) and
                    getattr(pokemon, 'current_hp_fraction', 1.0) > 0 and
                    not getattr(player, 'trapped', False)
                )
            return can_switch
        except Exception:
            # Default to all Pokemon available
            return [True] * 6
    
    def deserialize_state(self, state: BattleState) -> Any:
        """Reconstruct battle from serialized state.
        
        Note: Full battle reconstruction is complex and may require Pokemon Showdown
        integration. This implementation provides a framework for future development.
        """
        self._logger.warning("Battle deserialization not fully implemented - requires Pokemon Showdown integration")
        
        # Return state data for now - full reconstruction would need Pokemon Showdown
        return {
            'battle_id': state.battle_id,
            'format_id': state.format_id,
            'turn': state.turn,
            'players': [p.player_id for p in state.players],
            'serialized_data': state.to_dict()
        }
    
    def validate_state(self, state: BattleState) -> bool:
        """Validate battle state consistency."""
        try:
            # Basic validation checks
            if not state.battle_id or not state.format_id:
                return False
            
            if len(state.players) != 2:
                return False
            
            for player in state.players:
                if not player.team or len(player.team) == 0:
                    return False
                
                # Check active Pokemon index is valid
                if player.active_pokemon is not None:
                    if player.active_pokemon < 0 or player.active_pokemon >= len(player.team):
                        return False
            
            return True
            
        except Exception as e:
            self._logger.error(f"State validation failed: {e}")
            return False


class BattleStateManager:
    """Manager for battle state persistence operations."""
    
    def __init__(self, serializer: BattleStateSerializer, storage_dir: str = "battle_states"):
        self.serializer = serializer
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._logger = logging.getLogger(__name__)
    
    def save_state(self, battle: Any, filename: Optional[str] = None) -> str:
        """Save battle state to JSON file.
        
        Args:
            battle: Battle object to serialize
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Path to saved file
        """
        try:
            state = self.serializer.serialize_state(battle)
            
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{state.battle_id}_{timestamp}.json"
            
            filepath = self.storage_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
            
            self._logger.info(f"Saved battle state to {filepath}")
            return str(filepath)
            
        except Exception as e:
            self._logger.error(f"Failed to save battle state: {e}")
            raise
    
    def load_state(self, filepath: str) -> BattleState:
        """Load battle state from JSON file.
        
        Args:
            filepath: Path to saved state file
            
        Returns:
            Loaded battle state
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            state = BattleState.from_dict(data)
            
            if not self.serializer.validate_state(state):
                raise ValueError(f"Invalid battle state in {filepath}")
            
            self._logger.info(f"Loaded battle state from {filepath}")
            return state
            
        except Exception as e:
            self._logger.error(f"Failed to load battle state from {filepath}: {e}")
            raise
    
    def list_saved_states(self, battle_id: Optional[str] = None) -> List[str]:
        """List available saved battle states.
        
        Args:
            battle_id: Optional filter by battle ID
            
        Returns:
            List of available state files
        """
        try:
            pattern = f"{battle_id}_*.json" if battle_id else "*.json"
            files = list(self.storage_dir.glob(pattern))
            return [str(f.name) for f in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)]
            
        except Exception as e:
            self._logger.error(f"Failed to list saved states: {e}")
            return []
    
    def delete_state(self, filename: str) -> bool:
        """Delete a saved battle state file.
        
        Args:
            filename: Name of file to delete
            
        Returns:
            True if successful, False otherwise
        """
        try:
            filepath = self.storage_dir / filename
            if filepath.exists():
                filepath.unlink()
                self._logger.info(f"Deleted battle state {filename}")
                return True
            else:
                self._logger.warning(f"Battle state file {filename} not found")
                return False
                
        except Exception as e:
            self._logger.error(f"Failed to delete battle state {filename}: {e}")
            return False