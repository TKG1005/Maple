# src/action/action_helper.py
from typing import List, Tuple, Dict, Union, TypedDict
from collections import OrderedDict
import numpy as np
import logging

from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.player.player import Player
from poke_env.player.battle_order import BattleOrder

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Mask generation utilities
# -----------------------------------------------------------------------------


def generate_action_mask(
    available_moves: List[Move],
    available_switches: List[Pokemon],
    can_tera: bool,
    force_switch: bool = False,
) -> np.ndarray:
    """
    Return a fixed-length (11) action mask.

    Slot layout (indices):
        0-3 : normal moves
        4-7 : terastal moves (mirrors normal move slots)
        8-9 : switches
       10   : Struggle (when no other moves are usable)
    """
    mask = np.zeros(11, dtype=np.int8)

    # --- Move slots --------------------------------------------------------
    if not force_switch:
        for i in range(min(4, len(available_moves))):
            mask[i] = 1

        # --- Terastal move slots ------------------------------------------
        if can_tera:
            for i in range(min(4, len(available_moves))):
                mask[4 + i] = 1

    # --- Switch slots ------------------------------------------------------
    for i in range(min(2, len(available_switches))):
        mask[8 + i] = 1

    return mask


# -----------------------------------------------------------------------------
# Action helper utilities
# -----------------------------------------------------------------------------


class ActionDetail(TypedDict):
    type: str
    name: str
    id: Union[str, int]  # move.id or pokemon.species


def get_action_mapping(battle: Battle) -> OrderedDict[int, Tuple[str, Union[str, int], bool]]:
    """Return a fixed length mapping for the current battle."""

    force_switch = battle.force_switch

    # List all moves in slot order
    active_moves: List[Move] = []
    active = getattr(battle, "active_pokemon", None)
    if active is not None:
        try:
            active_moves = list(active.moves.values())
        except Exception:
            active_moves = []

    available_move_ids = {m.id for m in battle.available_moves}
    switches: List[Pokemon] = battle.available_switches
    
    # Debug: Log available switches and their status
    logger.info(
        "[ACTION MASK DEBUG] Available switches: %s",
        [(getattr(p, '_ident', '?'), getattr(p, 'species', '?'), getattr(p, 'pid', '?'),
          getattr(p, '_current_hp', '?'), getattr(p, 'fainted', '?')) 
         for p in battle.available_switches]
    )
    
    # Filter out fainted Pokemon from switches
    switches = [p for p in switches if not getattr(p, 'fainted', False)]
    logger.info(
        "[ACTION MASK DEBUG] Non-fainted switches: %s",
        [(getattr(p, '_ident', '?'), getattr(p, 'species', '?'), getattr(p, 'pid', '?')) for p in switches]
    )
    
    # Double-check that switches don't include the active Pokemon
    # This is a workaround for a poke-env bug where active status might be stale
    if active is not None:
        # Use Pokemon's unique identifier (_ident) instead of species name for filtering
        # This properly handles Ditto transform cases where species changes but identity remains
        active_ident = getattr(active, '_ident', None)
        active_species = getattr(active, 'species', 'unknown')
        
        # Debug: Log Ditto transformation details
        if 'ditto' in active_species.lower() or any('ditto' in getattr(p, '_ident', '').lower() for p in switches):
            logger.info(
                "[DITTO DEBUG] Active: %s (ident=%s, species=%s), Available switches: %s",
                active,
                active_ident,
                active_species,
                [(getattr(p, '_ident', '?'), getattr(p, 'species', '?')) for p in battle.available_switches]
            )
        
        if active_ident:
            original_switches = len(battle.available_switches)
            switches = [p for p in switches if getattr(p, '_ident', '') != active_ident]
            if len(switches) != original_switches:
                logger.warning(
                    "Filtered out active Pokemon %s from available_switches. "
                    "Original: %s, Filtered: %s",
                    active_ident,
                    [getattr(p, '_ident', '?') for p in battle.available_switches],
                    [getattr(p, '_ident', '?') for p in switches]
                )

    mapping: OrderedDict[int, Tuple[str, Union[str, int], bool]] = OrderedDict()

    # Move slots 0-3
    for i in range(4):
        if i < len(active_moves):
            mv = active_moves[i]
            move_id = mv.id
            disabled = (
                force_switch
                or mv.current_pp == 0
                or mv.id not in available_move_ids
            )
        else:
            move_id = ""
            disabled = True
        mapping[i] = ("move", move_id, disabled)

    can_tera = (battle.can_tera is not None) and (not force_switch)
    # Terastal slots 4-7
    for i in range(4):
        if i < len(active_moves):
            mv = active_moves[i]
            move_id = mv.id
            disabled = (
                force_switch
                or (battle.can_tera is None)
                or mv.current_pp == 0
                or mv.id not in available_move_ids
            )
        else:
            move_id = ""
            disabled = True
        mapping[4 + i] = ("terastal", move_id, disabled)

    # Switch slots 8-9 (map to team positions, not filtered switches)
    # Find valid switch positions based on original team order
    team_list = list(battle.team.values())
    active_pokemon = getattr(battle, 'active_pokemon', None)
    
    logger.info(
        "[MAPPING DEBUG] Team list: %s, Active: %s",
        [(i, getattr(p, '_ident', '?'), getattr(p, 'species', '?'), getattr(p, 'fainted', '?')) 
         for i, p in enumerate(team_list)],
        getattr(active_pokemon, '_ident', '?') if active_pokemon else None
    )
    
    available_team_positions = []
    for team_idx, team_pokemon in enumerate(team_list):
        # Use multiple methods to determine if Pokemon is active
        is_active_by_ident = (active_pokemon and 
                             getattr(active_pokemon, '_ident', '') == getattr(team_pokemon, '_ident', '') and
                             getattr(active_pokemon, '_ident', '') != '?')
        is_active_by_species = (active_pokemon and 
                               getattr(active_pokemon, 'species', '') == getattr(team_pokemon, 'species', '') and
                               getattr(active_pokemon, 'species', '') != '')
        is_active_by_object = (active_pokemon is team_pokemon)
        
        # Use available_switches to determine which Pokemon can be switched to
        is_in_available_switches = any(
            getattr(sw, 'species', '') == getattr(team_pokemon, 'species', '') and
            getattr(sw, '_ident', '') == getattr(team_pokemon, '_ident', '')
            for sw in switches
        )
        
        is_fainted = getattr(team_pokemon, 'fainted', False)
        
        # A Pokemon can be switched to if:
        # 1. It's in available_switches (poke-env's authoritative list)
        # 2. It's not fainted
        can_switch = is_in_available_switches and not is_fainted
        
        logger.info(
            "[MAPPING DEBUG] Team idx %d: %s, ident_active=%s, species_active=%s, obj_active=%s, "
            "in_available=%s, fainted=%s, can_switch=%s",
            team_idx, getattr(team_pokemon, 'species', '?'),
            is_active_by_ident, is_active_by_species, is_active_by_object,
            is_in_available_switches, is_fainted, can_switch
        )
        
        if can_switch:
            available_team_positions.append(team_idx)
    
    logger.info("[MAPPING DEBUG] Available team positions for switch: %s", available_team_positions)
    
    # Map switch slots to available team positions
    for i in range(2):  # Only support 2 switch slots (8, 9)
        if i < len(available_team_positions):
            team_position = available_team_positions[i]
            disabled = False
        else:
            team_position = None
            disabled = True
        
        logger.info("[MAPPING DEBUG] Switch slot %d -> team_position=%s, disabled=%s", 8+i, team_position, disabled)
        mapping[8 + i] = ("switch", team_position, disabled)

    # Struggle slot 10 (enabled only when no other moves are usable)
    only_struggle = (
        len(battle.available_moves) == 1
        and getattr(battle.available_moves[0], "id", "") == "struggle"
    )
    mapping[10] = ("move", "struggle", not only_struggle)

    return mapping


def get_available_actions(
    battle: Battle,
) -> Tuple[np.ndarray, OrderedDict[int, Tuple[str, Union[str, int], bool]]]:
    """Build an action mask and index mapping for the current battle state."""

    mapping = get_action_mapping(battle)
    mask = np.array([0 if mapping[i][2] else 1 for i in range(11)], dtype=np.int8)

    return mask, mapping


def get_available_actions_with_details(
    battle: Battle,
) -> Tuple[np.ndarray, Dict[int, ActionDetail]]:
    """
    Like `get_available_actions` but also attaches human‑readable information
    useful for rendering and debugging.
    """
    mask, mapping = get_available_actions(battle)
    detailed: Dict[int, ActionDetail] = {}

    moves_sorted = sorted(battle.available_moves, key=lambda m: m.id)
    switches_sorted = battle.available_switches

    for action_idx, (action_type, sub_id, disabled) in mapping.items():
        if action_type == "move":
            mv = next((m for m in moves_sorted if m.id == sub_id), None)
            if mv is not None:
                detailed[action_idx] = {
                    "type": "move",
                    "name": f"Move: {mv.id} (PP: {mv.current_pp}/{mv.max_pp})",
                    "id": mv.id,
                }
                continue
            if sub_id == "struggle":
                detailed[action_idx] = {
                    "type": "move",
                    "name": "Move: struggle",
                    "id": "struggle",
                }
                continue

        elif action_type == "terastal" and battle.can_tera:
            mv = next((m for m in moves_sorted if m.id == sub_id), None)
            if mv is not None:
                detailed[action_idx] = {
                    "type": "terastal",
                    "name": f"Terastal Move: {mv.id} (PP: {mv.current_pp}/{mv.max_pp})",
                    "id": f"tera_{mv.id}",
                }
                continue

        elif action_type == "switch" and isinstance(sub_id, int) and sub_id < len(switches_sorted):
            pkmn = switches_sorted[sub_id]
            detailed[action_idx] = {
                "type": "switch",
                "name": f"Switch to: {pkmn.species} (HP: {pkmn.current_hp_fraction * 100:.1f}%)",
                "id": pkmn.species,
            }

        else:
            detailed[action_idx] = {
                "type": action_type,
                "name": "Invalid",
                "id": "invalid",
            }

    return mask, detailed


def action_index_to_order_from_mapping(
    player: Player,
    battle: Battle,
    action_index: int,
    mapping: Dict[int, Tuple[str, Union[str, int], bool]],
) -> BattleOrder:
    """Translate ``action_index`` using a precomputed ``mapping``."""

    moves_sorted: List[Move] = sorted(battle.available_moves, key=lambda m: m.id)
    switches_list: List[Pokemon] = battle.available_switches

    if action_index not in mapping:
        logger.error(
            "Invalid or unavailable action index: %s for %s at turn %s in %s. Available: %s",
            action_index,
            getattr(player, 'username', '?'),
            getattr(battle, 'turn', '?'),
            getattr(battle, 'battle_tag', '?'),
            mapping,
        )
        raise ValueError(
            f"Invalid or unavailable action index: {action_index}. Available: {mapping}"
        )

    action_type, sub_id, disabled = mapping[action_index]

    if disabled:
        raise DisabledMoveError(f"Action index {action_index} is disabled")

    if action_type == "move":
        mv = next((m for m in moves_sorted if m.id == sub_id), None)
        if mv is None:
            if sub_id == "struggle":
                return player.create_order("move struggle")
            raise ValueError(f"Move id {sub_id} not available.")
        return player.create_order(mv, terastallize=False)

    if action_type == "terastal":
        mv = next((m for m in moves_sorted if m.id == sub_id), None)
        if mv is None:
            raise ValueError(f"Terastal move id {sub_id} not available.")
        if battle.can_tera is None:
            return player.create_order(mv, terastallize=False)
        return player.create_order(mv, terastallize=True)

    if action_type == "switch":
        # sub_id is now the team position (0-based index in battle.team)
        if not isinstance(sub_id, int):
            raise ValueError(f"Switch team position must be int, got: {sub_id}")
        
        team_list = list(battle.team.values())
        if sub_id >= len(team_list):
            raise ValueError(f"Switch team position {sub_id} out of range (team size: {len(team_list)}).")
        
        # Get Pokemon at the specified team position
        pokemon = team_list[sub_id]
        
        # Get selected team order from request message
        request = getattr(battle, '_last_request', None)
        if not request:
            raise ValueError(f"No request data available for battle {battle.battle_tag}")
        
        if 'side' not in request:
            raise ValueError(f"No 'side' data in request: {request}")
        
        if 'pokemon' not in request['side']:
            raise ValueError(f"No 'pokemon' data in request side: {request['side']}")
        
        selected_team = request['side']['pokemon']
        
        # Get the player role to construct full identifier
        player_role = getattr(battle, '_player_role', None)
        if not player_role:
            raise ValueError(f"No player role available for battle {battle.battle_tag}")
        
        pokemon_full_ident = f"{player_role}: {pokemon.name}"
        
        logger.info(
            "[SWITCH DEBUG] Selected team: %s, Looking for: %s",
            [mon['ident'] for mon in selected_team],
            pokemon_full_ident
        )
        
        # Find position in selected team (1-based for Pokemon Showdown)
        for i, selected_mon in enumerate(selected_team):
            if selected_mon['ident'] == pokemon_full_ident:
                team_position = i + 1
                logger.info(
                    "%s: choose switch to %s (position %d in selected team)",
                    player.username,
                    pokemon.species,
                    team_position
                )
                break
        else:
            # Strict error - no fallback
            raise ValueError(
                f"Pokemon {pokemon_full_ident} (species: {pokemon.species}) not found in selected team. "
                f"Selected team: {[mon['ident'] for mon in selected_team]}, "
                f"Battle team: {[f'{player_role}: {p.name}' for p in team_list]}"
            )
        
        if team_position is not None:
            # Create custom BattleOrder with position-based switch command
            class PositionalSwitchOrder(BattleOrder):
                def __init__(self, position: int):
                    self.position = position
                
                @property
                def message(self) -> str:
                    return f"/choose switch {self.position}"
            
            logger.info(
                "[SWITCH FIX] Using position-based switch: position=%d for Pokemon=%s (ident=%s)",
                team_position,
                getattr(pokemon, 'species', 'unknown'),
                getattr(pokemon, '_ident', 'unknown')
            )
            return PositionalSwitchOrder(team_position)
        else:
            # Strict error - no fallback to avoid species name conflicts
            raise ValueError(
                f"Could not find team position for Pokemon {getattr(pokemon, '_ident', 'unknown')}. "
                f"Team: {[getattr(p, '_ident', '?') for p in battle.team.values()]}"
            )

    raise ValueError(f"Unknown action_type: {action_type}")
def action_index_to_order(
    player: Player, battle: Battle, action_index: int
) -> BattleOrder:
    """
    Translate an action index into a Showdown‑compatible command string via
    Player.create_order.
    """
    moves_sorted: List[Move] = sorted(battle.available_moves, key=lambda m: m.id)
    switches_list: List[Pokemon] = battle.available_switches

    _, mapping = get_available_actions(battle)

    if action_index not in mapping:
        raise ValueError(
            f"Invalid or unavailable action index: {action_index}. Available: {mapping}"
        )

    action_type, sub_id, disabled = mapping[action_index]

    if disabled:
        raise DisabledMoveError(f"Action index {action_index} is disabled")

    if action_type == "move":
        mv = next((m for m in moves_sorted if m.id == sub_id), None)
        if mv is None:
            if sub_id == "struggle":
                return player.create_order("move struggle")
            raise ValueError(f"Move id {sub_id} not available.")
        return player.create_order(mv, terastallize=False)

    if action_type == "terastal":
        mv = next((m for m in moves_sorted if m.id == sub_id), None)
        if mv is None:
            raise ValueError(f"Terastal move id {sub_id} not available.")
        if battle.can_tera is None:
            return player.create_order(mv, terastallize=False)
        return player.create_order(mv, terastallize=True)

    if action_type == "switch":
        # sub_id is now the team position (0-based index in battle.team)
        if not isinstance(sub_id, int):
            raise ValueError(f"Switch team position must be int, got: {sub_id}")
        
        team_list = list(battle.team.values())
        if sub_id >= len(team_list):
            raise ValueError(f"Switch team position {sub_id} out of range (team size: {len(team_list)}).")
        
        # Get Pokemon at the specified team position
        pokemon = team_list[sub_id]
        
        # Get selected team order from request message
        request = getattr(battle, '_last_request', None)
        if not request:
            raise ValueError(f"No request data available for battle {battle.battle_tag}")
        
        if 'side' not in request:
            raise ValueError(f"No 'side' data in request: {request}")
        
        if 'pokemon' not in request['side']:
            raise ValueError(f"No 'pokemon' data in request side: {request['side']}")
        
        selected_team = request['side']['pokemon']
        
        # Get the player role to construct full identifier
        player_role = getattr(battle, '_player_role', None)
        if not player_role:
            raise ValueError(f"No player role available for battle {battle.battle_tag}")
        
        pokemon_full_ident = f"{player_role}: {pokemon.name}"
        
        logger.info(
            "[SWITCH DEBUG] Selected team: %s, Looking for: %s",
            [mon['ident'] for mon in selected_team],
            pokemon_full_ident
        )
        
        # Find position in selected team (1-based for Pokemon Showdown)
        for i, selected_mon in enumerate(selected_team):
            if selected_mon['ident'] == pokemon_full_ident:
                team_position = i + 1
                logger.info(
                    "%s: choose switch to %s (position %d in selected team)",
                    player.username,
                    pokemon.species,
                    team_position
                )
                break
        else:
            # Strict error - no fallback
            raise ValueError(
                f"Pokemon {pokemon_full_ident} (species: {pokemon.species}) not found in selected team. "
                f"Selected team: {[mon['ident'] for mon in selected_team]}, "
                f"Battle team: {[f'{player_role}: {p.name}' for p in team_list]}"
            )
        
        if team_position is not None:
            # Create custom BattleOrder with position-based switch command
            class PositionalSwitchOrder(BattleOrder):
                def __init__(self, position: int):
                    self.position = position
                
                @property
                def message(self) -> str:
                    return f"/choose switch {self.position}"
            
            logger.info(
                "[SWITCH FIX] Using position-based switch: position=%d for Pokemon=%s (ident=%s)",
                team_position,
                getattr(pokemon, 'species', 'unknown'),
                getattr(pokemon, '_ident', 'unknown')
            )
            return PositionalSwitchOrder(team_position)
        else:
            # Strict error - no fallback to avoid species name conflicts
            raise ValueError(
                f"Could not find team position for Pokemon {getattr(pokemon, '_ident', 'unknown')}. "
                f"Team: {[getattr(p, '_ident', '?') for p in battle.team.values()]}"
            )

    raise ValueError(f"Unknown action_type: {action_type}")

class DisabledMoveError(ValueError):
    pass

