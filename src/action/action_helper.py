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

    # Switch slots 8-9
    for i in range(2):
        disabled = i >= len(switches)
        mapping[8 + i] = ("switch", i, disabled)

    # Struggle slot 10 (enabled only when no other moves are usable)
    mapping[10] = ("move", "struggle", True)

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
        if not isinstance(sub_id, int) or sub_id >= len(switches_list):
            raise ValueError(f"Switch index {sub_id} out of range.")
        return player.create_order(switches_list[sub_id])

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
        if not isinstance(sub_id, int) or sub_id >= len(switches_list):
            raise ValueError(f"Switch index {sub_id} out of range.")
        return player.create_order(switches_list[sub_id])

    raise ValueError(f"Unknown action_type: {action_type}")

class DisabledMoveError(ValueError):
    pass

