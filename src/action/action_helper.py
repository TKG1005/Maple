# src/action/action_helper.py
from typing import List, Tuple, Dict, Union, TypedDict
import numpy as np

from poke_env.environment.battle import Battle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from poke_env.player.player import Player
from poke_env.player.battle_order import BattleOrder

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
    Return a fixed-length (10) action mask.

    Slot layout (indices):
        0-3 : normal moves
        4-7 : terastal moves (mirrors normal move slots)
        8-9 : switches
    """
    mask = np.zeros(10, dtype=np.int8)

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


def get_available_actions(
    battle: Battle,
) -> Tuple[np.ndarray, Dict[int, Tuple[str, int]]]:
    """
    Build an action mask and index mapping for the current battle state.
    """
    force_switch = battle.force_switch
    moves_sorted: List[Move] = (
        [] if force_switch else sorted(battle.available_moves, key=lambda m: m.id)
    )
    switches: List[Pokemon] = battle.available_switches
    can_tera: bool = (battle.can_tera is not None) and (not force_switch)

    mask = generate_action_mask(moves_sorted, switches, can_tera, force_switch)

    mapping: Dict[int, Tuple[str, int]] = {}

    # Move slots 0‒3
    if not force_switch:
        for i in range(min(4, len(moves_sorted))):
            mapping[i] = ("move", i)

        # Terastal slots 4‒7
        if can_tera:
            for i in range(min(4, len(moves_sorted))):
                mapping[4 + i] = ("terastal", i)

    # Switch slots 8‒9
    for i in range(min(2, len(switches))):
        mapping[8 + i] = ("switch", i)

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

    for action_idx, (action_type, sub_idx) in mapping.items():
        if action_type == "move" and sub_idx < len(moves_sorted):
            mv = moves_sorted[sub_idx]
            detailed[action_idx] = {
                "type": "move",
                "name": f"Move: {mv.id} (PP: {mv.current_pp}/{mv.max_pp})",
                "id": mv.id,
            }

        elif (
            action_type == "terastal"
            and sub_idx < len(moves_sorted)
            and battle.can_tera
        ):
            mv = moves_sorted[sub_idx]
            detailed[action_idx] = {
                "type": "terastal",
                "name": f"Terastal Move: {mv.id} (PP: {mv.current_pp}/{mv.max_pp})",
                "id": f"tera_{mv.id}",
            }

        elif action_type == "switch" and sub_idx < len(switches_sorted):
            pkmn = switches_sorted[sub_idx]
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

    action_type, sub_idx = mapping[action_index]

    if action_type == "move":
        if sub_idx >= len(moves_sorted):
            raise ValueError(f"Move index {sub_idx} out of range.")
        return player.create_order(moves_sorted[sub_idx], terastallize=False)

    if action_type == "terastal":
        if not battle.can_tera:
            raise ValueError("Terastallization not available.")
        if sub_idx >= len(moves_sorted):
            raise ValueError(f"Terastal move index {sub_idx} out of range.")
        return player.create_order(moves_sorted[sub_idx], terastallize=True)

    if action_type == "switch":
        if sub_idx >= len(switches_list):
            raise ValueError(f"Switch index {sub_idx} out of range.")
        return player.create_order(switches_list[sub_idx])

    raise ValueError(f"Unknown action_type: {action_type}")
