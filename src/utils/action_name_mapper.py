"""Maps action indices to human-readable names for Pokemon battles."""

from __future__ import annotations

from typing import Optional, List
from poke_env.environment import Battle


class ActionNameMapper:
    """Convert action indices to descriptive names based on battle state."""
    
    @staticmethod
    def get_action_name(
        action_idx: int, 
        battle: Optional[Battle] = None,
        player_id: Optional[str] = None
    ) -> str:
        """Get human-readable name for an action index.
        
        Args:
            action_idx: The action index
            battle: Current battle state from poke-env
            player_id: ID of the player making the action
            
        Returns:
            Human-readable action name
        """
        if battle is None:
            # Fallback when no battle context
            if action_idx < 4:
                return f"Move {action_idx + 1}"
            elif action_idx < 10:
                return f"Switch to Pokemon {action_idx - 3}"
            else:
                return f"Action {action_idx}"
                
        # Determine if we're looking at our moves or opponent's
        active_pokemon = battle.active_pokemon
        team = battle.team
        
        if active_pokemon is None:
            return f"Action {action_idx}"
            
        # Move actions (0-3)
        if action_idx < 4:
            moves = active_pokemon.moves
            move_names = list(moves.keys())
            if action_idx < len(move_names):
                move = moves[move_names[action_idx]]
                # Get the move's actual name from its ID
                move_display_name = move.id.replace('-', ' ').title()
                return f"Use {move_display_name}"
            else:
                return f"Move {action_idx + 1} (unavailable)"
                
        # Switch actions (4-9 for 6 Pokemon team)
        elif action_idx < 10:
            switch_idx = action_idx - 4
            available_switches = [p for p in team.values() if not p.fainted and p != active_pokemon]
            
            if switch_idx < len(available_switches):
                target_pokemon = available_switches[switch_idx]
                # Get the Pokemon's species name and level
                species_name = target_pokemon.species.title()
                level_str = f" L{target_pokemon.level}" if target_pokemon.level != 100 else ""
                return f"Switch to {species_name}{level_str}"
            else:
                return f"Switch {switch_idx + 1} (unavailable)"
                
        # Other actions (should not occur in standard battles)
        else:
            return f"Special Action {action_idx}"
            
    @staticmethod
    def get_all_action_names(
        action_mask: List[bool],
        battle: Optional[Battle] = None,
        player_id: Optional[str] = None
    ) -> List[tuple[int, str, bool]]:
        """Get names for all possible actions with their validity.
        
        Args:
            action_mask: Boolean mask indicating valid actions
            battle: Current battle state
            player_id: ID of the player
            
        Returns:
            List of tuples (action_idx, action_name, is_valid)
        """
        results = []
        for idx, is_valid in enumerate(action_mask):
            name = ActionNameMapper.get_action_name(idx, battle, player_id)
            results.append((idx, name, is_valid))
        return results