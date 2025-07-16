"""Custom battle implementation that tracks fail and immune actions."""

from __future__ import annotations

from poke_env.environment.battle import Battle
from typing import Any, List


class CustomBattle(Battle):
    """Custom battle class that tracks fail and immune actions for reward calculation."""
    
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Initialize fail and immune action tracking
        self._last_fail_action: bool = False
        self._last_immune_action: bool = False
    
    @property
    def last_fail_action(self) -> bool:
        """Return True if the last action failed."""
        return self._last_fail_action
    
    @property
    def last_immune_action(self) -> bool:
        """Return True if the last action was immune."""
        return self._last_immune_action
    
    def reset_invalid_action(self) -> None:
        """Reset the invalid action flags."""
        self._last_fail_action = False
        self._last_immune_action = False
    
    def parse_message(self, split_message: List[str]) -> None:
        """Override to intercept -fail and -immune messages."""
        # Debug print for testing
        # print(f"DEBUG: Processing message: {split_message}")
        # print(f"DEBUG: has _player_role: {hasattr(self, '_player_role')}")
        # if hasattr(self, '_player_role'):
        #     print(f"DEBUG: _player_role: {self._player_role}")
        
        # Check for -fail messages BEFORE calling parent
        if len(split_message) >= 2 and split_message[1] == "-fail":
            # Only set flag if this is our player's action
            if len(split_message) >= 3:
                pokemon_ident = split_message[2]
                # Check if this is our player's Pokemon
                if hasattr(self, '_player_role') and self._player_role is not None and pokemon_ident.startswith(f"p{self._player_role}"):
                    self._last_fail_action = True
                # For testing purposes, assume p1 is our player if _player_role is not set or None
                elif (not hasattr(self, '_player_role') or self._player_role is None) and pokemon_ident.startswith("p1"):
                    self._last_fail_action = True
        
        # Check for -immune messages BEFORE calling parent
        if len(split_message) >= 2 and split_message[1] == "-immune":
            # Only set flag if this is our opponent's Pokemon (immune to our action)
            if len(split_message) >= 3:
                pokemon_ident = split_message[2]
                if hasattr(self, '_player_role') and self._player_role is not None:
                    # Check if this is our opponent's Pokemon (immune to our action)
                    opponent_role = "1" if self._player_role == "2" else "2"
                    if pokemon_ident.startswith(f"p{opponent_role}"):
                        self._last_immune_action = True
                # For testing purposes, assume p2 is opponent if _player_role is not set or None
                elif (not hasattr(self, '_player_role') or self._player_role is None) and pokemon_ident.startswith("p2"):
                    self._last_immune_action = True
        
        # Reset flags at the start of each turn
        if len(split_message) >= 2 and split_message[1] == "turn":
            self._last_fail_action = False
            self._last_immune_action = False
            
        # Call parent implementation after our processing
        super().parse_message(split_message)