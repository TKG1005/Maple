"""Action probability logger for evaluation."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
from poke_env.environment import Battle

from .action_name_mapper import ActionNameMapper
from src.action.action_helper import get_available_actions_with_details


class ActionProbabilityLogger:
    """Logs action probabilities during battles for analysis."""
    
    def __init__(self, log_dir: str = "replays/action-probs", model_name: str = "model"):
        """Initialize the logger.
        
        Args:
            log_dir: Directory to save log files
            model_name: Name of the model being evaluated
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"action_probs_{model_name}_{timestamp}.json"
        self.txt_file = self.log_dir / f"action_probs_{model_name}_{timestamp}.txt"
        
        self.battles: List[Dict[str, Any]] = []
        self.current_battle: Optional[Dict[str, Any]] = None
        self.logger = logging.getLogger(__name__)
        
    def start_battle(self, battle_id: int, player_names: tuple[str, str]) -> None:
        """Start logging a new battle."""
        self.current_battle = {
            "battle_id": battle_id,
            "player_0": player_names[0],
            "player_1": player_names[1],
            "turns": []
        }
        
    def log_turn(
        self, 
        player_id: str,
        turn: int, 
        action_probs: np.ndarray, 
        action_mask: np.ndarray,
        selected_action: int,
        battle: Optional[Battle] = None
    ) -> None:
        """Log action probabilities for a turn.
        
        Args:
            player_id: ID of the player making the action
            turn: Turn number
            action_probs: Probability distribution over actions
            action_mask: Boolean mask of valid actions
            selected_action: The action that was selected
            battle: Optional Battle object for context
        """
        if self.current_battle is None:
            return
            
        turn_data = {
            "turn": turn,
            "player": player_id,
            "selected_action": int(selected_action),
            "action_probs": action_probs.tolist(),
            "action_mask": action_mask.tolist(),
            "valid_actions": np.where(action_mask)[0].tolist()
        }
        
        # Add action names if battle context available (use detailed mapping for accuracy)
        if battle:
            try:
                _, details = get_available_actions_with_details(battle)
                if selected_action in details:
                    turn_data["selected_action_name"] = details[selected_action]["name"]
                else:
                    turn_data["selected_action_name"] = ActionNameMapper.get_action_name(selected_action, battle, player_id)
            except Exception:
                turn_data["selected_action_name"] = ActionNameMapper.get_action_name(selected_action, battle, player_id)
            
        self.current_battle["turns"].append(turn_data)
        
        # Also write human-readable format
        self._write_human_readable(player_id, turn, action_probs, action_mask, selected_action, battle)
        
    def _write_human_readable(
        self,
        player_id: str,
        turn: int,
        action_probs: np.ndarray,
        action_mask: np.ndarray,
        selected_action: int,
        battle: Optional[Battle] = None
    ) -> None:
        """Write human-readable log entry."""
        with open(self.txt_file, 'a', encoding='utf-8') as f:
            if turn == 1 and len(self.current_battle["turns"]) == 1:
                f.write(f"\n{'='*60}\n")
                f.write(f"Battle {self.current_battle['battle_id']}: "
                       f"{self.current_battle['player_0']} vs {self.current_battle['player_1']}\n")
                f.write(f"{'='*60}\n")
                
            f.write(f"\nTurn {turn} - {player_id}:\n")
            
            # Sort actions by probability
            valid_indices = np.where(action_mask)[0]
            if len(valid_indices) > 0:
                probs_and_indices = [(action_probs[i], i) for i in valid_indices]
                probs_and_indices.sort(reverse=True)
                # Try to get accurate names via detailed mapping
                details = None
                if battle:
                    try:
                        _, details = get_available_actions_with_details(battle)
                    except Exception:
                        details = None

                f.write("Action probabilities:\n")
                for prob, idx in probs_and_indices:
                    if details and idx in details:
                        action_name = details[idx]["name"]
                    else:
                        action_name = ActionNameMapper.get_action_name(idx, battle, player_id)
                    selected = " [SELECTED]" if idx == selected_action else ""
                    f.write(f"  {action_name}: {prob*100:.1f}%{selected}\n")
                    
                # Show masked actions count
                masked_count = len(action_mask) - len(valid_indices)
                if masked_count > 0:
                    f.write(f"  ({masked_count} invalid actions masked)\n")
            else:
                f.write("  No valid actions available\n")
                
            
    def end_battle(self, winner: str, final_rewards: Dict[str, float]) -> None:
        """End the current battle and save results."""
        if self.current_battle is None:
            return
            
        self.current_battle["winner"] = winner
        self.current_battle["final_rewards"] = final_rewards
        self.battles.append(self.current_battle)
        
        # Write summary to text file
        with open(self.txt_file, 'a', encoding='utf-8') as f:
            f.write(f"\nBattle Result: {winner} wins\n")
            f.write(f"Final rewards: {final_rewards}\n")
            
        self.current_battle = None
        
    def save(self) -> None:
        """Save all logged data to JSON file."""
        data = {
            "timestamp": datetime.now().isoformat(),
            "total_battles": len(self.battles),
            "battles": self.battles
        }
        
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
            
        self.logger.info(f"Saved action probability logs to {self.log_file} and {self.txt_file}")
