"""Team loading utilities for Pokemon battles."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class TeamLoader:
    """Loads and manages Pokemon teams from files."""
    
    def __init__(self, teams_dir: str | Path) -> None:
        """Initialize TeamLoader with teams directory.
        
        Parameters
        ----------
        teams_dir : str | Path
            Directory containing team files (.txt files with Pokemon Showdown format)
        """
        self.teams_dir = Path(teams_dir)
        self.team_files: List[Path] = []
        self.teams: List[str] = []
        self._load_teams()
    
    def _load_teams(self) -> None:
        """Load all team files from the teams directory."""
        if not self.teams_dir.exists():
            logger.warning("Teams directory does not exist: %s", self.teams_dir)
            return
        
        # Find all .txt files in teams directory
        team_files = list(self.teams_dir.glob("*.txt"))
        
        if not team_files:
            logger.warning("No team files found in: %s", self.teams_dir)
            return
        
        logger.info("Loading teams from %s", self.teams_dir)
        
        for team_file in team_files:
            try:
                team_content = team_file.read_text(encoding="utf-8").strip()
                if team_content:
                    self.team_files.append(team_file)
                    self.teams.append(team_content)
                    logger.debug("Loaded team from %s", team_file.name)
                else:
                    logger.warning("Empty team file: %s", team_file.name)
            except Exception as e:
                logger.error("Failed to load team from %s: %s", team_file.name, e)
        
        logger.info("Successfully loaded %d teams", len(self.teams))
    
    def get_random_team(self) -> Optional[str]:
        """Get a randomly selected team.
        
        Returns
        -------
        str | None
            Random team content in Pokemon Showdown format, or None if no teams available
        """
        if not self.teams:
            logger.warning("No teams available for selection")
            return None
        
        selected_team = random.choice(self.teams)
        return selected_team
    
    def get_team_by_index(self, index: int) -> Optional[str]:
        """Get a team by its index.
        
        Parameters
        ----------
        index : int
            Index of the team to retrieve
            
        Returns
        -------
        str | None
            Team content or None if index is invalid
        """
        if 0 <= index < len(self.teams):
            return self.teams[index]
        return None
    
    def get_team_count(self) -> int:
        """Get the number of available teams.
        
        Returns
        -------
        int
            Number of loaded teams
        """
        return len(self.teams)
    
    def get_team_files(self) -> List[str]:
        """Get list of team file names.
        
        Returns
        -------
        List[str]
            List of team file names
        """
        return [f.name for f in self.team_files]


__all__ = ["TeamLoader"]