"""Team loading utilities for Pokemon battles."""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import List, Optional, Dict

from .team_cache import TeamCacheManager

logger = logging.getLogger(__name__)


class TeamLoader:
    """Loads and manages Pokemon teams from files with caching support."""
    
    def __init__(self, teams_dir: str | Path) -> None:
        """Initialize TeamLoader with teams directory.
        
        Parameters
        ----------
        teams_dir : str | Path
            Directory containing team files (.txt files with Pokemon Showdown format)
        """
        self.teams_dir = Path(teams_dir)
        self.teams: List[str] = []
        self.performance_stats: Dict = {}
        self._load_teams()
    
    def _load_teams(self) -> None:
        """Load all team files using the cached team manager."""
        load_start_time = time.time()
        
        # Use cached team manager for optimized loading
        self.teams, self.performance_stats = TeamCacheManager.get_teams(self.teams_dir)
        
        total_load_time = time.time() - load_start_time
        
        if self.teams:
            logger.info("TeamLoader initialized: %d teams loaded in %.3fs", 
                       len(self.teams), total_load_time)
            
            # Log performance details if this is the first load (not from cache)
            if self.performance_stats.get('cache_hits', 0) == 0:
                logger.info("Team loading performance: %.3fs I/O, %.3fs parsing, %d bytes total",
                           self.performance_stats.get('io_time', 0),
                           self.performance_stats.get('parse_time', 0), 
                           self.performance_stats.get('total_size', 0))
        else:
            logger.warning("No teams loaded from %s", self.teams_dir)
    
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
    
    def get_performance_stats(self) -> Dict:
        """Get detailed performance statistics for team loading.
        
        Returns
        -------
        Dict
            Performance statistics including load times and cache information
        """
        return self.performance_stats.copy()
    
    def print_performance_report(self) -> None:
        """Print a performance report for this TeamLoader instance."""
        if not self.performance_stats:
            logger.info("No performance statistics available")
            return
        
        stats = self.performance_stats
        print(f"\nTeamLoader Performance Report for: {stats.get('teams_dir', 'Unknown')}")
        print("-" * 60)
        print(f"Teams loaded: {stats.get('team_count', 0)}")
        print(f"Total load time: {stats.get('total_load_time', 0):.3f}s")
        print(f"Average per team: {stats.get('avg_load_time_per_team', 0):.3f}s")
        print(f"Cache hits: {stats.get('cache_hits', 0)}")
        
        if 'io_time' in stats:
            print(f"I/O time: {stats['io_time']:.3f}s")
            print(f"Parse time: {stats['parse_time']:.3f}s")
            print(f"Total data size: {stats['total_size']:,} bytes")
            print(f"Average file size: {stats['avg_file_size']:.0f} bytes")
        
        print("-" * 60)


__all__ = ["TeamLoader"]