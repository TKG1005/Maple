"""Team caching system for optimized team loading performance."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import threading

logger = logging.getLogger(__name__)


class TeamCacheManager:
    """Global team cache manager with performance monitoring."""
    
    # Global cache storage
    _cache: Dict[str, Dict[str, str]] = {}
    _cache_stats: Dict[str, Dict] = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_teams(cls, teams_dir: str | Path) -> Tuple[List[str], Dict]:
        """Get teams from cache or load them with performance monitoring.
        
        Parameters
        ----------
        teams_dir : str | Path
            Directory containing team files
            
        Returns
        -------
        Tuple[List[str], Dict]
            (list_of_team_contents, performance_stats)
        """
        teams_dir_str = str(Path(teams_dir).resolve())
        
        with cls._lock:
            # Check if already cached
            if teams_dir_str in cls._cache:
                logger.debug("Teams loaded from cache for: %s", teams_dir_str)
                stats = cls._cache_stats[teams_dir_str]
                stats['cache_hits'] += 1
                return list(cls._cache[teams_dir_str].values()), stats
            
            # Load teams with performance monitoring
            start_time = time.time()
            teams_data, load_stats = cls._load_teams_with_timing(Path(teams_dir))
            total_time = time.time() - start_time
            
            # Cache the results
            cls._cache[teams_dir_str] = teams_data
            
            # Store performance stats
            performance_stats = {
                'teams_dir': teams_dir_str,
                'team_count': len(teams_data),
                'total_load_time': total_time,
                'avg_load_time_per_team': total_time / len(teams_data) if teams_data else 0,
                'cache_hits': 0,
                'initial_load_time': time.strftime("%Y-%m-%d %H:%M:%S"),
                **load_stats
            }
            
            cls._cache_stats[teams_dir_str] = performance_stats
            
            logger.info("Teams cached for %s: %d teams loaded in %.3fs (avg %.3fs/team)", 
                       teams_dir_str, len(teams_data), total_time, 
                       performance_stats['avg_load_time_per_team'])
            
            return list(teams_data.values()), performance_stats
    
    @classmethod
    def _load_teams_with_timing(cls, teams_dir: Path) -> Tuple[Dict[str, str], Dict]:
        """Load teams with detailed timing information.
        
        Parameters
        ----------
        teams_dir : Path
            Directory containing team files
            
        Returns
        -------
        Tuple[Dict[str, str], Dict]
            (team_data_dict, detailed_timing_stats)
        """
        if not teams_dir.exists():
            logger.warning("Teams directory does not exist: %s", teams_dir)
            return {}, {'file_count': 0, 'io_time': 0, 'parse_time': 0, 'total_size': 0}
        
        team_files = list(teams_dir.glob("*.txt"))
        
        if not team_files:
            logger.warning("No team files found in: %s", teams_dir)
            return {}, {'file_count': 0, 'io_time': 0, 'parse_time': 0, 'total_size': 0}
        
        teams_data = {}
        total_io_time = 0
        total_parse_time = 0
        total_size = 0
        successful_loads = 0
        
        logger.info("Loading %d team files from %s", len(team_files), teams_dir)
        
        for team_file in team_files:
            try:
                # Time file I/O
                io_start = time.time()
                team_content = team_file.read_text(encoding="utf-8")
                io_time = time.time() - io_start
                total_io_time += io_time
                
                # Time parsing/processing
                parse_start = time.time()
                team_content = team_content.strip()
                parse_time = time.time() - parse_start
                total_parse_time += parse_time
                
                if team_content:
                    teams_data[team_file.name] = team_content
                    file_size = len(team_content.encode('utf-8'))
                    total_size += file_size
                    successful_loads += 1
                    
                    logger.debug("Loaded %s: %.3fms I/O, %.3fms parse, %d bytes", 
                               team_file.name, io_time * 1000, parse_time * 1000, file_size)
                else:
                    logger.warning("Empty team file: %s", team_file.name)
                    
            except Exception as e:
                logger.error("Failed to load team from %s: %s", team_file.name, e)
        
        timing_stats = {
            'file_count': len(team_files),
            'successful_loads': successful_loads,
            'io_time': total_io_time,
            'parse_time': total_parse_time,
            'total_size': total_size,
            'avg_file_size': total_size / successful_loads if successful_loads > 0 else 0
        }
        
        return teams_data, timing_stats
    
    @classmethod
    def get_cache_info(cls) -> Dict:
        """Get information about the current cache state.
        
        Returns
        -------
        Dict
            Cache information including stats for all cached directories
        """
        with cls._lock:
            return {
                'cached_directories': list(cls._cache.keys()),
                'total_teams_cached': sum(len(teams) for teams in cls._cache.values()),
                'cache_stats': cls._cache_stats.copy()
            }
    
    @classmethod
    def clear_cache(cls, teams_dir: Optional[str] = None) -> None:
        """Clear cache for a specific directory or all directories.
        
        Parameters
        ----------
        teams_dir : Optional[str]
            Specific directory to clear, or None to clear all
        """
        with cls._lock:
            if teams_dir:
                teams_dir_str = str(Path(teams_dir).resolve())
                cls._cache.pop(teams_dir_str, None)
                cls._cache_stats.pop(teams_dir_str, None)
                logger.info("Cleared cache for: %s", teams_dir_str)
            else:
                cls._cache.clear()
                cls._cache_stats.clear()
                logger.info("Cleared all team cache")
    
    @classmethod
    def print_performance_report(cls) -> None:
        """Print a detailed performance report of team loading."""
        cache_info = cls.get_cache_info()
        
        if not cache_info['cache_stats']:
            logger.info("No team cache statistics available")
            return
        
        print("\n" + "=" * 80)
        print("TEAM CACHE PERFORMANCE REPORT")
        print("=" * 80)
        
        total_teams = 0
        total_load_time = 0
        total_cache_hits = 0
        
        for dir_path, stats in cache_info['cache_stats'].items():
            print(f"\nDirectory: {dir_path}")
            print(f"  Teams loaded: {stats['team_count']}")
            print(f"  Initial load time: {stats['total_load_time']:.3f}s")
            print(f"  Average per team: {stats['avg_load_time_per_team']:.3f}s")
            print(f"  Cache hits: {stats['cache_hits']}")
            print(f"  Load timestamp: {stats['initial_load_time']}")
            
            if 'io_time' in stats:
                print(f"  I/O time: {stats['io_time']:.3f}s")
                print(f"  Parse time: {stats['parse_time']:.3f}s")
                print(f"  Total size: {stats['total_size']:,} bytes")
                print(f"  Average file size: {stats['avg_file_size']:.0f} bytes")
            
            total_teams += stats['team_count']
            total_load_time += stats['total_load_time']
            total_cache_hits += stats['cache_hits']
        
        print(f"\nSUMMARY:")
        print(f"  Total directories cached: {len(cache_info['cache_stats'])}")
        print(f"  Total teams cached: {total_teams}")
        print(f"  Total initial load time: {total_load_time:.3f}s")
        print(f"  Total cache hits: {total_cache_hits}")
        
        if total_teams > 0:
            print(f"  Average load time per team: {total_load_time / total_teams:.3f}s")
        
        # Calculate performance improvement
        if total_cache_hits > 0:
            estimated_time_saved = total_cache_hits * (total_load_time / max(1, total_teams))
            print(f"  Estimated time saved by caching: {estimated_time_saved:.3f}s")
        
        print("=" * 80 + "\n")


__all__ = ["TeamCacheManager"]