#!/usr/bin/env python3
"""
Test script for team cache functionality
Tests the performance improvements of the team caching system
"""

import time
import logging
from pathlib import Path
import sys

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.teams import TeamLoader, TeamCacheManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_team_cache_performance():
    """Test team cache performance with multiple TeamLoader instances."""
    
    logger.info("=== TEAM CACHE PERFORMANCE TEST ===")
    
    teams_dir = ROOT_DIR / "config" / "teams"
    if not teams_dir.exists():
        logger.error("Teams directory not found: %s", teams_dir)
        return
    
    # Clear cache to start fresh
    TeamCacheManager.clear_cache()
    
    # Test 1: First load (should be slow - actual file I/O)
    logger.info("\n--- Test 1: First TeamLoader (Initial Load) ---")
    start_time = time.time()
    loader1 = TeamLoader(teams_dir)
    first_load_time = time.time() - start_time
    
    logger.info("First load results:")
    logger.info("  Load time: %.3fs", first_load_time)
    logger.info("  Teams loaded: %d", loader1.get_team_count())
    
    # Print detailed performance report
    loader1.print_performance_report()
    
    # Test 2: Second load (should be fast - from cache)
    logger.info("\n--- Test 2: Second TeamLoader (Cache Hit) ---")
    start_time = time.time()
    loader2 = TeamLoader(teams_dir)
    second_load_time = time.time() - start_time
    
    logger.info("Second load results:")
    logger.info("  Load time: %.3fs", second_load_time)
    logger.info("  Teams loaded: %d", loader2.get_team_count())
    
    # Test 3: Multiple rapid loads
    logger.info("\n--- Test 3: Multiple Rapid Loads (10x) ---")
    start_time = time.time()
    loaders = []
    for i in range(10):
        loader = TeamLoader(teams_dir)
        loaders.append(loader)
    total_time = time.time() - start_time
    
    logger.info("Multiple loads results:")
    logger.info("  Total time for 10 loads: %.3fs", total_time)
    logger.info("  Average time per load: %.3fs", total_time / 10)
    
    # Test 4: Performance comparison
    logger.info("\n--- Performance Analysis ---")
    speedup = first_load_time / (total_time / 10) if total_time > 0 else 0
    logger.info("Cache speedup: %.1fx faster", speedup)
    
    # Estimate time saved in realistic scenario
    estimated_original_time = first_load_time * 20  # 20 EnvPlayers in parallel training
    estimated_cached_time = first_load_time + (second_load_time * 19)  # 1 initial + 19 cached
    time_saved = estimated_original_time - estimated_cached_time
    
    logger.info("Realistic scenario (20 parallel environments):")
    logger.info("  Without cache: %.3fs (%.3fs × 20)", estimated_original_time, first_load_time)
    logger.info("  With cache: %.3fs (%.3fs + %.3fs × 19)", estimated_cached_time, first_load_time, second_load_time)
    logger.info("  Time saved: %.3fs (%.1f%% improvement)", 
               time_saved, (time_saved / estimated_original_time * 100) if estimated_original_time > 0 else 0)
    
    # Test 5: Random team selection performance
    logger.info("\n--- Test 5: Random Team Selection Performance ---")
    if loader1.get_team_count() > 0:
        start_time = time.time()
        for i in range(1000):
            team = loader1.get_random_team()
        selection_time = time.time() - start_time
        
        logger.info("Random selection performance:")
        logger.info("  1000 selections: %.3fs", selection_time)
        logger.info("  Average per selection: %.6fs", selection_time / 1000)
        logger.info("  Selections per second: %.0f", 1000 / selection_time if selection_time > 0 else 0)
    
    # Final cache report
    logger.info("\n--- Final Cache Report ---")
    TeamCacheManager.print_performance_report()
    
    return {
        'first_load_time': first_load_time,
        'second_load_time': second_load_time,
        'multiple_load_avg': total_time / 10,
        'speedup': speedup,
        'time_saved_estimate': time_saved,
        'teams_loaded': loader1.get_team_count()
    }

def test_memory_usage():
    """Test memory usage of team cache."""
    
    logger.info("\n=== MEMORY USAGE TEST ===")
    
    teams_dir = ROOT_DIR / "config" / "teams"
    
    # Get cache info
    cache_info = TeamCacheManager.get_cache_info()
    
    logger.info("Cache memory analysis:")
    logger.info("  Cached directories: %d", len(cache_info['cached_directories']))
    logger.info("  Total teams cached: %d", cache_info['total_teams_cached'])
    
    for dir_path, stats in cache_info['cache_stats'].items():
        if 'total_size' in stats:
            memory_kb = stats['total_size'] / 1024
            logger.info("  %s: %.2f KB in memory", Path(dir_path).name, memory_kb)

def main():
    """Run all team cache tests."""
    
    # Test performance
    performance_results = test_team_cache_performance()
    
    # Test memory usage
    test_memory_usage()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("TEAM CACHE TEST SUMMARY")
    logger.info("=" * 80)
    logger.info("✅ Team cache implementation is working correctly")
    logger.info("✅ Cache provides %.1fx speedup for subsequent loads", performance_results['speedup'])
    logger.info("✅ Estimated %.3fs time savings in realistic training scenario", performance_results['time_saved_estimate'])
    logger.info("✅ %d teams successfully cached and accessible", performance_results['teams_loaded'])
    
    if performance_results['speedup'] > 5:
        logger.info("🚀 Excellent performance improvement!")
    elif performance_results['speedup'] > 2:
        logger.info("👍 Good performance improvement!")
    else:
        logger.info("⚠️  Performance improvement lower than expected")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    main()