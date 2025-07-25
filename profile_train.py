#!/usr/bin/env python3
"""
Train.py performance profiling script
Analyzes bottlenecks in the training process
"""

import cProfile
import pstats
import time
import subprocess
import sys
from pathlib import Path
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def profile_train_execution(episodes=1, parallel=2, config_path="config/train_config.yml"):
    """Profile train.py execution with cProfile"""
    
    logger.info("Starting profiling of train.py execution...")
    logger.info(f"Parameters: episodes={episodes}, parallel={parallel}, config={config_path}")
    
    # Create profile output directory
    profile_dir = Path("profile_results")
    profile_dir.mkdir(exist_ok=True)
    
    # Profile file path
    profile_file = profile_dir / f"train_profile_{int(time.time())}.prof"
    
    # Prepare command
    cmd = [
        sys.executable, "-m", "cProfile", "-o", str(profile_file),
        "train.py",
        "--episodes", str(episodes),
        "--parallel", str(parallel),
        "--config", config_path,
        "--tensorboard"  # Enable for full profiling
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Execute profiling
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout
        execution_time = time.time() - start_time
        
        logger.info(f"Training completed in {execution_time:.2f} seconds")
        
        if result.returncode != 0:
            logger.error(f"Training failed with return code {result.returncode}")
            logger.error(f"STDERR: {result.stderr}")
            return None
            
        logger.info("Profiling completed successfully")
        return profile_file
        
    except subprocess.TimeoutExpired:
        logger.error("Training timed out after 5 minutes")
        return None
    except Exception as e:
        logger.error(f"Profiling failed with error: {e}")
        return None

def analyze_profile(profile_file):
    """Analyze the generated profile file"""
    
    if not profile_file or not Path(profile_file).exists():
        logger.error("Profile file not found")
        return
    
    logger.info(f"Analyzing profile: {profile_file}")
    
    # Load and analyze profile
    stats = pstats.Stats(str(profile_file))
    
    # Create analysis report
    report_file = Path(profile_file).with_suffix('.txt')
    
    with open(report_file, 'w') as f:
        # Original stdout
        original_stdout = sys.stdout
        sys.stdout = f
        
        print("=" * 80)
        print("TRAIN.PY PERFORMANCE PROFILE ANALYSIS")
        print("=" * 80)
        print()
        
        # Top 20 functions by cumulative time
        print("TOP 20 FUNCTIONS BY CUMULATIVE TIME:")
        print("-" * 50)
        stats.sort_stats('cumulative').print_stats(20)
        print()
        
        # Top 20 functions by total time
        print("TOP 20 FUNCTIONS BY TOTAL TIME:")
        print("-" * 50)
        stats.sort_stats('time').print_stats(20)
        print()
        
        # Functions with most calls
        print("TOP 20 FUNCTIONS BY CALL COUNT:")
        print("-" * 50)
        stats.sort_stats('calls').print_stats(20)
        print()
        
        # Pokemon-specific bottlenecks
        print("POKEMON ENVIRONMENT SPECIFIC FUNCTIONS:")
        print("-" * 50)
        stats.print_stats('pokemon.*', 20)
        print()
        
        # Network/algorithm bottlenecks
        print("NEURAL NETWORK AND ALGORITHM FUNCTIONS:")
        print("-" * 50)
        stats.print_stats('(network|algorithm|torch).*', 20)
        print()
        
        # State observation bottlenecks
        print("STATE OBSERVATION FUNCTIONS:")
        print("-" * 50)
        stats.print_stats('state.*', 20)
        print()
        
        # Restore stdout
        sys.stdout = original_stdout
    
    logger.info(f"Analysis report saved to: {report_file}")
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("PROFILE ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Get top bottlenecks
    stats.sort_stats('cumulative')
    stats.print_stats(10)

def benchmark_components():
    """Benchmark individual components"""
    
    logger.info("Running component benchmarks...")
    
    # Test imports
    start_time = time.time()
    try:
        from src.env.pokemon_env import PokemonEnv
        from src.state.state_observer import StateObserver
        import_time = time.time() - start_time
        logger.info(f"Import time: {import_time:.3f}s")
    except Exception as e:
        logger.error(f"Import failed: {e}")
        return
    
    # Test environment initialization
    start_time = time.time()
    try:
        observer = StateObserver("config/state_spec.yml")
        env = PokemonEnv(
            opponent_player=None,
            state_observer=observer,
            action_helper=None,
            reward="composite",
        )
        init_time = time.time() - start_time
        logger.info(f"Environment initialization time: {init_time:.3f}s")
    except Exception as e:
        logger.error(f"Environment initialization failed: {e}")
        return
    
    # Test state observation
    start_time = time.time()
    try:
        # This would require a battle instance
        obs_time = time.time() - start_time
        logger.info(f"State observation time: {obs_time:.3f}s")
    except Exception as e:
        logger.error(f"State observation failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Profile train.py performance")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to profile")
    parser.add_argument("--parallel", type=int, default=2, help="Number of parallel environments")
    parser.add_argument("--config", default="config/train_config.yml", help="Config file path")
    parser.add_argument("--benchmark-only", action="store_true", help="Only run component benchmarks")
    
    args = parser.parse_args()
    
    if args.benchmark_only:
        benchmark_components()
        return
    
    # Run full profiling
    profile_file = profile_train_execution(args.episodes, args.parallel, args.config)
    
    if profile_file:
        analyze_profile(profile_file)
        
        # Run component benchmarks as well
        benchmark_components()
        
        print(f"\nProfiling complete! Check {profile_file} and corresponding .txt file for detailed analysis.")
    else:
        logger.error("Profiling failed")

if __name__ == "__main__":
    main()