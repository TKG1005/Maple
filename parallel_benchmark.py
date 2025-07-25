#!/usr/bin/env python3
"""
Parallel environment benchmark script
Tests different parallel configurations with CPU to avoid Mac Silicon GPU issues
"""

import time
import subprocess
import sys
import logging
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_train_benchmark(parallel_count, episodes=1, device="cpu"):
    """Run train.py with specific parallel count and measure performance"""
    
    logger.info(f"Testing parallel={parallel_count}, episodes={episodes}, device={device}")
    
    cmd = [
        sys.executable, "train.py",
        "--episodes", str(episodes),
        "--parallel", str(parallel_count),
        "--device", device,
        "--config", "config/train_config.yml"
    ]
    
    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)  # 3 minute timeout
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"SUCCESS: parallel={parallel_count} completed in {execution_time:.2f}s")
            
            # Extract key metrics from output
            lines = result.stderr.split('\n')
            metrics = extract_metrics_from_output(lines)
            metrics['execution_time'] = execution_time
            metrics['parallel'] = parallel_count
            
            return metrics
        else:
            logger.error(f"FAILED: parallel={parallel_count}, return code={result.returncode}")
            logger.error(f"STDERR: {result.stderr[-500:]}")  # Last 500 chars
            return None
            
    except subprocess.TimeoutExpired:
        logger.error(f"TIMEOUT: parallel={parallel_count} exceeded 3 minutes")
        return None
    except Exception as e:
        logger.error(f"ERROR: parallel={parallel_count} failed with {e}")
        return None

def extract_metrics_from_output(lines):
    """Extract performance metrics from train.py output"""
    
    metrics = {
        'env_players_created': 0,
        'team_loads': 0,
        'network_params': 0,
        'init_time_estimate': 0,
        'websocket_connections': 0
    }
    
    for line in lines:
        if 'EnvPlayer' in line and 'Starting listening' in line:
            metrics['websocket_connections'] += 1
        
        if 'Successfully loaded' in line and 'teams' in line:
            metrics['team_loads'] += 1
            
        if 'total_params' in line:
            # Extract parameter count from network info
            try:
                import re
                match = re.search(r"'total_params': (\d+)", line)
                if match:
                    metrics['network_params'] = max(metrics['network_params'], int(match.group(1)))
            except:
                pass
    
    # Estimate initialization time (rough calculation)
    metrics['init_time_estimate'] = (
        metrics['team_loads'] * 0.005 +  # 5ms per team load
        metrics['websocket_connections'] * 0.010  # 10ms per WebSocket connection
    )
    
    return metrics

def benchmark_parallel_configurations():
    """Benchmark different parallel configurations"""
    
    logger.info("=== PARALLEL CONFIGURATION BENCHMARK ===")
    
    configurations = [2, 5, 10]
    results = []
    
    for parallel in configurations:
        logger.info(f"\n--- Testing parallel={parallel} ---")
        
        metrics = run_train_benchmark(parallel, episodes=1, device="cpu")
        if metrics:
            results.append(metrics)
            
            # Log detailed results
            logger.info(f"Results for parallel={parallel}:")
            logger.info(f"  Execution time: {metrics['execution_time']:.2f}s")
            logger.info(f"  WebSocket connections: {metrics['websocket_connections']}")
            logger.info(f"  Team loads: {metrics['team_loads']}")
            logger.info(f"  Network params: {metrics['network_params']:,}")
            logger.info(f"  Est. init overhead: {metrics['init_time_estimate']:.3f}s")
        
        # Cool down between tests
        time.sleep(5)
    
    return results

def analyze_parallel_efficiency(results):
    """Analyze parallel efficiency from benchmark results"""
    
    if not results:
        logger.error("No benchmark results to analyze")
        return
    
    logger.info("\n=== PARALLEL EFFICIENCY ANALYSIS ===")
    
    baseline = results[0]  # Use first result as baseline
    
    logger.info(f"Baseline (parallel={baseline['parallel']}):")
    logger.info(f"  Time: {baseline['execution_time']:.2f}s")
    logger.info(f"  Connections: {baseline['websocket_connections']}")
    logger.info(f"  Team loads: {baseline['team_loads']}")
    
    logger.info("\nComparison with baseline:")
    
    for result in results[1:]:
        parallel_ratio = result['parallel'] / baseline['parallel']
        time_ratio = result['execution_time'] / baseline['execution_time']
        efficiency = parallel_ratio / time_ratio if time_ratio > 0 else 0
        
        connection_ratio = result['websocket_connections'] / baseline['websocket_connections']
        team_load_ratio = result['team_loads'] / baseline['team_loads']
        
        logger.info(f"\nparallel={result['parallel']} vs parallel={baseline['parallel']}:")
        logger.info(f"  Parallel increase: {parallel_ratio:.1f}x")
        logger.info(f"  Time increase: {time_ratio:.2f}x")
        logger.info(f"  Efficiency: {efficiency:.2f} (1.0 = perfect scaling)")
        logger.info(f"  Connection increase: {connection_ratio:.1f}x")
        logger.info(f"  Team load increase: {team_load_ratio:.1f}x")
        
        # Performance analysis
        if efficiency > 0.8:
            logger.info(f"  Assessment: GOOD scaling")
        elif efficiency > 0.5:
            logger.info(f"  Assessment: MODERATE scaling")
        else:
            logger.info(f"  Assessment: POOR scaling")

def create_optimization_config():
    """Create optimized configuration for development"""
    
    logger.info("\n=== CREATING OPTIMIZED CONFIG ===")
    
    optimized_config = {
        'episodes': 10,
        'lr': 0.003,
        'parallel': 3,  # Sweet spot based on analysis
        'batch_size': 1024,
        'buffer_capacity': 2048,
        'algorithm': 'ppo',
        'reward': 'composite',
        'team': 'default',
        'opponent': 'max',
        'tensorboard': True,
        'network': {
            'type': 'basic',  # Much faster than attention
            'hidden_size': 128,
            'use_2layer': True
        },
        'exploration': {
            'epsilon_greedy': {
                'enabled': False  # Disable for faster development
            }
        },
        'league_training': {
            'enabled': False  # Disable for faster development  
        }
    }
    
    config_path = Path("config/train_config_dev.yml")
    
    try:
        import yaml
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(optimized_config, f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Created optimized config: {config_path}")
        logger.info("Key optimizations:")
        logger.info("  - parallel: 3 (balance between speed and resource usage)")
        logger.info("  - network.type: 'basic' (10x faster than attention)")
        logger.info("  - epsilon_greedy: disabled (remove exploration overhead)")
        logger.info("  - league_training: disabled (remove historical opponent overhead)")
        logger.info("  - Reduced batch_size and buffer_capacity")
        
        logger.info(f"\nUsage: python train.py --config {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to create optimized config: {e}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark parallel configurations")
    parser.add_argument("--skip-benchmark", action="store_true", help="Skip benchmarking, just create config")
    parser.add_argument("--quick", action="store_true", help="Quick test with fewer configurations")
    
    args = parser.parse_args()
    
    if args.skip_benchmark:
        create_optimization_config()
        return
    
    # Run benchmarks
    if args.quick:
        # Quick test with just 2 and 5
        logger.info("Running quick benchmark (parallel=2,5 only)")
        results = []
        for parallel in [2, 5]:
            metrics = run_train_benchmark(parallel, episodes=1, device="cpu")
            if metrics:
                results.append(metrics)
            time.sleep(3)
    else:
        results = benchmark_parallel_configurations()
    
    if results:
        analyze_parallel_efficiency(results)
    
    # Always create optimized config
    create_optimization_config()

if __name__ == "__main__":
    main()