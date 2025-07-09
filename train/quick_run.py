#!/usr/bin/env python3
"""Quick training script for network architecture benchmarking."""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
import yaml

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from train_selfplay import main as train_main


def setup_logging():
    """Set up logging for quick runs."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def create_quick_config(steps: int, config_path: str, network_type: str = "basic",
                       use_2layer: bool = True, use_lstm: bool = False,
                       use_attention: bool = False) -> dict:
    """Create configuration for quick benchmark runs."""
    
    # Load base config
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        # Default config if file not found
        config = {
            'episodes': 100,
            'lr': 0.002,
            'batch_size': 512,
            'buffer_capacity': 1000,
            'gamma': 0.997,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'value_coef': 0.6,
            'entropy_coef': 0.02,
            'ppo_epochs': 4,
            'algorithm': 'ppo',
            'reward': 'composite',
            'reward_config': 'config/reward.yaml'
        }
    
    # Override episodes based on steps (approximate conversion)
    config['episodes'] = max(10, steps // 100)
    
    # Set network configuration
    config['network'] = {
        'type': network_type,
        'hidden_size': 128,
        'use_2layer': use_2layer,
        'use_lstm': use_lstm,
        'use_attention': use_attention,
        'lstm_hidden_size': 128,
        'attention_heads': 4,
        'attention_dropout': 0.1
    }
    
    return config


def run_benchmark(config: dict, run_name: str) -> dict:
    """Run a single benchmark with the given configuration."""
    
    print(f"\n=== Running benchmark: {run_name} ===")
    print(f"Network config: {config['network']}")
    
    # Create temporary config file
    config_path = f"/tmp/quick_config_{run_name}.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Run training
    start_time = time.time()
    
    try:
        # Simulate command line arguments
        sys.argv = [
            'train_selfplay.py',
            '--config', config_path,
            '--episodes', str(config['episodes']),
            '--checkpoint-dir', f'checkpoints/quick_{run_name}',
            '--log-dir', f'logs/quick_{run_name}',
            '--disable-tensorboard'  # Disable TB for quick runs
        ]
        
        train_main()
        
    except Exception as e:
        print(f"Error in benchmark {run_name}: {e}")
        return {"error": str(e), "duration": time.time() - start_time}
    
    duration = time.time() - start_time
    
    # Clean up
    Path(config_path).unlink(missing_ok=True)
    
    return {
        "duration": duration,
        "episodes": config['episodes'],
        "network": config['network']
    }


def main():
    """Main function for quick benchmarking."""
    parser = argparse.ArgumentParser(description='Quick network architecture benchmarking')
    parser.add_argument('--steps', type=int, default=10000,
                       help='Number of training steps (default: 10000)')
    parser.add_argument('--config', type=str, default='config/m7.yaml',
                       help='Base configuration file (default: config/m7.yaml)')
    parser.add_argument('--networks', type=str, nargs='+',
                       default=['basic_1layer', 'basic_2layer', 'lstm', 'attention'],
                       help='Network types to benchmark')
    parser.add_argument('--output', type=str, default='benchmark_results.yaml',
                       help='Output file for results (default: benchmark_results.yaml)')
    
    args = parser.parse_args()
    
    setup_logging()
    
    # Define network configurations
    network_configs = {
        'basic_1layer': {
            'type': 'basic',
            'use_2layer': False,
            'use_lstm': False,
            'use_attention': False
        },
        'basic_2layer': {
            'type': 'basic',
            'use_2layer': True,
            'use_lstm': False,
            'use_attention': False
        },
        'lstm': {
            'type': 'lstm',
            'use_2layer': True,
            'use_lstm': True,
            'use_attention': False
        },
        'attention': {
            'type': 'attention',
            'use_2layer': True,
            'use_lstm': False,
            'use_attention': True
        },
        'lstm_attention': {
            'type': 'attention',
            'use_2layer': True,
            'use_lstm': True,
            'use_attention': True
        }
    }
    
    results = {}
    
    for network_name in args.networks:
        if network_name not in network_configs:
            print(f"Unknown network type: {network_name}")
            continue
            
        net_config = network_configs[network_name]
        config = create_quick_config(
            steps=args.steps,
            config_path=args.config,
            network_type=net_config['type'],
            use_2layer=net_config['use_2layer'],
            use_lstm=net_config['use_lstm'],
            use_attention=net_config['use_attention']
        )
        
        result = run_benchmark(config, network_name)
        results[network_name] = result
        
        print(f"Completed {network_name}: {result}")
    
    # Save results
    with open(args.output, 'w') as f:
        yaml.dump(results, f, default_flow_style=False)
    
    print(f"\n=== Benchmark Results ===")
    print(f"Results saved to: {args.output}")
    
    # Print summary
    for name, result in results.items():
        if 'error' in result:
            print(f"{name}: ERROR - {result['error']}")
        else:
            print(f"{name}: {result['duration']:.2f}s ({result['episodes']} episodes)")


if __name__ == '__main__':
    main()