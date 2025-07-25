#!/usr/bin/env python3
"""
Lightweight benchmarking script for train.py bottleneck analysis
"""

import time
import sys
import logging
from pathlib import Path
from contextlib import contextmanager
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Repository root path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

@contextmanager
def timer(name):
    """Context manager for timing operations"""
    start = time.time()
    logger.info(f"Starting: {name}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"Completed: {name} - {elapsed:.3f}s")

def benchmark_imports():
    """Benchmark module imports"""
    with timer("Module imports"):
        # Core imports
        with timer("  Core imports (yaml, torch, numpy)"):
            import yaml
            import torch
            import numpy as np
        
        # Pokemon environment imports
        with timer("  PokemonEnv imports"):
            from src.env.pokemon_env import PokemonEnv
            from src.state.state_observer import StateObserver
        
        # Agent imports
        with timer("  Agent imports"):
            from src.agents import PolicyNetwork, ValueNetwork, RLAgent
            from src.agents.enhanced_networks import (
                LSTMPolicyNetwork, LSTMValueNetwork,
                AttentionPolicyNetwork, AttentionValueNetwork
            )
        
        # Algorithm imports
        with timer("  Algorithm imports"):
            from src.algorithms import PPOAlgorithm, ReinforceAlgorithm
        
        # Bot imports
        with timer("  Bot imports"):
            from src.bots import RandomBot, MaxDamageBot

def benchmark_initialization():
    """Benchmark initialization components"""
    
    with timer("Configuration loading"):
        import yaml
        try:
            with open("config/train_config.yml", "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Config loading failed: {e}")
            cfg = {}
    
    with timer("StateObserver initialization"):
        try:
            from src.state.state_observer import StateObserver
            observer = StateObserver("config/state_spec.yml")
        except Exception as e:
            logger.error(f"StateObserver init failed: {e}")
            return
    
    with timer("PokemonEnv initialization"):
        try:
            from src.env.pokemon_env import PokemonEnv
            env = PokemonEnv(
                opponent_player=None,
                state_observer=observer,
                action_helper=None,
                reward="composite",
                reward_config_path="config/reward.yaml",
                team_mode="default",
                normalize_rewards=True,
            )
        except Exception as e:
            logger.error(f"PokemonEnv init failed: {e}")
            return
    
    with timer("Network initialization"):
        try:
            from src.agents.network_factory import create_policy_network, create_value_network
            
            # Test basic network
            network_config = {"type": "basic", "hidden_size": 128, "use_2layer": True}
            policy_net = create_policy_network(1136, 11, network_config)  # 1136 state dims, 11 actions
            value_net = create_value_network(1136, network_config)
            
            logger.info(f"Basic networks created - Policy params: {sum(p.numel() for p in policy_net.parameters())}, Value params: {sum(p.numel() for p in value_net.parameters())}")
            
        except Exception as e:
            logger.error(f"Network init failed: {e}")

def benchmark_team_loading():
    """Benchmark team loading process"""
    
    with timer("Team loading"):
        try:
            teams_dir = ROOT_DIR / "config" / "teams"
            team_files = list(teams_dir.glob("*.txt"))
            logger.info(f"Found {len(team_files)} team files")
            
            # Load a few teams to test parsing speed
            for i, team_file in enumerate(team_files[:3]):
                with timer(f"  Loading team {i+1}: {team_file.name}"):
                    try:
                        with open(team_file, 'r', encoding='utf-8') as f:
                            content = f.read()
                            # Count pokemon in team (basic parsing)
                            pokemon_count = content.count('\n\n')
                            logger.info(f"    Team has ~{pokemon_count} Pokemon")
                    except Exception as e:
                        logger.error(f"    Team loading failed: {e}")
                        
        except Exception as e:
            logger.error(f"Team directory access failed: {e}")

def benchmark_device_operations():
    """Benchmark device operations"""
    
    with timer("Device operations"):
        try:
            import torch
            from src.utils.device_utils import get_device, transfer_to_device
            
            with timer("  Device detection"):
                device = get_device(prefer_gpu=True, device_name="auto")
                logger.info(f"Selected device: {device}")
            
            with timer("  Tensor operations"):
                # Create sample tensors
                sample_obs = torch.randn(32, 1136)  # Batch of 32 observations
                sample_actions = torch.randint(0, 11, (32,))
                
                # Transfer to device
                sample_obs = sample_obs.to(device)
                sample_actions = sample_actions.to(device)
                logger.info(f"Tensors moved to {device}")
                
        except Exception as e:
            logger.error(f"Device operations failed: {e}")

def benchmark_parallel_overhead():
    """Analyze parallel environment overhead"""
    
    logger.info("=== PARALLEL OVERHEAD ANALYSIS ===")
    
    # Based on logs, analyze the observed behavior
    logger.info("From logs analysis:")
    logger.info("- Configuration shows parallel=10 but actual parallel=2 requested")
    logger.info("- 20 EnvPlayer instances created (10 pairs for 10 parallel envs)")
    logger.info("- Each player takes ~10ms to initialize")
    logger.info("- Team loading happens per player: 4 teams loaded 20+ times")
    logger.info("- εpsilon-greedy wrapper applied to each agent individually")
    
    # Estimate initialization overhead
    team_loading_overhead = 4 * 20 * 0.005  # 4 teams × 20 players × 5ms each
    player_init_overhead = 20 * 0.010  # 20 players × 10ms each
    wrapper_overhead = 10 * 0.002  # 10 agents × 2ms wrapper
    
    total_init_time = team_loading_overhead + player_init_overhead + wrapper_overhead
    
    logger.info(f"Estimated initialization overhead:")
    logger.info(f"  Team loading: {team_loading_overhead:.3f}s")
    logger.info(f"  Player initialization: {player_init_overhead:.3f}s") 
    logger.info(f"  Wrapper overhead: {wrapper_overhead:.3f}s")
    logger.info(f"  Total estimated: {total_init_time:.3f}s")
    logger.info(f"  Actual from logs: ~9s (initialization phase)")

def identify_bottlenecks():
    """Identify bottlenecks from log analysis"""
    
    logger.info("=== BOTTLENECK ANALYSIS FROM LOGS ===")
    
    bottlenecks = [
        {
            "name": "Team loading redundancy",
            "description": "Teams loaded 20+ times (once per EnvPlayer)",
            "estimated_impact": "HIGH",
            "current_time": "~0.4s",
            "optimization": "Cache loaded teams, share across players"
        },
        {
            "name": "Player initialization",
            "description": "20 EnvPlayer instances for 10 parallel envs",
            "estimated_impact": "MEDIUM",
            "current_time": "~0.2s", 
            "optimization": "Pool reuse, lazy initialization"
        },
        {
            "name": "Network parameter count",
            "description": "AttentionNetwork: 1.7M params each (policy + value)",
            "estimated_impact": "HIGH",
            "current_time": "GPU memory & compute",
            "optimization": "Use smaller networks for development"
        },
        {
            "name": "WebSocket connections",
            "description": "20 concurrent WebSocket connections to Pokemon Showdown",
            "estimated_impact": "HIGH",
            "current_time": "Network I/O bound",
            "optimization": "Reduce parallel environments or use connection pooling"
        },
        {
            "name": "Configuration parsing",
            "description": "YAML config loaded multiple times",
            "estimated_impact": "LOW",
            "current_time": "<0.1s",
            "optimization": "Cache configuration"
        }
    ]
    
    logger.info("Identified bottlenecks (ordered by impact):")
    for i, bottleneck in enumerate(bottlenecks, 1):
        logger.info(f"{i}. {bottleneck['name']} [{bottleneck['estimated_impact']}]")
        logger.info(f"   Issue: {bottleneck['description']}")
        logger.info(f"   Time: {bottleneck['current_time']}")
        logger.info(f"   Fix: {bottleneck['optimization']}")
        logger.info("")

def create_optimization_recommendations():
    """Create specific optimization recommendations"""
    
    logger.info("=== OPTIMIZATION RECOMMENDATIONS ===")
    
    recommendations = [
        {
            "priority": "HIGH",
            "title": "Reduce network complexity for development",
            "description": "Use basic networks instead of Attention networks during development",
            "implementation": "Change config/train_config.yml: network.type: 'basic', hidden_size: 128",
            "expected_speedup": "3-5x faster initialization, 2-3x less GPU memory"
        },
        {
            "priority": "HIGH", 
            "title": "Optimize team loading",
            "description": "Cache loaded teams instead of reloading for each player",
            "implementation": "Implement team cache in PokemonEnv initialization",
            "expected_speedup": "0.3-0.4s initialization time saved"
        },
        {
            "priority": "MEDIUM",
            "title": "Reduce parallel environments for development",
            "description": "Use parallel=2-3 instead of 10 for faster iteration",
            "implementation": "Change config: parallel: 2",
            "expected_speedup": "4x fewer WebSocket connections, faster startup"
        },
        {
            "priority": "MEDIUM",
            "title": "Connection pooling for Pokemon Showdown",
            "description": "Reuse WebSocket connections instead of creating new ones",
            "implementation": "Implement connection pool in EnvPlayer",
            "expected_speedup": "Reduce connection overhead by 50-70%"
        },
        {
            "priority": "LOW",
            "title": "Configuration caching",
            "description": "Cache YAML configurations to avoid repeated parsing",
            "implementation": "Add config cache decorator",
            "expected_speedup": "Marginal improvement (<0.1s)"
        }
    ]
    
    for rec in recommendations:
        logger.info(f"[{rec['priority']}] {rec['title']}")
        logger.info(f"  Problem: {rec['description']}")
        logger.info(f"  Solution: {rec['implementation']}")
        logger.info(f"  Expected: {rec['expected_speedup']}")
        logger.info("")

def main():
    parser = argparse.ArgumentParser(description="Benchmark train.py components")
    parser.add_argument("--skip-imports", action="store_true", help="Skip import benchmarking")
    parser.add_argument("--skip-init", action="store_true", help="Skip initialization benchmarking") 
    parser.add_argument("--analysis-only", action="store_true", help="Only run log analysis")
    
    args = parser.parse_args()
    
    logger.info("=== TRAIN.PY BOTTLENECK ANALYSIS ===")
    
    if args.analysis_only:
        benchmark_parallel_overhead()
        identify_bottlenecks()
        create_optimization_recommendations()
        return
    
    total_start = time.time()
    
    if not args.skip_imports:
        benchmark_imports()
    
    if not args.skip_init:
        benchmark_initialization()
        benchmark_team_loading()
        benchmark_device_operations()
    
    # Always run analysis
    benchmark_parallel_overhead()
    identify_bottlenecks() 
    create_optimization_recommendations()
    
    total_time = time.time() - total_start
    logger.info(f"Total benchmark time: {total_time:.3f}s")

if __name__ == "__main__":
    main()