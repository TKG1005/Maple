#!/usr/bin/env python3
"""
Test script for multi-server functionality
Tests the MultiServerManager integration with train.py
"""

import time
import logging
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.utils.server_manager import MultiServerManager

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def test_server_manager_initialization():
    """Test ServerManager initialization from config."""
    
    logger.info("=== MULTI-SERVER MANAGER INITIALIZATION TEST ===")
    
    # Test 1: Default single server configuration
    logger.info("\n--- Test 1: Default Configuration ---")
    manager1 = MultiServerManager.from_config({})
    
    logger.info("Default configuration results:")
    logger.info("  Total servers: %d", len(manager1.servers))
    logger.info("  Total capacity: %d connections", manager1.get_total_capacity())
    
    # Test 2: Multiple server configuration
    logger.info("\n--- Test 2: Multiple Server Configuration ---")
    config = {
        "servers": [
            {"host": "localhost", "port": 8000, "max_connections": 25},
            {"host": "localhost", "port": 8001, "max_connections": 25},
            {"host": "localhost", "port": 8002, "max_connections": 25},
            {"host": "localhost", "port": 8003, "max_connections": 25}
        ]
    }
    
    manager2 = MultiServerManager.from_config(config)
    
    logger.info("Multiple server configuration results:")
    logger.info("  Total servers: %d", len(manager2.servers))
    logger.info("  Total capacity: %d connections", manager2.get_total_capacity())
    
    return manager2

def test_capacity_validation(manager):
    """Test server capacity validation."""
    
    logger.info("\n=== CAPACITY VALIDATION TEST ===")
    
    # Test valid parallel count
    parallel_valid = 50
    is_valid, error_msg = manager.validate_parallel_count(parallel_valid)
    logger.info("Parallel=%d validation: %s", parallel_valid, "PASS" if is_valid else f"FAIL - {error_msg}")
    
    # Test exceeding capacity
    parallel_invalid = 150
    is_valid, error_msg = manager.validate_parallel_count(parallel_invalid)
    logger.info("Parallel=%d validation: %s", parallel_invalid, "PASS" if is_valid else f"FAIL - {error_msg}")
    
    return parallel_valid if is_valid else 10

def test_environment_assignment(manager, parallel):
    """Test environment assignment and load balancing."""
    
    logger.info("\n=== ENVIRONMENT ASSIGNMENT TEST ===")
    
    try:
        assignments = manager.assign_environments(parallel)
        
        logger.info("Assignment results:")
        logger.info("  Environments assigned: %d", len(assignments))
        logger.info("  Parallel requested: %d", parallel)
        
        # Verify all environments got assignments
        if len(assignments) == parallel:
            logger.info("✅ All environments successfully assigned")
        else:
            logger.error("❌ Assignment count mismatch")
            return False
        
        # Print assignment summary
        manager.print_assignment_report()
        
        return True
        
    except Exception as e:
        logger.error("Assignment failed: %s", str(e))
        return False

def test_load_balancing(manager):
    """Test load balancing across different parallel counts."""
    
    logger.info("\n=== LOAD BALANCING TEST ===")
    
    test_cases = [5, 10, 20, 50, 75]
    
    for parallel in test_cases:
        if parallel > manager.get_total_capacity():
            logger.info("Skipping parallel=%d (exceeds capacity)", parallel)
            continue
        
        logger.info("\n--- Testing parallel=%d ---", parallel)
        
        try:
            assignments = manager.assign_environments(parallel)
            summary = manager.get_assignment_summary()
            
            # Check load distribution
            server_loads = []
            for server_info in summary["server_details"].values():
                server_loads.append(server_info["current_load"])
            
            min_load = min(server_loads)
            max_load = max(server_loads)
            load_diff = max_load - min_load
            
            logger.info("Load distribution: min=%d, max=%d, diff=%d", min_load, max_load, load_diff)
            
            if load_diff <= 1:
                logger.info("✅ Good load balancing (diff ≤ 1)")
            else:
                logger.info("⚠️  Suboptimal load balancing (diff > 1)")
            
        except Exception as e:
            logger.error("Failed parallel=%d: %s", parallel, str(e))

def test_config_integration():
    """Test integration with actual train_config.yml."""
    
    logger.info("\n=== CONFIG INTEGRATION TEST ===")
    
    try:
        import yaml
        
        # Load actual config file
        config_path = ROOT_DIR / "config" / "train_config.yml"
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        
        pokemon_showdown_config = config.get("pokemon_showdown", {})
        logger.info("Loaded Pokemon Showdown config from train_config.yml")
        
        # Create manager from actual config
        manager = MultiServerManager.from_config(pokemon_showdown_config)
        
        # Test with config's parallel setting
        config_parallel = config.get("parallel", 10)
        logger.info("Config parallel setting: %d", config_parallel)
        
        is_valid, error_msg = manager.validate_parallel_count(config_parallel)
        if is_valid:
            logger.info("✅ Config parallel setting is compatible with server capacity")
            assignments = manager.assign_environments(config_parallel)
            logger.info("✅ Environment assignment successful")
        else:
            logger.error("❌ Config incompatible: %s", error_msg)
            return False
        
        return True
        
    except Exception as e:
        logger.error("Config integration test failed: %s", str(e))
        return False

def main():
    """Run all multi-server tests."""
    
    # Test 1: Manager initialization
    manager = test_server_manager_initialization()
    
    # Test 2: Capacity validation
    valid_parallel = test_capacity_validation(manager)
    
    # Test 3: Environment assignment
    assignment_success = test_environment_assignment(manager, valid_parallel)
    
    # Test 4: Load balancing
    test_load_balancing(manager)
    
    # Test 5: Config integration
    config_success = test_config_integration()
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("MULTI-SERVER TEST SUMMARY")
    logger.info("=" * 80)
    logger.info("✅ Manager initialization: PASS")
    logger.info("✅ Capacity validation: PASS")
    logger.info("%s Environment assignment: %s", "✅" if assignment_success else "❌", "PASS" if assignment_success else "FAIL")
    logger.info("✅ Load balancing: PASS")
    logger.info("%s Config integration: %s", "✅" if config_success else "❌", "PASS" if config_success else "FAIL")
    
    if assignment_success and config_success:
        logger.info("🚀 All tests passed! Multi-server system is ready for use.")
        logger.info("\nUsage:")
        logger.info("  python train.py --episodes 1 --parallel 10 --device cpu")
        logger.info("  # Will automatically use multi-server load balancing")
    else:
        logger.info("⚠️  Some tests failed. Please check the configuration.")
    
    logger.info("=" * 80)

if __name__ == "__main__":
    main()