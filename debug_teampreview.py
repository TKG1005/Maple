#!/usr/bin/env python3
"""Debug script to analyze teampreview battle state differences between online and local mode."""

import asyncio
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.env.pokemon_env import PokemonEnv
from src.state.state_observer import StateObserver
from src.action import action_helper
from src.agents.random_agent import RandomAgent
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def analyze_battle_state(battle, mode_name):
    """Analyze and log battle state during teampreview."""
    print(f"\n=== {mode_name} MODE BATTLE STATE ANALYSIS ===")
    
    # Basic battle info
    print(f"Battle tag: {getattr(battle, 'battle_tag', 'N/A')}")
    print(f"Battle type: {type(battle).__name__}")
    print(f"Turn: {getattr(battle, 'turn', 'N/A')}")
    print(f"Finished: {getattr(battle, 'finished', 'N/A')}")
    
    # Active Pokemon analysis
    active_pokemon = getattr(battle, 'active_pokemon', None)
    print(f"Active Pokemon: {active_pokemon}")
    if active_pokemon:
        print(f"  - Species: {getattr(active_pokemon, 'species', 'N/A')}")
        print(f"  - Active: {getattr(active_pokemon, 'active', 'N/A')}")
        print(f"  - Fainted: {getattr(active_pokemon, 'fainted', 'N/A')}")
        print(f"  - HP: {getattr(active_pokemon, 'current_hp', 'N/A')}/{getattr(active_pokemon, 'max_hp', 'N/A')}")
    else:
        print("  - Active Pokemon is None")
    
    # Team analysis
    team = getattr(battle, 'team', {})
    print(f"Team size: {len(team) if team else 0}")
    if team:
        for key, pokemon in team.items():
            print(f"  - {key}: {getattr(pokemon, 'species', 'N/A')} (active: {getattr(pokemon, 'active', False)})")
    
    # Team preview specific
    teampreview_team = getattr(battle, 'teampreview_opponent_team', None)
    print(f"Teampreview opponent team: {len(teampreview_team) if teampreview_team else 0} Pokemon")
    
    opponent_team = getattr(battle, 'opponent_team', {})
    print(f"Opponent team size: {len(opponent_team) if opponent_team else 0}")
    
    # Last request analysis
    last_request = getattr(battle, 'last_request', None)
    print(f"Last request: {type(last_request) if last_request else None}")
    if last_request:
        print(f"  - Has teamPreview: {'teamPreview' in last_request}")
        if 'teamPreview' in last_request:
            print(f"  - TeamPreview value: {last_request['teamPreview']}")
        if 'side' in last_request:
            side_pokemon = last_request.get('side', {}).get('pokemon', [])
            print(f"  - Side Pokemon count: {len(side_pokemon)}")
    
    # Available actions
    available_moves = getattr(battle, 'available_moves', [])
    available_switches = getattr(battle, 'available_switches', [])
    print(f"Available moves: {len(available_moves)}")
    print(f"Available switches: {len(available_switches)}")
    
    print("=" * 50)

async def test_mode(mode, full_ipc=False):
    """Test a specific battle mode and analyze teampreview state."""
    print(f"\n🧪 Testing {mode} mode (full_ipc={full_ipc})")
    
    try:
        # Initialize components
        state_observer = StateObserver("config/state_spec.yml")
        
        # Create environment
        env = PokemonEnv(
            state_observer=state_observer,
            action_helper=action_helper,
            battle_mode=mode,
            server_configuration=LocalhostServerConfiguration,
            full_ipc=full_ipc,
            log_level=logging.INFO
        )
        
        # Register agents
        agent1 = RandomAgent(env)
        agent2 = RandomAgent(env)
        env.register_agent(agent1, "player_0")
        env.register_agent(agent2, "player_1")
        
        # Reset environment (this should trigger teampreview)
        observations, info = env.reset()
        
        # Get battle objects for analysis
        battle0 = env.get_current_battle("player_0")
        battle1 = env.get_current_battle("player_1")
        
        # Analyze both battle objects
        analyze_battle_state(battle0, f"{mode.upper()}_PLAYER_0")
        analyze_battle_state(battle1, f"{mode.upper()}_PLAYER_1")
        
        # Test state observation
        obs0 = state_observer.observe(battle0)
        obs1 = state_observer.observe(battle1)
        
        print(f"Observation shapes: player_0={obs0.shape}, player_1={obs1.shape}")
        
        # Check if active_pokemon becomes None during teampreview
        print(f"Active Pokemon None check: player_0={battle0.active_pokemon is None}, player_1={battle1.active_pokemon is None}")
        
        # Clean up
        env.close()
        
        return {
            'mode': mode,
            'full_ipc': full_ipc,
            'battle0_active_none': battle0.active_pokemon is None,
            'battle1_active_none': battle1.active_pokemon is None,
            'team0_size': len(battle0.team) if battle0.team else 0,
            'team1_size': len(battle1.team) if battle1.team else 0,
            'last_request0': battle0.last_request is not None,
            'last_request1': battle1.last_request is not None,
        }
        
    except Exception as e:
        logger.error(f"Error testing {mode} mode: {e}")
        import traceback
        traceback.print_exc()
        return {'mode': mode, 'full_ipc': full_ipc, 'error': str(e)}

async def main():
    """Run teampreview analysis for different battle modes."""
    print("🔍 Analyzing teampreview battle state differences between online and local modes")
    
    results = []
    
    # Test online mode (traditional WebSocket)
    try:
        result = await test_mode("online")
        results.append(result)
    except Exception as e:
        logger.error(f"Online mode test failed: {e}")
        results.append({'mode': 'online', 'error': str(e)})
    
    # Test local mode (IPC with WebSocket fallback)
    try:
        result = await test_mode("local", full_ipc=False)
        results.append(result)
    except Exception as e:
        logger.error(f"Local mode test failed: {e}")
        results.append({'mode': 'local', 'full_ipc': False, 'error': str(e)})
    
    # Test full IPC mode (no WebSocket fallback)
    try:
        result = await test_mode("local", full_ipc=True)
        results.append(result)
    except Exception as e:
        logger.error(f"Full IPC mode test failed: {e}")
        results.append({'mode': 'local', 'full_ipc': True, 'error': str(e)})
    
    # Summary comparison
    print("\n📊 SUMMARY COMPARISON")
    print("=" * 60)
    for result in results:
        if 'error' in result:
            print(f"{result['mode']} (full_ipc={result.get('full_ipc', False)}): ERROR - {result['error']}")
        else:
            print(f"{result['mode']} (full_ipc={result.get('full_ipc', False)}):")
            print(f"  - Active Pokemon None: P0={result.get('battle0_active_none')}, P1={result.get('battle1_active_none')}")
            print(f"  - Team sizes: P0={result.get('team0_size')}, P1={result.get('team1_size')}")
            print(f"  - Has last_request: P0={result.get('last_request0')}, P1={result.get('last_request1')}")

if __name__ == "__main__":
    asyncio.run(main())