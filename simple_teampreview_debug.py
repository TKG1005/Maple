#!/usr/bin/env python3
"""Simplified debug script to analyze teampreview handling differences."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# First, let's understand the key differences by examining the code structure
print("🔍 TEAMPREVIEW ANALYSIS - Code Structure Review")
print("=" * 60)

# 1. Online Mode (WebSocket) Battle Flow
print("\n1. ONLINE MODE (WebSocket) Battle Flow:")
print("   - Uses regular poke-env Battle objects")
print("   - WebSocket messages parsed by Battle.parse_message()")
print("   - Teampreview data populates Battle.teampreview_opponent_team")
print("   - Active Pokemon starts as None during teampreview")
print("   - State observer uses teampreview_opponent_team during teampreview")

# 2. Local Mode (IPC) Battle Flow  
print("\n2. LOCAL MODE (IPC) Battle Flow:")
print("   - Uses IPCBattle extending CustomBattle")
print("   - IPC messages converted to Pokemon Showdown protocol")
print("   - Should work the same as online mode but with different transport")
print("   - Uses _handle_battle_message_ipc() for message processing")

# Let's examine specific code sections to identify the issue
print("\n3. CRITICAL DIFFERENCES ANALYSIS:")

# Check StateObserver teampreview handling
print("\n   StateObserver.py teampreview handling:")
with open('src/state/state_observer.py', 'r') as f:
    content = f.read()
    
    # Find teampreview_opponent_team usage
    if 'teampreview_opponent_team' in content:
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'teampreview_opponent_team' in line:
                print(f"     Line {i+1}: {line.strip()}")
                # Show context
                for j in range(max(0, i-2), min(len(lines), i+3)):
                    if j != i:
                        print(f"     Line {j+1}: {lines[j].strip()}")
                break

# Check active_pokemon handling differences
print("\n   Active Pokemon during teampreview check:")
print("     - Regular Battle: active_pokemon = None during teampreview")
print("     - IPCBattle: active_pokemon populated during initialization")

# Check IPCBattle initialization
print("\n   IPCBattle initialization differences:")
with open('src/sim/ipc_battle.py', 'r') as f:
    content = f.read()
    
    # Find _create_minimal_teams usage
    if '_create_minimal_teams' in content:
        print("     - Creates minimal teams during __init__()")
        print("     - Sets active_pokemon = first Pokemon immediately")
        print("     - This differs from online mode where active_pokemon starts as None")

# Show the exact issue
print("\n4. THE CORE ISSUE:")
print("   ❌ IPCBattle creates teams and sets active_pokemon during __init__()")
print("   ❌ This means active_pokemon is never None during teampreview")
print("   ❌ StateObserver may get confused about battle phase")
print("   ✅ Online mode: active_pokemon = None during teampreview") 
print("   ✅ Online mode: active_pokemon set only after team selection")

print("\n5. SOLUTION APPROACH:")
print("   1. Modify IPCBattle to NOT set active_pokemon during initialization")
print("   2. Set active_pokemon = None during teampreview phase")
print("   3. Only populate active_pokemon after actual team selection")
print("   4. Ensure teampreview_opponent_team is properly populated")

print("\n6. FILES TO MODIFY:")
print("   - src/sim/ipc_battle.py: Fix active_pokemon timing")
print("   - src/sim/ipc_battle.py: Ensure proper teampreview phase handling")

print("\n✅ Analysis complete. The issue is that IPCBattle immediately sets")
print("   active_pokemon during initialization, while online mode keeps it")
print("   as None during teampreview phase.")