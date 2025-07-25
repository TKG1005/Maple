#!/bin/bash

# Pokemon Showdown Multiple Server Restart Script
# Usage: ./restart_showdown_servers.sh [number_of_servers] [starting_port]
# Example: ./restart_showdown_servers.sh 5 8000

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Default values
DEFAULT_SERVERS=5
DEFAULT_START_PORT=8000

# Parse arguments
NUM_SERVERS=${1:-$DEFAULT_SERVERS}
START_PORT=${2:-$DEFAULT_START_PORT}

echo "🔄 Restarting Pokemon Showdown servers..."
echo "🔧 Configuration: $NUM_SERVERS servers starting from port $START_PORT"
echo ""

# Calculate port range for stopping
END_PORT=$((START_PORT + NUM_SERVERS - 1))

echo "🛑 Step 1: Stopping existing servers (ports $START_PORT-$END_PORT)..."
"$SCRIPT_DIR/stop_showdown_servers.sh" $START_PORT $END_PORT

echo ""
echo "⏳ Waiting 3 seconds for clean shutdown..."
sleep 3

echo ""
echo "🚀 Step 2: Starting $NUM_SERVERS new servers..."
"$SCRIPT_DIR/start_showdown_servers.sh" $NUM_SERVERS $START_PORT

echo ""
echo "✅ Restart operation completed!"
echo ""
echo "🔍 Check status with: ./scripts/status_showdown_servers.sh $START_PORT $END_PORT"