#!/bin/bash

# Pokemon Showdown Multiple Server Startup Script
# Usage: ./start_showdown_servers.sh [number_of_servers] [starting_port]
# Example: ./start_showdown_servers.sh 5 8000

set -e

# Default values
DEFAULT_SERVERS=5
DEFAULT_START_PORT=8000
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
POKEMON_SHOWDOWN_DIR="$PROJECT_ROOT/pokemon-showdown"
PIDS_DIR="$PROJECT_ROOT/logs/showdown_pids"
LOGS_DIR="$PROJECT_ROOT/logs/showdown_logs"

# Parse arguments
NUM_SERVERS=${1:-$DEFAULT_SERVERS}
START_PORT=${2:-$DEFAULT_START_PORT}

# Validate arguments
if ! [[ "$NUM_SERVERS" =~ ^[0-9]+$ ]] || [ "$NUM_SERVERS" -lt 1 ] || [ "$NUM_SERVERS" -gt 20 ]; then
    echo "❌ Error: Number of servers must be between 1 and 20"
    echo "Usage: $0 [number_of_servers] [starting_port]"
    exit 1
fi

if ! [[ "$START_PORT" =~ ^[0-9]+$ ]] || [ "$START_PORT" -lt 1000 ] || [ "$START_PORT" -gt 65000 ]; then
    echo "❌ Error: Starting port must be between 1000 and 65000"
    echo "Usage: $0 [number_of_servers] [starting_port]"
    exit 1
fi

echo "🚀 Starting $NUM_SERVERS Pokemon Showdown servers..."
echo "📂 Project root: $PROJECT_ROOT"
echo "🔌 Port range: $START_PORT-$((START_PORT + NUM_SERVERS - 1))"

# Create necessary directories
mkdir -p "$PIDS_DIR"
mkdir -p "$LOGS_DIR"

# Check if pokemon-showdown directory exists
if [ ! -d "$POKEMON_SHOWDOWN_DIR" ]; then
    echo "❌ Error: Pokemon Showdown directory not found at $POKEMON_SHOWDOWN_DIR"
    echo "Please ensure pokemon-showdown is installed in the project root."
    exit 1
fi

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is not installed or not in PATH"
    exit 1
fi

# Function to check if port is available
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 1  # Port is in use
    else
        return 0  # Port is available
    fi
}

# Function to start a single server
start_server() {
    local port=$1
    local server_id=$((port - START_PORT + 1))
    local pid_file="$PIDS_DIR/showdown_server_${port}.pid"
    local log_file="$LOGS_DIR/showdown_server_${port}.log"
    
    echo "🔧 Starting Pokemon Showdown server #${server_id} on port ${port}..."
    
    # Check if port is available
    if ! check_port $port; then
        echo "⚠️  Warning: Port $port is already in use, skipping server #${server_id}"
        return 1
    fi
    
    # Check if server is already running
    if [ -f "$pid_file" ]; then
        local existing_pid=$(cat "$pid_file")
        if ps -p $existing_pid > /dev/null 2>&1; then
            echo "⚠️  Warning: Server #${server_id} (PID: $existing_pid) is already running on port $port"
            return 1
        else
            # Remove stale PID file
            rm -f "$pid_file"
        fi
    fi
    
    # Start the server in background
    cd "$POKEMON_SHOWDOWN_DIR"
    nohup node pokemon-showdown start --no-security --port $port > "$log_file" 2>&1 &
    local pid=$!
    
    # Save PID to file
    echo $pid > "$pid_file"
    
    # Give the server a moment to start
    sleep 1
    
    # Verify the server started successfully
    if ps -p $pid > /dev/null 2>&1; then
        echo "✅ Server #${server_id} started successfully (PID: $pid, Port: $port)"
        echo "   Log file: $log_file"
        return 0
    else
        echo "❌ Failed to start server #${server_id} on port $port"
        rm -f "$pid_file"
        return 1
    fi
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Interrupt received. Use './stop_showdown_servers.sh' to stop all servers."
    exit 0
}

# Set trap for cleanup
trap cleanup INT TERM

# Start all servers
echo ""
echo "🏁 Starting servers..."
echo "─────────────────────────────────────────────────────────────"

STARTED_SERVERS=0
FAILED_SERVERS=0

for ((i=0; i<NUM_SERVERS; i++)); do
    port=$((START_PORT + i))
    if start_server $port; then
        ((STARTED_SERVERS++))
    else
        ((FAILED_SERVERS++))
    fi
done

echo "─────────────────────────────────────────────────────────────"
echo ""

# Summary
if [ $STARTED_SERVERS -gt 0 ]; then
    echo "🎉 Successfully started $STARTED_SERVERS Pokemon Showdown server(s)"
    if [ $FAILED_SERVERS -gt 0 ]; then
        echo "⚠️  $FAILED_SERVERS server(s) failed to start or were already running"
    fi
    echo ""
    echo "📋 Server Status:"
    for ((i=0; i<NUM_SERVERS; i++)); do
        port=$((START_PORT + i))
        pid_file="$PIDS_DIR/showdown_server_${port}.pid"
        if [ -f "$pid_file" ]; then
            pid=$(cat "$pid_file")
            if ps -p $pid > /dev/null 2>&1; then
                echo "   ✅ Server #$((i+1)): http://localhost:$port (PID: $pid) - Running"
            else
                echo "   ❌ Server #$((i+1)): http://localhost:$port - Failed"
            fi
        else
            echo "   ⚪ Server #$((i+1)): http://localhost:$port - Not started"
        fi
    done
    echo ""
    echo "🔍 Management commands:"
    echo "   • Check status: ./scripts/status_showdown_servers.sh"
    echo "   • Stop all servers: ./scripts/stop_showdown_servers.sh"
    echo "   • View logs: tail -f $LOGS_DIR/showdown_server_[PORT].log"
    echo ""
    echo "⚙️  Update train_config.yml servers configuration if needed:"
    echo "   pokemon_showdown:"
    echo "     servers:"
    for ((i=0; i<NUM_SERVERS; i++)); do
        port=$((START_PORT + i))
        echo "       - host: \"localhost\""
        echo "         port: $port"
        echo "         max_connections: 25"
    done
else
    echo "❌ No servers were started successfully"
    exit 1
fi

echo "🚀 All servers are running in the background."
echo "   Press Ctrl+C to return to terminal (servers will continue running)"
echo "   Use './scripts/stop_showdown_servers.sh' to stop all servers when done"