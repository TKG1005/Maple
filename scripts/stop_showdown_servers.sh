#!/bin/bash

# Pokemon Showdown Multiple Server Stop Script
# Usage: ./stop_showdown_servers.sh [port_range_start] [port_range_end]
# Example: ./stop_showdown_servers.sh 8000 8004

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PIDS_DIR="$PROJECT_ROOT/logs/showdown_pids"
LOGS_DIR="$PROJECT_ROOT/logs/showdown_logs"

# Parse arguments for port range
START_PORT=${1:-8000}
END_PORT=${2:-8010}  # Default to checking ports 8000-8010

echo "🛑 Stopping Pokemon Showdown servers..."
echo "🔍 Checking port range: $START_PORT-$END_PORT"

# Create directories if they don't exist
mkdir -p "$PIDS_DIR"

STOPPED_SERVERS=0
NOT_RUNNING_SERVERS=0

# Function to stop a single server
stop_server() {
    local port=$1
    local pid_file="$PIDS_DIR/showdown_server_${port}.pid"
    local log_file="$LOGS_DIR/showdown_server_${port}.log"
    
    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        
        if ps -p $pid > /dev/null 2>&1; then
            echo "🔄 Stopping server on port $port (PID: $pid)..."
            
            # Try graceful shutdown first
            kill $pid 2>/dev/null || true
            
            # Wait a moment for graceful shutdown
            sleep 2
            
            # Force kill if still running
            if ps -p $pid > /dev/null 2>&1; then
                echo "   ⚡ Force killing server on port $port..."
                kill -9 $pid 2>/dev/null || true
                sleep 1
            fi
            
            # Verify the process is stopped
            if ! ps -p $pid > /dev/null 2>&1; then
                echo "   ✅ Server on port $port stopped successfully"
                ((STOPPED_SERVERS++))
            else
                echo "   ❌ Failed to stop server on port $port"
            fi
        else
            echo "⚪ Server on port $port was not running (stale PID file)"
            ((NOT_RUNNING_SERVERS++))
        fi
        
        # Remove PID file
        rm -f "$pid_file"
    else
        # Check if something else is running on the port
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            local running_pid=$(lsof -Pi :$port -sTCP:LISTEN -t)
            echo "⚠️  Found process $running_pid running on port $port (not managed by this script)"
            echo "   Use 'kill $running_pid' to stop it manually if needed"
        fi
    fi
}

# Function to stop all servers by scanning for PID files
stop_all_by_pid_files() {
    echo ""
    echo "🔍 Scanning for running Pokemon Showdown servers..."
    
    if [ -d "$PIDS_DIR" ]; then
        local found_servers=false
        for pid_file in "$PIDS_DIR"/showdown_server_*.pid; do
            if [ -f "$pid_file" ]; then
                found_servers=true
                local filename=$(basename "$pid_file")
                local port=$(echo "$filename" | grep -o '[0-9]\+')
                stop_server "$port"
            fi
        done
        
        if [ "$found_servers" = false ]; then
            echo "ℹ️  No PID files found in $PIDS_DIR"
        fi
    else
        echo "ℹ️  PID directory $PIDS_DIR does not exist"
    fi
}

# Function to stop servers in specified port range
stop_by_port_range() {
    echo ""
    echo "🔍 Stopping servers in port range $START_PORT-$END_PORT..."
    
    for ((port=START_PORT; port<=END_PORT; port++)); do
        stop_server "$port"
    done
}

# Main execution
echo "─────────────────────────────────────────────────────────────"

# If no arguments, stop all servers found in PID directory
if [ $# -eq 0 ]; then
    stop_all_by_pid_files
else
    stop_by_port_range
fi

echo "─────────────────────────────────────────────────────────────"
echo ""

# Summary
if [ $STOPPED_SERVERS -gt 0 ]; then
    echo "✅ Successfully stopped $STOPPED_SERVERS Pokemon Showdown server(s)"
fi

if [ $NOT_RUNNING_SERVERS -gt 0 ]; then
    echo "ℹ️  $NOT_RUNNING_SERVERS server(s) were not running"
fi

# Clean up empty directories
if [ -d "$PIDS_DIR" ] && [ -z "$(ls -A "$PIDS_DIR")" ]; then
    rmdir "$PIDS_DIR" 2>/dev/null || true
fi

# Show any remaining processes on Pokemon Showdown ports
echo ""
echo "🔍 Checking for any remaining Pokemon Showdown processes..."
REMAINING_PROCESSES=0

for ((port=8000; port<=8020; port++)); do
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        local pid=$(lsof -Pi :$port -sTCP:LISTEN -t)
        local cmd=$(ps -p $pid -o comm= 2>/dev/null || echo "unknown")
        if [[ "$cmd" == *"node"* ]] || [[ "$cmd" == *"pokemon-showdown"* ]]; then
            echo "⚠️  Process $pid ($cmd) still running on port $port"
            ((REMAINING_PROCESSES++))
        fi
    fi
done

if [ $REMAINING_PROCESSES -eq 0 ]; then
    echo "✨ All Pokemon Showdown servers have been stopped"
else
    echo ""
    echo "⚠️  $REMAINING_PROCESSES process(es) may still be running"
    echo "   Use 'ps aux | grep pokemon-showdown' to check manually"
    echo "   Use 'kill -9 [PID]' to force stop if needed"
fi

echo ""
echo "🏁 Stop operation completed"