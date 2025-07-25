#!/bin/bash

# Pokemon Showdown Multiple Server Status Script
# Usage: ./status_showdown_servers.sh [start_port] [end_port]
# Example: ./status_showdown_servers.sh 8000 8010

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PIDS_DIR="$PROJECT_ROOT/logs/showdown_pids"
LOGS_DIR="$PROJECT_ROOT/logs/showdown_logs"

# Parse arguments
START_PORT=${1:-8000}
END_PORT=${2:-8010}

echo "🔍 Pokemon Showdown Server Status"
echo "📂 Project root: $PROJECT_ROOT"
echo "🔌 Checking port range: $START_PORT-$END_PORT"
echo "═════════════════════════════════════════════════════════════"

RUNNING_SERVERS=0
MANAGED_SERVERS=0
UNMANAGED_SERVERS=0
TOTAL_CHECKED=0

# Function to get process info
get_process_info() {
    local pid=$1
    local start_time=$(ps -p $pid -o lstart= 2>/dev/null | xargs)
    local cpu_usage=$(ps -p $pid -o %cpu= 2>/dev/null | xargs)
    local mem_usage=$(ps -p $pid -o %mem= 2>/dev/null | xargs)
    echo "Start: $start_time | CPU: ${cpu_usage}% | Mem: ${mem_usage}%"
}

# Function to check server status
check_server_status() {
    local port=$1
    local pid_file="$PIDS_DIR/showdown_server_${port}.pid"
    local log_file="$LOGS_DIR/showdown_server_${port}.log"
    
    ((TOTAL_CHECKED++))
    
    # Check if port is in use
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        local running_pid=$(lsof -Pi :$port -sTCP:LISTEN -t)
        ((RUNNING_SERVERS++))
        
        # Check if it's managed by our script
        if [ -f "$pid_file" ]; then
            local managed_pid=$(cat "$pid_file")
            if [ "$running_pid" = "$managed_pid" ]; then
                ((MANAGED_SERVERS++))
                echo "🟢 Port $port: RUNNING (Managed)"
                echo "   PID: $running_pid"
                echo "   $(get_process_info $running_pid)"
                echo "   Log: $log_file"
                
                # Show log file size and recent activity
                if [ -f "$log_file" ]; then
                    local log_size=$(du -h "$log_file" 2>/dev/null | cut -f1)
                    local last_modified=$(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" "$log_file" 2>/dev/null)
                    echo "   Log size: $log_size | Last activity: $last_modified"
                else
                    echo "   ⚠️  Log file not found"
                fi
            else
                ((UNMANAGED_SERVERS++))
                echo "🟡 Port $port: RUNNING (PID mismatch - $running_pid vs $managed_pid)"
                echo "   Current PID: $running_pid"
                echo "   $(get_process_info $running_pid)"
                echo "   ⚠️  PID file contains different PID: $managed_pid"
            fi
        else
            ((UNMANAGED_SERVERS++))
            echo "🟡 Port $port: RUNNING (Unmanaged)"
            echo "   PID: $running_pid"
            echo "   $(get_process_info $running_pid)"
            echo "   ⚠️  No PID file found (not started by this script)"
        fi
    else
        # Port is not in use
        if [ -f "$pid_file" ]; then
            local managed_pid=$(cat "$pid_file")
            if ps -p $managed_pid > /dev/null 2>&1; then
                echo "🔴 Port $port: STOPPED (Process $managed_pid running but not listening)"
                echo "   ⚠️  Process exists but not bound to port (may have crashed)"
            else
                echo "🔴 Port $port: STOPPED (Stale PID file)"
                echo "   ⚠️  PID file exists but process $managed_pid is not running"
            fi
        else
            echo "⚪ Port $port: NOT RUNNING"
        fi
    fi
    echo ""
}

# Function to show overall system info
show_system_info() {
    echo "💻 System Information:"
    echo "   Node.js: $(node --version 2>/dev/null || echo 'Not found')"
    echo "   Current time: $(date)"
    echo "   Uptime: $(uptime | cut -d',' -f1 | cut -d' ' -f4-)"
    echo ""
}

# Function to show process tree
show_process_tree() {
    echo "🌳 Pokemon Showdown Process Tree:"
    local pokemon_processes=$(pgrep -f "pokemon-showdown" 2>/dev/null || true)
    if [ -n "$pokemon_processes" ]; then
        echo "$pokemon_processes" | while read -r pid; do
            local port_info=$(lsof -p $pid 2>/dev/null | grep LISTEN | grep -o ':[0-9]\+' | head -1 | cut -d':' -f2)
            local cmd=$(ps -p $pid -o args= 2>/dev/null | cut -c1-80)
            if [ -n "$port_info" ]; then
                echo "   📡 PID $pid (Port $port_info): $cmd"
            else
                echo "   📡 PID $pid (No port): $cmd"
            fi
        done
    else
        echo "   ℹ️  No Pokemon Showdown processes found"
    fi
    echo ""
}

# Function to show logs summary
show_logs_summary() {
    echo "📋 Log Files Summary:"
    if [ -d "$LOGS_DIR" ]; then
        local log_count=$(find "$LOGS_DIR" -name "showdown_server_*.log" | wc -l)
        if [ $log_count -gt 0 ]; then
            echo "   Total log files: $log_count"
            local total_size=$(du -sh "$LOGS_DIR" 2>/dev/null | cut -f1)
            echo "   Total log size: $total_size"
            echo ""
            echo "   Recent log files:"
            find "$LOGS_DIR" -name "showdown_server_*.log" -exec ls -lh {} \; | tail -5 | while read -r line; do
                echo "   $line"
            done
        else
            echo "   ℹ️  No log files found"
        fi
    else
        echo "   ℹ️  Log directory does not exist"
    fi
    echo ""
}

# Main execution
show_system_info

echo "🔍 Checking servers on ports $START_PORT-$END_PORT..."
echo "─────────────────────────────────────────────────────────────"

for ((port=START_PORT; port<=END_PORT; port++)); do
    check_server_status $port
done

echo "═════════════════════════════════════════════════════════════"
echo ""

# Summary
echo "📊 Summary:"
echo "   🔍 Ports checked: $TOTAL_CHECKED"
echo "   🟢 Running servers: $RUNNING_SERVERS"
echo "   📊 Managed by script: $MANAGED_SERVERS"
echo "   ⚠️  Unmanaged processes: $UNMANAGED_SERVERS"
echo "   🔴 Stopped servers: $((TOTAL_CHECKED - RUNNING_SERVERS))"
echo ""

show_process_tree
show_logs_summary

# Health recommendations
if [ $UNMANAGED_SERVERS -gt 0 ]; then
    echo "⚠️  Recommendations:"
    echo "   • $UNMANAGED_SERVERS unmanaged server(s) detected"
    echo "   • Consider stopping them manually: kill [PID]"
    echo "   • Or restart with: ./scripts/stop_showdown_servers.sh && ./scripts/start_showdown_servers.sh"
    echo ""
fi

if [ $MANAGED_SERVERS -gt 0 ]; then
    echo "✅ $MANAGED_SERVERS managed server(s) are running correctly"
    echo ""
    echo "🔧 Management commands:"
    echo "   • Stop all servers: ./scripts/stop_showdown_servers.sh"
    echo "   • Restart servers: ./scripts/restart_showdown_servers.sh $MANAGED_SERVERS"
    echo "   • View live logs: tail -f $LOGS_DIR/showdown_server_[PORT].log"
fi

echo ""
echo "🏁 Status check completed"