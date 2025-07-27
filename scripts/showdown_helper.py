#!/usr/bin/env python3
"""
Helper script for Windows Pokemon Showdown server management.
Provides cross-platform utilities for the batch script.
"""

import os
import sys
import psutil
import signal
import time
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
PID_DIR = PROJECT_ROOT / "logs" / "pids"
LOG_DIR = PROJECT_ROOT / "logs" / "showdown_logs"
TRAIN_CONFIG = PROJECT_ROOT / "config" / "train_config.yml"
DEFAULT_PORT = 8000
MAX_CONNECTIONS = 25


def ensure_directories():
    """Create necessary directories if they don't exist."""
    PID_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def get_server_count_from_config() -> int:
    """Parse train_config.yml to determine required server count."""
    try:
        with open(TRAIN_CONFIG, 'r') as f:
            config = yaml.safe_load(f)
        
        # Check for multi-server configuration
        pokemon_showdown = config.get('pokemon_showdown', {})
        servers = pokemon_showdown.get('servers', [])
        if servers:
            return len(servers)
        
        # Fallback to calculating from parallel count
        parallel = config.get('parallel', 10)
        return max(1, (parallel + MAX_CONNECTIONS - 1) // MAX_CONNECTIONS)
    except Exception as e:
        print(f"Warning: Could not parse config: {e}")
        return 5


def find_process_by_port(port: int) -> Optional[int]:
    """Find process PID listening on the given port."""
    for conn in psutil.net_connections(kind='inet'):
        if conn.laddr.port == port and conn.status == 'LISTEN':
            return conn.pid
    return None


def is_process_running(pid: int) -> bool:
    """Check if a process with given PID is running."""
    try:
        process = psutil.Process(pid)
        return process.is_running() and "node" in process.name().lower()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False


def start_server(port: int) -> Optional[int]:
    """Start a Pokemon Showdown server on the given port."""
    import subprocess
    
    log_file = LOG_DIR / f"showdown_server_{port}.log"
    showdown_dir = PROJECT_ROOT / "pokemon-showdown"
    
    # Start server
    with open(log_file, 'w') as log:
        if sys.platform == "win32":
            # Windows: Use CREATE_NEW_PROCESS_GROUP for better process management
            process = subprocess.Popen(
                ["node", "pokemon-showdown", "start", "--no-security", "--port", str(port)],
                cwd=showdown_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
        else:
            # Unix-like systems
            process = subprocess.Popen(
                ["node", "pokemon-showdown", "start", "--no-security", "--port", str(port)],
                cwd=showdown_dir,
                stdout=log,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid
            )
    
    # Wait for server to start
    time.sleep(2)
    
    # Verify server started
    if process.poll() is None:
        # Save PID
        pid_file = PID_DIR / f"showdown_{port}.pid"
        pid_file.write_text(str(process.pid))
        return process.pid
    
    return None


def stop_server(port: int) -> bool:
    """Stop a Pokemon Showdown server on the given port."""
    pid_file = PID_DIR / f"showdown_{port}.pid"
    
    # Try to read PID from file
    if pid_file.exists():
        try:
            pid = int(pid_file.read_text().strip())
            if is_process_running(pid):
                # Terminate process
                if sys.platform == "win32":
                    os.kill(pid, signal.SIGTERM)
                else:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                
                # Wait for graceful shutdown
                time.sleep(1)
                
                # Force kill if still running
                if is_process_running(pid):
                    if sys.platform == "win32":
                        os.kill(pid, signal.SIGTERM)  # Windows doesn't have SIGKILL
                    else:
                        os.killpg(os.getpgid(pid), signal.SIGKILL)
                
                pid_file.unlink()
                return True
        except Exception as e:
            print(f"Error stopping server on port {port}: {e}")
        
        # Clean up stale PID file
        if pid_file.exists():
            pid_file.unlink()
    
    # Also check if port is in use by unmanaged process
    pid = find_process_by_port(port)
    if pid:
        try:
            process = psutil.Process(pid)
            process.terminate()
            return True
        except Exception:
            pass
    
    return False


def get_server_status() -> List[Dict]:
    """Get status of all servers."""
    status = []
    
    for i in range(10):  # Check ports 8000-8009
        port = DEFAULT_PORT + i
        pid_file = PID_DIR / f"showdown_{port}.pid"
        log_file = LOG_DIR / f"showdown_server_{port}.log"
        
        server_info = {
            'port': port,
            'status': 'NOT RUNNING',
            'pid': None,
            'managed': False,
            'log_size': 0,
            'log_modified': None
        }
        
        # Check PID file
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                if is_process_running(pid):
                    server_info['status'] = 'RUNNING'
                    server_info['pid'] = pid
                    server_info['managed'] = True
                else:
                    # Stale PID file
                    pid_file.unlink()
            except Exception:
                pass
        
        # Check if port is in use
        if server_info['status'] == 'NOT RUNNING':
            pid = find_process_by_port(port)
            if pid:
                server_info['status'] = 'IN USE'
                server_info['pid'] = pid
                server_info['managed'] = False
        
        # Log file info
        if log_file.exists():
            stat = log_file.stat()
            server_info['log_size'] = stat.st_size
            server_info['log_modified'] = time.strftime('%Y-%m-%d %H:%M:%S', 
                                                       time.localtime(stat.st_mtime))
        
        status.append(server_info)
    
    return status


def main():
    """Main entry point for command-line usage."""
    ensure_directories()
    
    if len(sys.argv) < 2:
        print("Usage: showdown_helper.py [command] [args...]")
        print("Commands: start, stop, status, config")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "start":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT
        pid = start_server(port)
        if pid:
            print(f"Started server on port {port} (PID: {pid})")
        else:
            print(f"Failed to start server on port {port}")
            sys.exit(1)
    
    elif command == "stop":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PORT
        if stop_server(port):
            print(f"Stopped server on port {port}")
        else:
            print(f"No server running on port {port}")
    
    elif command == "status":
        status = get_server_status()
        running = sum(1 for s in status if s['status'] == 'RUNNING')
        print(f"\nPokemon Showdown Server Status")
        print(f"{'='*50}")
        print(f"Running servers: {running}/{len(status)}")
        print(f"\nPort    Status         PID      Managed  Log")
        print(f"{'-'*50}")
        for s in status:
            managed = "Yes" if s['managed'] else "No" if s['status'] != 'NOT RUNNING' else "-"
            pid = str(s['pid']) if s['pid'] else "-"
            log_info = f"{s['log_size']:,} bytes" if s['log_size'] > 0 else "-"
            print(f"{s['port']}  {s['status']:13}  {pid:7}  {managed:7}  {log_info}")
    
    elif command == "config":
        count = get_server_count_from_config()
        print(count)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()