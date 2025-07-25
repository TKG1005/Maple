"""Multi-server management for Pokemon Showdown connections."""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from poke_env.ps_client import ServerConfiguration

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    """Configuration for a single Pokemon Showdown server."""
    
    host: str = "localhost"
    port: int = 8000
    max_connections: int = 20
    websocket_url: Optional[str] = None
    authentication_url: Optional[str] = None
    
    def __post_init__(self):
        """Generate URLs if not provided."""
        if self.websocket_url is None:
            self.websocket_url = f"ws://{self.host}:{self.port}/showdown/websocket"
        if self.authentication_url is None:
            self.authentication_url = "https://play.pokemonshowdown.com/action.php?"
    
    def to_server_configuration(self) -> ServerConfiguration:
        """Convert to poke-env ServerConfiguration."""
        return ServerConfiguration(
            websocket_url=self.websocket_url,
            authentication_url=self.authentication_url
        )


class MultiServerManager:
    """Manages multiple Pokemon Showdown servers for load balancing."""
    
    def __init__(self, server_configs: List[Dict]) -> None:
        """Initialize multi-server manager.
        
        Parameters
        ----------
        server_configs : List[Dict]
            List of server configuration dictionaries
        """
        self.servers: List[ServerConfig] = []
        self.server_assignments: Dict[int, int] = {}  # env_id -> server_index
        self.current_connections: List[int] = []  # connections per server
        
        # Parse server configurations
        for config in server_configs:
            server_config = ServerConfig(
                host=config.get("host", "localhost"),
                port=config.get("port", 8000),
                max_connections=config.get("max_connections", 20),
                websocket_url=config.get("websocket_url"),
                authentication_url=config.get("authentication_url")
            )
            self.servers.append(server_config)
            self.current_connections.append(0)
        
        if not self.servers:
            # Default single server configuration
            self.servers.append(ServerConfig())
            self.current_connections.append(0)
        
        logger.info("MultiServerManager initialized with %d servers", len(self.servers))
        for i, server in enumerate(self.servers):
            logger.info("  Server %d: %s:%d (max %d connections)", 
                       i, server.host, server.port, server.max_connections)
    
    def get_total_capacity(self) -> int:
        """Get total connection capacity across all servers.
        
        Returns
        -------
        int
            Total maximum connections
        """
        return sum(server.max_connections for server in self.servers)
    
    def validate_parallel_count(self, parallel: int) -> Tuple[bool, str]:
        """Validate that parallel count doesn't exceed server capacity.
        
        Parameters
        ----------
        parallel : int
            Number of parallel environments requested
            
        Returns
        -------
        Tuple[bool, str]
            (is_valid, error_message)
        """
        total_capacity = self.get_total_capacity()
        
        if parallel > total_capacity:
            return False, (
                f"Parallel environments ({parallel}) exceeds server capacity ({total_capacity}). "
                f"Available servers: {[f'{s.host}:{s.port}({s.max_connections})' for s in self.servers]}"
            )
        
        return True, ""
    
    def assign_environments(self, parallel: int) -> Dict[int, Tuple[ServerConfiguration, int]]:
        """Assign environments to servers with load balancing.
        
        Parameters
        ----------
        parallel : int
            Number of parallel environments
            
        Returns
        -------
        Dict[int, Tuple[ServerConfiguration, int]]
            Mapping from environment_id to (server_config, server_index)
        """
        # Validate capacity
        is_valid, error_msg = self.validate_parallel_count(parallel)
        if not is_valid:
            raise ValueError(error_msg)
        
        assignments = {}
        server_loads = [0] * len(self.servers)
        
        # Distribute environments evenly across servers
        for env_id in range(parallel):
            # Find server with lowest current load
            min_load_server = min(range(len(self.servers)), 
                                key=lambda i: server_loads[i])
            
            # Check if server has capacity
            if server_loads[min_load_server] >= self.servers[min_load_server].max_connections:
                # This should not happen due to validation, but safety check
                raise RuntimeError(f"Server {min_load_server} exceeds capacity")
            
            # Assign environment to server
            server_config = self.servers[min_load_server]
            assignments[env_id] = (server_config.to_server_configuration(), min_load_server)
            server_loads[min_load_server] += 1
            
            # Track assignment for monitoring
            self.server_assignments[env_id] = min_load_server
        
        # Update connection counts
        self.current_connections = server_loads.copy()
        
        # Log assignment distribution
        logger.info("Environment assignment across servers:")
        for server_idx, load in enumerate(server_loads):
            if load > 0:
                server = self.servers[server_idx]
                logger.info("  Server %d (%s:%d): %d/%d environments (%.1f%%)",
                           server_idx, server.host, server.port, load, 
                           server.max_connections, (load / server.max_connections) * 100)
        
        return assignments
    
    def get_server_for_environment(self, env_id: int) -> Optional[ServerConfiguration]:
        """Get server configuration for a specific environment.
        
        Parameters
        ----------
        env_id : int
            Environment ID
            
        Returns
        -------
        Optional[ServerConfiguration]
            Server configuration or None if not assigned
        """
        if env_id in self.server_assignments:
            server_idx = self.server_assignments[env_id]
            return self.servers[server_idx].to_server_configuration()
        return None
    
    def get_assignment_summary(self) -> Dict:
        """Get summary of current server assignments.
        
        Returns
        -------
        Dict
            Summary information about server assignments
        """
        server_loads = {}
        for server_idx, load in enumerate(self.current_connections):
            server = self.servers[server_idx]
            server_loads[f"server_{server_idx}"] = {
                "host": server.host,
                "port": server.port,
                "current_load": load,
                "max_capacity": server.max_connections,
                "utilization": (load / server.max_connections) * 100 if server.max_connections > 0 else 0,
                "websocket_url": server.websocket_url
            }
        
        return {
            "total_servers": len(self.servers),
            "total_capacity": self.get_total_capacity(),
            "total_assigned": sum(self.current_connections),
            "server_details": server_loads
        }
    
    def print_assignment_report(self) -> None:
        """Print detailed assignment report."""
        summary = self.get_assignment_summary()
        
        print("\n" + "=" * 80)
        print("MULTI-SERVER ASSIGNMENT REPORT")
        print("=" * 80)
        print(f"Total servers: {summary['total_servers']}")
        print(f"Total capacity: {summary['total_capacity']} connections")
        print(f"Total assigned: {summary['total_assigned']} environments")
        print(f"Overall utilization: {(summary['total_assigned'] / summary['total_capacity']) * 100:.1f}%")
        
        print("\nPer-server breakdown:")
        for server_name, details in summary['server_details'].items():
            print(f"  {server_name.upper()}:")
            print(f"    Address: {details['host']}:{details['port']}")
            print(f"    Load: {details['current_load']}/{details['max_capacity']} ({details['utilization']:.1f}%)")
            print(f"    WebSocket: {details['websocket_url']}")
        
        print("=" * 80 + "\n")
    
    @classmethod
    def from_config(cls, config: Dict) -> 'MultiServerManager':
        """Create MultiServerManager from configuration dictionary.
        
        Parameters
        ----------
        config : Dict
            Configuration dictionary with 'servers' key
            
        Returns
        -------
        MultiServerManager
            Configured server manager
        """
        servers_config = config.get("servers", [])
        
        if not servers_config:
            # Default single server configuration
            servers_config = [{"host": "localhost", "port": 8000, "max_connections": 100}]
            logger.info("No server configuration found, using default single server")
        
        return cls(servers_config)


__all__ = ["MultiServerManager", "ServerConfig"]