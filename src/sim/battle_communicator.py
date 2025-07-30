"""Battle communication interface abstraction for dual-mode support.

This module provides the abstract interface for battle communication, supporting
both WebSocket (online) and IPC (local) modes for Pokemon Showdown integration.
"""

from __future__ import annotations

import json
import asyncio
import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BattleCommunicator(ABC):
    """Abstract interface for battle communication.
    
    This interface abstracts the communication layer between the Python environment
    and Pokemon Showdown, allowing for both WebSocket (online) and IPC (local) modes.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.connected = False
    
    @abstractmethod
    async def connect(self) -> None:
        """Establish connection to the battle server."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the battle server."""
        pass
    
    @abstractmethod
    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send a message to the battle server.
        
        Args:
            message: Dictionary containing the message data in JSON format
        """
        pass
    
    @abstractmethod
    async def receive_message(self) -> Dict[str, Any]:
        """Receive a message from the battle server.
        
        Returns:
            Dictionary containing the received message data
        """
        pass
    
    @abstractmethod
    async def is_alive(self) -> bool:
        """Check if the connection is alive and responsive."""
        pass
    
    async def save_battle_state(self, battle_id: str) -> Dict[str, Any]:
        """Save current battle state for later restoration.
        
        Args:
            battle_id: Identifier of the battle to save
            
        Returns:
            Dictionary containing the saved state information
        """
        # Default implementation - subclasses can override for optimized behavior
        message = {
            "type": "save_battle_state",
            "battle_id": battle_id
        }
        await self.send_message(message)
        response = await self.receive_message()
        
        if response.get("type") != "battle_state_saved" or not response.get("success"):
            raise RuntimeError(f"Failed to save battle state: {response}")
        
        return response
    
    async def restore_battle_state(self, battle_id: str, state_data: Dict[str, Any]) -> bool:
        """Restore battle state from saved data.
        
        Args:
            battle_id: Identifier of the battle to restore
            state_data: Previously saved state data
            
        Returns:
            True if restoration was successful
        """
        # Default implementation - subclasses can override for optimized behavior
        message = {
            "type": "restore_battle_state",
            "battle_id": battle_id,
            "state_data": state_data
        }
        await self.send_message(message)
        response = await self.receive_message()
        
        return response.get("type") == "battle_state_restored" and response.get("success", False)
    
    async def get_battle_state(self, battle_id: str) -> Dict[str, Any]:
        """Get current battle state without saving.
        
        Args:
            battle_id: Identifier of the battle
            
        Returns:
            Dictionary containing current battle state
        """
        # Default implementation - subclasses can override for optimized behavior
        message = {
            "type": "get_battle_state",
            "battle_id": battle_id
        }
        await self.send_message(message)
        response = await self.receive_message()
        
        if response.get("type") != "battle_state" or not response.get("success"):
            raise RuntimeError(f"Failed to get battle state: {response}")
        
        return response.get("state", {})


class WebSocketCommunicator(BattleCommunicator):
    """WebSocket-based communicator for online battles.
    
    This implementation maintains compatibility with the existing Pokemon Showdown
    WebSocket protocol for online battles and tournaments.
    """
    
    def __init__(self, url: str, **kwargs):
        super().__init__(**kwargs)
        self.url = url
        self.websocket = None
        self._connection_lock = asyncio.Lock()
    
    async def connect(self) -> None:
        """Connect to Pokemon Showdown WebSocket server."""
        async with self._connection_lock:
            if self.connected:
                return
                
            try:
                import websockets
                self.websocket = await websockets.connect(self.url)
                self.connected = True
                self.logger.info(f"Connected to WebSocket server at {self.url}")
            except Exception as e:
                self.logger.error(f"Failed to connect to WebSocket server: {e}")
                raise
    
    async def disconnect(self) -> None:
        """Disconnect from WebSocket server."""
        async with self._connection_lock:
            if self.websocket and self.connected:
                await self.websocket.close()
                self.connected = False
                self.logger.info("Disconnected from WebSocket server")
    
    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send JSON message via WebSocket."""
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        try:
            json_data = json.dumps(message)
            await self.websocket.send(json_data)
            self.logger.debug(f"Sent WebSocket message: {json_data}")
        except Exception as e:
            self.logger.error(f"Failed to send WebSocket message: {e}")
            raise
    
    async def receive_message(self) -> Dict[str, Any]:
        """Receive JSON message via WebSocket."""
        if not self.connected or not self.websocket:
            raise RuntimeError("WebSocket not connected")
        
        try:
            raw_data = await self.websocket.recv()
            message = json.loads(raw_data)
            self.logger.debug(f"Received WebSocket message: {raw_data}")
            return message
        except Exception as e:
            self.logger.error(f"Failed to receive WebSocket message: {e}")
            raise
    
    async def is_alive(self) -> bool:
        """Check WebSocket connection status."""
        return self.connected and self.websocket and not self.websocket.closed


class IPCCommunicator(BattleCommunicator):
    """IPC-based communicator for local high-speed battles.
    
    This implementation uses subprocess communication with Node.js for
    local battles with minimal latency compared to WebSocket communication.
    """
    
    def __init__(self, node_script_path: str = "sim/ipc-battle-server.js", **kwargs):
        super().__init__(**kwargs)
        self.node_script_path = node_script_path
        self.process = None
        self._connection_lock = asyncio.Lock()
        self._message_queue = asyncio.Queue()
        self._reader_task = None
    
    async def connect(self) -> None:
        """Start Node.js subprocess for IPC communication."""
        async with self._connection_lock:
            if self.connected:
                return
            
            try:
                self.logger.info(f"🚀 Starting Node.js IPC process...")
                self.logger.info(f"📄 Node.js script: {self.node_script_path}")
                self.logger.info(f"📁 Working directory: pokemon-showdown")
                self.logger.info(f"📍 Current directory: {os.getcwd()}")
                
                # Check if script file exists
                if not os.path.exists(self.node_script_path):
                    raise FileNotFoundError(f"Node.js script not found: {self.node_script_path}")
                
                # Check if Pokemon Showdown directory exists
                if not os.path.exists('pokemon-showdown'):
                    raise FileNotFoundError("Pokemon Showdown directory not found: pokemon-showdown")
                
                # Start Node.js subprocess with IPC (run from pokemon-showdown directory)
                self.process = await asyncio.create_subprocess_exec(
                    'node', self.node_script_path,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd='pokemon-showdown'  # Run from pokemon-showdown directory where dist/ is located
                )
                
                self.logger.info(f"✅ Node.js process started with PID: {self.process.pid}")
                
                # Start background task to read responses
                self._reader_task = asyncio.create_task(self._read_responses())
                
                self.connected = True
                self.logger.info(f"🔗 IPC connection established: {self.node_script_path}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to start Node.js IPC process: {type(e).__name__}: {e}")
                await self._cleanup_process()
                raise
    
    async def disconnect(self) -> None:
        """Stop Node.js subprocess."""
        async with self._connection_lock:
            if not self.connected:
                return
            
            await self._cleanup_process()
            self.connected = False
            self.logger.info("Stopped Node.js IPC process")
    
    async def _cleanup_process(self) -> None:
        """Clean up subprocess and associated resources."""
        # Cancel reader task
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass
        
        # Terminate process
        if self.process:
            if self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.logger.warning("Process did not terminate gracefully, killing...")
                    self.process.kill()
                    await self.process.wait()
            self.process = None
    
    async def _read_responses(self) -> None:
        """Background task to read responses from Node.js process."""
        if not self.process or not self.process.stdout:
            return
        
        try:
            while self.connected and self.process.returncode is None:
                line = await self.process.stdout.readline()
                if not line:
                    break
                
                try:
                    # Parse JSON response
                    response_data = line.decode().strip()
                    if response_data:
                        message = json.loads(response_data)
                        await self._message_queue.put(message)
                        self.logger.debug(f"Received IPC message: {response_data}")
                except json.JSONDecodeError as e:
                    self.logger.error(f"Failed to parse IPC response: {e}")
                except Exception as e:
                    self.logger.error(f"Error processing IPC response: {e}")
                    
        except Exception as e:
            self.logger.error(f"Error in IPC reader task: {e}")
        finally:
            self.logger.debug("IPC reader task finished")
    
    async def send_message(self, message: Dict[str, Any]) -> None:
        """Send JSON message via IPC."""
        if not self.connected or not self.process or not self.process.stdin:
            raise RuntimeError("IPC process not connected")
        
        try:
            # Send JSON message with newline delimiter
            json_data = json.dumps(message) + '\n'
            self.process.stdin.write(json_data.encode())
            await self.process.stdin.drain()
            self.logger.debug(f"Sent IPC message: {json_data.strip()}")
        except Exception as e:
            self.logger.error(f"Failed to send IPC message: {e}")
            raise
    
    async def receive_message(self) -> Dict[str, Any]:
        """Receive JSON message via IPC."""
        if not self.connected:
            raise RuntimeError("IPC process not connected")
        
        try:
            # Wait for message from background reader
            message = await asyncio.wait_for(self._message_queue.get(), timeout=30.0)
            return message
        except asyncio.TimeoutError:
            self.logger.error("Timeout waiting for IPC message")
            raise
        except Exception as e:
            self.logger.error(f"Failed to receive IPC message: {e}")
            raise
    
    async def is_alive(self) -> bool:
        """Check if Node.js process is running."""
        return (self.connected and 
                self.process is not None and 
                self.process.returncode is None)


class CommunicatorFactory:
    """Factory class for creating appropriate communicator instances."""
    
    @staticmethod
    def create_communicator(mode: str, **kwargs) -> BattleCommunicator:
        """Create a communicator instance based on the specified mode.
        
        Args:
            mode: Communication mode ("websocket" or "ipc")
            **kwargs: Additional arguments for the communicator
        
        Returns:
            Appropriate BattleCommunicator instance
        
        Raises:
            ValueError: If mode is not supported
        """
        if mode == "websocket":
            url = kwargs.pop("url", "ws://localhost:8000/showdown/websocket")
            return WebSocketCommunicator(url=url, **kwargs)
        elif mode == "ipc":
            script_path = kwargs.pop("node_script_path", "sim/ipc-battle-server.js")
            return IPCCommunicator(node_script_path=script_path, **kwargs)
        else:
            raise ValueError(f"Unsupported communication mode: {mode}")