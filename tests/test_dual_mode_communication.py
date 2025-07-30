"""Test suite for dual-mode communication system.

This test suite validates the functionality of the dual-mode communication
system including WebSocket and IPC communicators, mode switching, and
integration with the battle environment.
"""

import pytest
import asyncio
import json
import tempfile
import subprocess
import os
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from src.sim.battle_communicator import (
    BattleCommunicator,
    WebSocketCommunicator, 
    IPCCommunicator,
    CommunicatorFactory
)
from src.env.dual_mode_player import (
    DualModeEnvPlayer,
    IPCClientWrapper,
    create_dual_mode_players,
    get_mode_from_config,
    validate_mode_configuration
)


class TestBattleCommunicatorInterface:
    """Test the abstract BattleCommunicator interface."""
    
    def test_abstract_interface(self):
        """Test that BattleCommunicator cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BattleCommunicator()


class TestWebSocketCommunicator:
    """Test WebSocket communication functionality."""
    
    @pytest.fixture
    def ws_communicator(self):
        """Create WebSocket communicator for testing."""
        return WebSocketCommunicator("ws://localhost:8000/showdown/websocket")
    
    def test_initialization(self, ws_communicator):
        """Test WebSocket communicator initialization."""
        assert ws_communicator.url == "ws://localhost:8000/showdown/websocket"
        assert not ws_communicator.connected
        assert ws_communicator.websocket is None
    
    @pytest.mark.asyncio
    async def test_connection_failure(self, ws_communicator):
        """Test WebSocket connection failure handling."""
        # Mock websockets to raise connection error
        with patch('websockets.connect', side_effect=ConnectionRefusedError("Connection refused")):
            with pytest.raises(ConnectionRefusedError):
                await ws_communicator.connect()
        
        assert not ws_communicator.connected
    
    @pytest.mark.asyncio
    async def test_send_without_connection(self, ws_communicator):
        """Test sending message without connection raises error."""
        with pytest.raises(RuntimeError, match="WebSocket not connected"):
            await ws_communicator.send_message({"type": "test"})
    
    @pytest.mark.asyncio
    async def test_receive_without_connection(self, ws_communicator):
        """Test receiving message without connection raises error."""
        with pytest.raises(RuntimeError, match="WebSocket not connected"):
            await ws_communicator.receive_message()
    
    @pytest.mark.asyncio
    async def test_mock_websocket_communication(self, ws_communicator):
        """Test WebSocket communication with mocked websocket."""
        mock_websocket = AsyncMock()
        mock_websocket.closed = False
        
        with patch('websockets.connect', return_value=mock_websocket):
            await ws_communicator.connect()
            assert ws_communicator.connected
            
            # Test sending
            test_message = {"type": "test", "data": "hello"}
            await ws_communicator.send_message(test_message)
            mock_websocket.send.assert_called_once_with(json.dumps(test_message))
            
            # Test receiving
            mock_websocket.recv.return_value = json.dumps({"response": "ok"})
            response = await ws_communicator.receive_message()
            assert response == {"response": "ok"}
            
            # Test is_alive
            assert await ws_communicator.is_alive()
            
            # Test disconnect
            await ws_communicator.disconnect()
            assert not ws_communicator.connected
            mock_websocket.close.assert_called_once()


class TestIPCCommunicator:
    """Test IPC communication functionality."""
    
    @pytest.fixture
    def ipc_communicator(self):
        """Create IPC communicator for testing."""
        return IPCCommunicator("test_script.js")
    
    def test_initialization(self, ipc_communicator):
        """Test IPC communicator initialization."""
        assert ipc_communicator.node_script_path == "test_script.js"
        assert not ipc_communicator.connected
        assert ipc_communicator.process is None
    
    @pytest.mark.asyncio
    async def test_connection_failure(self, ipc_communicator):
        """Test IPC connection failure handling."""
        with patch('asyncio.create_subprocess_exec', side_effect=FileNotFoundError("node not found")):
            with pytest.raises(FileNotFoundError):
                await ipc_communicator.connect()
        
        assert not ipc_communicator.connected
    
    @pytest.mark.asyncio
    async def test_send_without_connection(self, ipc_communicator):
        """Test sending message without connection raises error."""
        with pytest.raises(RuntimeError, match="IPC process not connected"):
            await ipc_communicator.send_message({"type": "test"})
    
    @pytest.mark.asyncio
    async def test_receive_without_connection(self, ipc_communicator):
        """Test receiving message without connection raises error."""
        with pytest.raises(RuntimeError, match="IPC process not connected"):
            await ipc_communicator.receive_message()
    
    @pytest.mark.asyncio
    async def test_mock_ipc_communication(self, ipc_communicator):
        """Test IPC communication with mocked subprocess."""
        mock_process = Mock()
        mock_process.returncode = None
        mock_process.stdin = AsyncMock()
        mock_process.stdout = AsyncMock()
        mock_process.terminate = Mock()
        mock_process.kill = Mock()
        mock_process.wait = AsyncMock(return_value=0)
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            await ipc_communicator.connect()
            assert ipc_communicator.connected
            
            # Test sending
            test_message = {"type": "test", "data": "hello"}
            await ipc_communicator.send_message(test_message)
            expected_data = (json.dumps(test_message) + '\n').encode()
            mock_process.stdin.write.assert_called_once_with(expected_data)
            mock_process.stdin.drain.assert_called_once()
            
            # Test is_alive
            assert await ipc_communicator.is_alive()
            
            # Test disconnect
            await ipc_communicator.disconnect()
            assert not ipc_communicator.connected
            mock_process.terminate.assert_called_once()


class TestCommunicatorFactory:
    """Test communicator factory functionality."""
    
    def test_create_websocket_communicator(self):
        """Test creating WebSocket communicator via factory."""
        communicator = CommunicatorFactory.create_communicator(
            "websocket", 
            url="ws://test:8000/websocket"
        )
        assert isinstance(communicator, WebSocketCommunicator)
        assert communicator.url == "ws://test:8000/websocket"
    
    def test_create_ipc_communicator(self):
        """Test creating IPC communicator via factory."""
        communicator = CommunicatorFactory.create_communicator(
            "ipc",
            node_script_path="test.js"
        )
        assert isinstance(communicator, IPCCommunicator)
        assert communicator.node_script_path == "test.js"
    
    def test_create_invalid_communicator(self):
        """Test creating communicator with invalid mode."""
        with pytest.raises(ValueError, match="Unsupported communication mode"):
            CommunicatorFactory.create_communicator("invalid_mode")


class TestDualModeEnvPlayer:
    """Test dual-mode environment player functionality."""
    
    @pytest.fixture
    def mock_env(self):
        """Create mock environment for testing."""
        env = Mock()
        env._action_queues = {"player_0": asyncio.Queue(), "player_1": asyncio.Queue()}
        env._battle_queues = {"player_0": asyncio.Queue(), "player_1": asyncio.Queue()}
        env.timeout = 30.0
        return env
    
    @pytest.fixture
    def mock_server_config(self):
        """Create mock server configuration."""
        config = Mock()
        config.server_host = "localhost"
        config.server_port = 8000
        return config
    
    def test_local_mode_initialization(self, mock_env):
        """Test player initialization in local mode."""
        with patch('src.env.dual_mode_player.CommunicatorFactory.create_communicator') as mock_factory:
            mock_communicator = Mock()
            mock_factory.return_value = mock_communicator
            
            player = DualModeEnvPlayer(
                env=mock_env,
                player_id="player_0",
                mode="local"
            )
            
            assert player.mode == "local"
            assert player._communicator == mock_communicator
            mock_factory.assert_called_once_with(
                mode="ipc",
                node_script_path="sim/ipc-battle-server.js",
                logger=player._logger
            )
    
    def test_online_mode_initialization(self, mock_env, mock_server_config):
        """Test player initialization in online mode."""
        with patch('src.env.dual_mode_player.CommunicatorFactory.create_communicator') as mock_factory:
            mock_communicator = Mock()
            mock_factory.return_value = mock_communicator
            
            player = DualModeEnvPlayer(
                env=mock_env,
                player_id="player_0",
                mode="online",
                server_configuration=mock_server_config
            )
            
            assert player.mode == "online"
            assert player._communicator == mock_communicator
    
    def test_online_mode_requires_server_config(self, mock_env):
        """Test that online mode requires server configuration."""
        with pytest.raises(ValueError, match="server_configuration required for online mode"):
            DualModeEnvPlayer(
                env=mock_env,
                player_id="player_0", 
                mode="online"
            )


class TestIPCClientWrapper:
    """Test IPC client wrapper functionality."""
    
    @pytest.fixture
    def mock_communicator(self):
        """Create mock communicator for testing."""
        communicator = AsyncMock()
        communicator.is_alive.return_value = True
        return communicator
    
    @pytest.fixture
    def ipc_wrapper(self, mock_communicator):
        """Create IPC client wrapper for testing."""
        import logging
        logger = logging.getLogger(__name__)
        return IPCClientWrapper(mock_communicator, logger)
    
    @pytest.mark.asyncio
    async def test_send_message(self, ipc_wrapper, mock_communicator):
        """Test sending battle message via IPC wrapper."""
        await ipc_wrapper.send_message("move 1", "battle-test-123")
        
        expected_message = {
            "type": "battle_command",
            "battle_id": "battle-test-123",
            "player": "p1",
            "command": "move 1"
        }
        mock_communicator.send_message.assert_called_once_with(expected_message)
    
    @pytest.mark.asyncio
    async def test_create_battle(self, ipc_wrapper, mock_communicator):
        """Test creating battle via IPC wrapper."""
        players = [{"name": "player1"}, {"name": "player2"}]
        await ipc_wrapper.create_battle("battle-123", "gen9randombattle", players)
        
        expected_message = {
            "type": "create_battle",
            "battle_id": "battle-123", 
            "format": "gen9randombattle",
            "players": players,
            "seed": None
        }
        mock_communicator.send_message.assert_called_once_with(expected_message)
    
    @pytest.mark.asyncio
    async def test_get_battle_state(self, ipc_wrapper, mock_communicator):
        """Test getting battle state via IPC wrapper."""
        mock_response = {
            "type": "battle_state",
            "success": True,
            "state": {"turn": 5, "ended": False}
        }
        mock_communicator.receive_message.return_value = mock_response
        
        state = await ipc_wrapper.get_battle_state("battle-123")
        
        expected_request = {
            "type": "get_battle_state",
            "battle_id": "battle-123"
        }
        mock_communicator.send_message.assert_called_once_with(expected_request)
        assert state == {"turn": 5, "ended": False}
    
    @pytest.mark.asyncio
    async def test_ping(self, ipc_wrapper, mock_communicator):
        """Test ping functionality."""
        mock_response = {"type": "pong", "success": True}
        mock_communicator.receive_message.return_value = mock_response
        
        result = await ipc_wrapper.ping()
        
        assert result is True
        mock_communicator.send_message.assert_called_once()
        sent_message = mock_communicator.send_message.call_args[0][0]
        assert sent_message["type"] == "ping"
        assert "timestamp" in sent_message


class TestUtilityFunctions:
    """Test utility functions for mode management."""
    
    def test_get_mode_from_config(self):
        """Test extracting mode from configuration."""
        config_local = {"battle_mode": "local"}
        config_online = {"battle_mode": "online"}
        config_default = {}
        
        assert get_mode_from_config(config_local) == "local"
        assert get_mode_from_config(config_online) == "online"
        assert get_mode_from_config(config_default) == "local"  # default
    
    def test_validate_mode_configuration_online(self):
        """Test validation of online mode configuration."""
        valid_config = {
            "pokemon_showdown": {
                "servers": [{"host": "localhost", "port": 8000}]
            }
        }
        invalid_config_no_showdown = {}
        invalid_config_no_servers = {"pokemon_showdown": {}}
        
        # Valid configuration should not raise
        validate_mode_configuration("online", valid_config)
        
        # Invalid configurations should raise
        with pytest.raises(ValueError, match="Online mode requires 'pokemon_showdown' configuration"):
            validate_mode_configuration("online", invalid_config_no_showdown)
        
        with pytest.raises(ValueError, match="Online mode requires server configuration"):
            validate_mode_configuration("online", invalid_config_no_servers)
    
    def test_validate_mode_configuration_local(self):
        """Test validation of local mode configuration."""
        valid_config = {"local_mode": {"max_processes": 5}}
        invalid_config = {"local_mode": {"max_processes": 0}}
        
        # Valid configuration should not raise
        validate_mode_configuration("local", valid_config)
        
        # Invalid configuration should raise
        with pytest.raises(ValueError, match="local_mode.max_processes must be at least 1"):
            validate_mode_configuration("local", invalid_config)
    
    def test_validate_mode_configuration_invalid_mode(self):
        """Test validation with invalid mode."""
        with pytest.raises(ValueError, match="Unsupported mode: invalid"):
            validate_mode_configuration("invalid", {})


# Integration tests (marked as slow)
@pytest.mark.slow
class TestIntegration:
    """Integration tests for dual-mode communication system."""
    
    @pytest.mark.asyncio
    async def test_ipc_communicator_with_real_node(self):
        """Test IPC communicator with actual Node.js process."""
        # Create a simple test Node.js script
        test_script = """
        const readline = require('readline');
        const rl = readline.createInterface({
            input: process.stdin,
            output: process.stdout
        });
        
        rl.on('line', (line) => {
            try {
                const message = JSON.parse(line);
                if (message.type === 'ping') {
                    console.log(JSON.stringify({type: 'pong', success: true}));
                }
            } catch (e) {
                console.log(JSON.stringify({type: 'error', message: e.message}));
            }
        });
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
            f.write(test_script)
            script_path = f.name
        
        try:
            # Test that node is available
            subprocess.run(['node', '--version'], check=True, capture_output=True)
            
            communicator = IPCCommunicator(script_path)
            
            await communicator.connect()
            assert await communicator.is_alive()
            
            # Test ping
            ping_message = {"type": "ping"}
            await communicator.send_message(ping_message)
            
            response = await asyncio.wait_for(
                communicator.receive_message(),
                timeout=5.0
            )
            
            assert response["type"] == "pong"
            assert response["success"] is True
            
            await communicator.disconnect()
            assert not await communicator.is_alive()
            
        except subprocess.CalledProcessError:
            pytest.skip("Node.js not available for integration test")
        except FileNotFoundError:
            pytest.skip("Node.js not found in PATH")
        finally:
            # Clean up test script
            if os.path.exists(script_path):
                os.unlink(script_path)


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])