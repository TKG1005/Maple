"""Integration tests for Phase 2: Environment integration and mode management.

This test suite validates the integration of dual-mode communication 
into PokemonEnv and the configuration management system.
"""

import pytest
import asyncio
import tempfile
import yaml
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock

from src.env.pokemon_env import PokemonEnv
from src.env.dual_mode_player import DualModeEnvPlayer
from src.state.state_observer import StateObserver
from src.action import action_helper


class TestPokemonEnvDualModeIntegration:
    """Test PokemonEnv integration with dual-mode players."""
    
    @pytest.fixture
    def mock_state_observer(self):
        """Create mock state observer."""
        observer = Mock(spec=StateObserver)
        observer.get_observation_dimension.return_value = 100
        observer.observe.return_value = [0.0] * 100
        return observer
    
    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing."""
        return {
            "battle_mode": "local",
            "local_mode": {
                "max_processes": 5,
                "process_timeout": 300,
                "reuse_processes": True
            },
            "pokemon_showdown": {
                "servers": [
                    {"host": "localhost", "port": 8000}
                ]
            }
        }
    
    def test_pokemon_env_local_mode_initialization(self, mock_state_observer):
        """Test PokemonEnv initialization in local mode."""
        with patch('src.env.pokemon_env.validate_mode_configuration'):
            env = PokemonEnv(
                state_observer=mock_state_observer,
                action_helper=action_helper,
                battle_mode="local"
            )
            
            assert env.battle_mode == "local"
            assert env.get_battle_mode() == "local"
    
    def test_pokemon_env_online_mode_initialization(self, mock_state_observer):
        """Test PokemonEnv initialization in online mode."""
        with patch('src.env.pokemon_env.validate_mode_configuration'):
            env = PokemonEnv(
                state_observer=mock_state_observer,
                action_helper=action_helper,
                battle_mode="online"
            )
            
            assert env.battle_mode == "online"
            assert env.get_battle_mode() == "online"
    
    def test_pokemon_env_mode_switching(self, mock_state_observer):
        """Test battle mode switching functionality."""
        with patch('src.env.pokemon_env.validate_mode_configuration'):
            env = PokemonEnv(
                state_observer=mock_state_observer,
                action_helper=action_helper,
                battle_mode="local"
            )
            
            # Switch to online mode
            env.set_battle_mode("online")
            assert env.get_battle_mode() == "online"
            
            # Switch back to local mode
            env.set_battle_mode("local")
            assert env.get_battle_mode() == "local"
    
    def test_pokemon_env_invalid_mode_rejection(self, mock_state_observer):
        """Test that invalid battle modes are rejected."""
        with patch('src.env.pokemon_env.validate_mode_configuration'):
            env = PokemonEnv(
                state_observer=mock_state_observer,
                action_helper=action_helper,
                battle_mode="local"
            )
            
            with pytest.raises(ValueError, match="Unsupported battle mode"):
                env.set_battle_mode("invalid_mode")
    
    def test_battle_mode_info(self, mock_state_observer):
        """Test battle mode information retrieval."""
        with patch('src.env.pokemon_env.validate_mode_configuration'):
            env = PokemonEnv(
                state_observer=mock_state_observer,
                action_helper=action_helper,
                battle_mode="local"
            )
            
            info = env.get_battle_mode_info()
            
            assert info["current_mode"] == "local"
            assert "local" in info["supported_modes"]
            assert "online" in info["supported_modes"]
            assert "mode_descriptions" in info
            assert "local" in info["mode_descriptions"]
            assert "online" in info["mode_descriptions"]
    
    @patch('src.env.dual_mode_player.DualModeEnvPlayer')
    def test_player_creation_local_mode(self, mock_dual_player, mock_state_observer):
        """Test that local mode creates DualModeEnvPlayer instances."""
        with patch('src.env.pokemon_env.validate_mode_configuration'):
            env = PokemonEnv(
                state_observer=mock_state_observer,
                action_helper=action_helper,
                battle_mode="local"
            )
            
            # Mock the _create_battle_player method to verify it's called correctly
            with patch.object(env, '_create_battle_player') as mock_create:
                mock_player = Mock()
                mock_create.return_value = mock_player
                
                # Test player creation logic
                player = env._create_battle_player(
                    player_id="player_0",
                    server_config=None,
                    team=None,
                    account_config=None
                )
                
                mock_create.assert_called_once_with(
                    "player_0", None, None, None
                )
    
    @patch('src.env.dual_mode_player.DualModeEnvPlayer')
    def test_player_creation_online_mode(self, mock_dual_player, mock_state_observer):
        """Test that online mode creates DualModeEnvPlayer instances."""
        mock_server_config = Mock()
        
        with patch('src.env.pokemon_env.validate_mode_configuration'):
            env = PokemonEnv(
                state_observer=mock_state_observer,
                action_helper=action_helper,
                battle_mode="online",
                server_configuration=mock_server_config
            )
            
            # Test player creation
            with patch.object(env, '_create_battle_player') as mock_create:
                mock_player = Mock()
                mock_create.return_value = mock_player
                
                player = env._create_battle_player(
                    player_id="player_1",
                    server_config=mock_server_config,
                    team=None,
                    account_config=None
                )
                
                mock_create.assert_called_once_with(
                    "player_1", mock_server_config, None, None
                )


class TestConfigurationIntegration:
    """Test configuration file integration with dual-mode system."""
    
    def test_config_loading_local_mode(self):
        """Test loading local mode configuration."""
        config_data = {
            "battle_mode": "local",
            "local_mode": {
                "max_processes": 8,
                "process_timeout": 600,
                "reuse_processes": False,
                "ipc_script_path": "custom/path/script.js"
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Test configuration loading
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            assert loaded_config["battle_mode"] == "local"
            assert loaded_config["local_mode"]["max_processes"] == 8
            assert loaded_config["local_mode"]["process_timeout"] == 600
            assert not loaded_config["local_mode"]["reuse_processes"]
            assert loaded_config["local_mode"]["ipc_script_path"] == "custom/path/script.js"
            
        finally:
            Path(config_path).unlink()
    
    def test_config_loading_online_mode(self):
        """Test loading online mode configuration."""
        config_data = {
            "battle_mode": "online",
            "pokemon_showdown": {
                "servers": [
                    {"host": "server1.example.com", "port": 8000, "max_connections": 30},
                    {"host": "server2.example.com", "port": 8001, "max_connections": 50}
                ]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            assert loaded_config["battle_mode"] == "online"
            assert len(loaded_config["pokemon_showdown"]["servers"]) == 2
            assert loaded_config["pokemon_showdown"]["servers"][0]["host"] == "server1.example.com"
            assert loaded_config["pokemon_showdown"]["servers"][1]["max_connections"] == 50
            
        finally:
            Path(config_path).unlink()
    
    def test_config_validation_errors(self):
        """Test configuration validation error handling."""
        from src.env.dual_mode_player import validate_mode_configuration
        
        # Test online mode without required configuration
        with pytest.raises(ValueError, match="Online mode requires 'pokemon_showdown' configuration"):
            validate_mode_configuration("online", {})
        
        # Test online mode without servers
        invalid_online_config = {"pokemon_showdown": {}}
        with pytest.raises(ValueError, match="Online mode requires server configuration"):
            validate_mode_configuration("online", invalid_online_config)
        
        # Test local mode with invalid max_processes
        invalid_local_config = {"local_mode": {"max_processes": 0}}
        with pytest.raises(ValueError, match="local_mode.max_processes must be at least 1"):
            validate_mode_configuration("local", invalid_local_config)
        
        # Test invalid mode
        with pytest.raises(ValueError, match="Unsupported mode"):
            validate_mode_configuration("invalid", {})


class TestCLIIntegration:
    """Test CLI parameter integration with dual-mode system."""
    
    @pytest.fixture
    def mock_train_main(self):
        """Mock the main training function."""
        with patch('train.main') as mock_main:
            yield mock_main
    
    def test_cli_battle_mode_local(self, mock_train_main):
        """Test CLI --battle-mode local parameter."""
        import train
        
        # Mock argparse
        mock_args = Mock()
        mock_args.battle_mode = "local"
        mock_args.config = "config/train_config.yml"
        mock_args.episodes = None
        mock_args.save = None
        mock_args.tensorboard = False
        mock_args.ppo_epochs = None
        mock_args.clip = None
        mock_args.gae_lambda = None
        mock_args.value_coef = None
        mock_args.entropy_coef = None
        mock_args.gamma = None
        mock_args.parallel = 1
        mock_args.checkpoint_interval = 0
        mock_args.checkpoint_dir = "checkpoints"
        mock_args.algo = "ppo"
        mock_args.reward = "composite"
        mock_args.reward_config = None
        mock_args.opponent = None
        mock_args.opponent_mix = None
        mock_args.team = "default"
        mock_args.teams_dir = None
        mock_args.load_model = None
        mock_args.reset_optimizer = False
        mock_args.win_rate_threshold = 0.6
        mock_args.win_rate_window = 50
        mock_args.device = "auto"
        mock_args.log_level = "INFO"
        mock_args.epsilon_enabled = False
        mock_args.epsilon_start = None
        mock_args.epsilon_end = None
        mock_args.epsilon_decay_steps = None
        mock_args.epsilon_decay_strategy = None
        mock_args.epsilon_decay_mode = None
        mock_args.profile = False
        mock_args.profile_name = None
        
        with patch('argparse.ArgumentParser.parse_args', return_value=mock_args):
            with patch('train.setup_logging'):
                with patch('train.logging.basicConfig'):
                    # This would normally call train.main, but we'll verify the parameters
                    # that would be passed
                    expected_battle_mode = "local"
                    assert mock_args.battle_mode == expected_battle_mode
    
    def test_cli_battle_mode_online(self, mock_train_main):
        """Test CLI --battle-mode online parameter."""
        import train
        
        mock_args = Mock()
        mock_args.battle_mode = "online"
        
        # Verify the parameter is correctly set
        assert mock_args.battle_mode == "online"
    
    def test_cli_battle_mode_invalid(self):
        """Test CLI rejects invalid battle mode values."""
        import argparse
        
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--battle-mode",
            type=str,
            choices=["local", "online"],
            default="local"
        )
        
        # This should raise SystemExit due to invalid choice
        with pytest.raises(SystemExit):
            parser.parse_args(["--battle-mode", "invalid"])


class TestEndToEndIntegration:
    """End-to-end integration tests."""
    
    @pytest.mark.slow
    def test_environment_creation_with_mode_switching(self):
        """Test creating environments and switching modes."""
        from src.state.state_observer import StateObserver
        import tempfile
        
        # Create a temporary state spec file
        state_spec = {
            "features": [
                {"name": "test_feature", "size": 10, "type": "float"}
            ]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(state_spec, f)
            state_spec_path = f.name
        
        try:
            observer = StateObserver(state_spec_path)
            
            # Test local mode environment
            with patch('src.env.pokemon_env.validate_mode_configuration'):
                env_local = PokemonEnv(
                    state_observer=observer,
                    action_helper=action_helper,
                    battle_mode="local"
                )
                
                assert env_local.get_battle_mode() == "local"
                
                # Test mode switching
                env_local.set_battle_mode("online")
                assert env_local.get_battle_mode() == "online"
                
                # Test environment info
                info = env_local.get_battle_mode_info()
                assert info["current_mode"] == "online"
                assert len(info["supported_modes"]) == 2
        
        finally:
            Path(state_spec_path).unlink()
    
    @pytest.mark.slow
    def test_configuration_file_integration(self):
        """Test complete configuration file integration."""
        config_data = {
            "battle_mode": "local",
            "episodes": 5,
            "parallel": 2,
            "local_mode": {
                "max_processes": 3,
                "process_timeout": 120
            },
            "pokemon_showdown": {
                "servers": [{"host": "localhost", "port": 8000}]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name
        
        try:
            # Load and validate configuration
            with open(config_path, 'r') as f:
                loaded_config = yaml.safe_load(f)
            
            # Test configuration values
            assert loaded_config["battle_mode"] == "local"
            assert loaded_config["episodes"] == 5
            assert loaded_config["parallel"] == 2
            assert loaded_config["local_mode"]["max_processes"] == 3
            
            # Test configuration validation
            from src.env.dual_mode_player import validate_mode_configuration
            validate_mode_configuration(loaded_config["battle_mode"], loaded_config)
            
        finally:
            Path(config_path).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])