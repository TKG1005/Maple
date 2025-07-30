"""Comprehensive test suite for Phase 3 battle state serialization.

This test suite validates the battle state serialization system including:
- BattleStateSerializer interface and implementations
- BattleStateManager file operations
- PokemonEnv integration
- JSON format consistency
- Error handling and edge cases
"""

import json
import pytest
import tempfile
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.sim.battle_state_serializer import (
    BattleState, PokemonState, PlayerState,
    BattleStateSerializer, PokeEnvBattleSerializer, BattleStateManager
)
from src.sim.battle_communicator import BattleCommunicator


class TestBattleStateDataStructures:
    """Test battle state data structures and serialization."""
    
    def test_pokemon_state_creation(self):
        """Test PokemonState creation and serialization."""
        pokemon_state = PokemonState(
            species="Pikachu",
            nickname="Pika",
            level=50,
            gender="M",
            hp=100,
            max_hp=100,
            status=None,
            stats={"hp": 100, "atk": 75, "def": 60},
            base_stats={"hp": 35, "atk": 55, "def": 40},
            moves=[
                {"id": "thundershock", "name": "Thunder Shock", "pp": 30, "max_pp": 30}
            ],
            ability="Static",
            item="Light Ball",
            types=["Electric"],
            boosts={},
            volatile_status=[],
            position=0,
            active=True
        )
        
        assert pokemon_state.species == "Pikachu"
        assert pokemon_state.nickname == "Pika"
        assert pokemon_state.level == 50
        assert pokemon_state.active is True
        assert len(pokemon_state.moves) == 1
        assert pokemon_state.moves[0]["id"] == "thundershock"
    
    def test_player_state_creation(self):
        """Test PlayerState creation with team."""
        pokemon = PokemonState(
            species="Charizard", nickname=None, level=50, gender=None,
            hp=78, max_hp=100, status="burn", stats={}, base_stats={},
            moves=[], ability=None, item=None, types=["Fire", "Flying"],
            boosts={"atk": 1}, volatile_status=["substitute"], position=0, active=True
        )
        
        player_state = PlayerState(
            player_id="p1",
            username="Trainer1",
            team=[pokemon],
            active_pokemon=0,
            side_conditions={"reflect": 3},
            last_move="flamethrower",
            can_switch=[False, True, True],
            can_dynamax=True,
            dynamax_turns_left=0
        )
        
        assert player_state.player_id == "p1"
        assert len(player_state.team) == 1
        assert player_state.team[0].species == "Charizard"
        assert player_state.side_conditions["reflect"] == 3
        assert player_state.can_switch == [False, True, True]
    
    def test_battle_state_serialization(self):
        """Test complete BattleState serialization to/from dict."""
        # Create mock Pokemon and players
        pokemon1 = PokemonState(
            species="Pikachu", nickname=None, level=50, gender="M",
            hp=85, max_hp=100, status=None, stats={}, base_stats={},
            moves=[], ability="Static", item=None, types=["Electric"],
            boosts={}, volatile_status=[], position=0, active=True
        )
        
        pokemon2 = PokemonState(
            species="Charizard", nickname=None, level=50, gender="M",
            hp=78, max_hp=100, status="burn", stats={}, base_stats={},
            moves=[], ability="Blaze", item=None, types=["Fire", "Flying"],
            boosts={}, volatile_status=[], position=0, active=True
        )
        
        player1 = PlayerState(
            player_id="p1", username="Player1", team=[pokemon1],
            active_pokemon=0, side_conditions={}, last_move=None,
            can_switch=[False], can_dynamax=True, dynamax_turns_left=0
        )
        
        player2 = PlayerState(
            player_id="p2", username="Player2", team=[pokemon2],
            active_pokemon=0, side_conditions={}, last_move=None,
            can_switch=[False], can_dynamax=True, dynamax_turns_left=0
        )
        
        battle_state = BattleState(
            battle_id="battle-test-123",
            format_id="gen9randombattle",
            turn=5,
            weather="sun",
            weather_turns_left=3,
            terrain=None,
            terrain_turns_left=0,
            field_effects={},
            players=[player1, player2],
            battle_log=["Turn 1", "Turn 2", "Turn 3", "Turn 4", "Turn 5"],
            timestamp="2025-07-30T12:00:00",
            metadata={"test": True}
        )
        
        # Test serialization
        state_dict = battle_state.to_dict()
        assert state_dict["battle_id"] == "battle-test-123"
        assert state_dict["turn"] == 5
        assert state_dict["weather"] == "sun"
        assert len(state_dict["players"]) == 2
        assert len(state_dict["battle_log"]) == 5
        
        # Test deserialization
        restored_state = BattleState.from_dict(state_dict)
        assert restored_state.battle_id == "battle-test-123"
        assert restored_state.turn == 5
        assert restored_state.weather == "sun"
        assert len(restored_state.players) == 2
        assert restored_state.players[0].team[0].species == "Pikachu"
        assert restored_state.players[1].team[0].species == "Charizard"


class TestPokeEnvBattleSerializer:
    """Test PokeEnv-specific battle serializer."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.serializer = PokeEnvBattleSerializer()
    
    def create_mock_battle(self):
        """Create a mock poke-env Battle object."""
        # Create mock Pokemon
        mock_pokemon1 = Mock()
        mock_pokemon1.species = "Pikachu"
        mock_pokemon1.current_hp_fraction = 0.85
        mock_pokemon1.max_hp = 100
        mock_pokemon1.status = None
        mock_pokemon1.moves = []
        mock_pokemon1.types = [Mock(name="Electric")]
        mock_pokemon1.types[0].name = "Electric"
        
        mock_pokemon2 = Mock()
        mock_pokemon2.species = "Charizard"
        mock_pokemon2.current_hp_fraction = 0.78
        mock_pokemon2.max_hp = 100
        mock_pokemon2.status = Mock(name="burn")
        mock_pokemon2.status.name = "burn"
        mock_pokemon2.moves = []
        mock_pokemon2.types = [Mock(name="Fire"), Mock(name="Flying")]
        mock_pokemon2.types[0].name = "Fire"
        mock_pokemon2.types[1].name = "Flying"
        
        # Create mock players
        mock_p1 = Mock()
        mock_p1.team = {"pokemon1": mock_pokemon1}
        mock_p1.active_pokemon = mock_pokemon1
        mock_p1.username = "Player1"
        
        mock_p2 = Mock()
        mock_p2.team = {"pokemon1": mock_pokemon2}
        mock_p2.active_pokemon = mock_pokemon2
        mock_p2.username = "Player2"
        
        # Create mock battle
        mock_battle = Mock()
        mock_battle.battle_tag = "battle-test-456"
        mock_battle.turn = 3
        mock_battle.weather = Mock(name="sun")
        mock_battle.weather.name = "sun"
        mock_battle.terrain = None
        mock_battle.p1 = mock_p1
        mock_battle.p2 = mock_p2
        mock_battle.history = ["Turn 1", "Turn 2", "Turn 3"]
        
        return mock_battle
    
    def test_serialize_mock_battle(self):
        """Test serialization of mock battle object."""
        mock_battle = self.create_mock_battle()
        
        # Test serialization
        battle_state = self.serializer.serialize_state(mock_battle)
        
        assert battle_state.battle_id == "battle-test-456"
        assert battle_state.turn == 3
        assert battle_state.weather == "sun"
        assert len(battle_state.players) == 2
        assert battle_state.players[0].username == "Player1"
        assert battle_state.players[1].username == "Player2"
        
        # Check Pokemon data
        p1_pokemon = battle_state.players[0].team[0]
        assert p1_pokemon.species == "Pikachu"
        assert p1_pokemon.hp == 85  # 0.85 * 100
        
        p2_pokemon = battle_state.players[1].team[0]
        assert p2_pokemon.species == "Charizard"
        assert p2_pokemon.status == "burn"
        assert p2_pokemon.types == ["Fire", "Flying"]
    
    def test_validate_state_valid(self):
        """Test state validation with valid state."""
        pokemon = PokemonState(
            species="Pikachu", nickname=None, level=50, gender=None,
            hp=100, max_hp=100, status=None, stats={}, base_stats={},
            moves=[], ability=None, item=None, types=[], boosts={},
            volatile_status=[], position=0, active=True
        )
        
        player = PlayerState(
            player_id="p1", username="Player1", team=[pokemon],
            active_pokemon=0, side_conditions={}, last_move=None,
            can_switch=[False], can_dynamax=True, dynamax_turns_left=0
        )
        
        battle_state = BattleState(
            battle_id="test-battle", format_id="gen9randombattle", turn=1,
            weather=None, weather_turns_left=0, terrain=None, terrain_turns_left=0,
            field_effects={}, players=[player], battle_log=[], timestamp="2025-07-30T12:00:00",
            metadata={}
        )
        
        # Should fail - only one player
        assert not self.serializer.validate_state(battle_state)
        
        # Add second player
        player2 = PlayerState(
            player_id="p2", username="Player2", team=[pokemon],
            active_pokemon=0, side_conditions={}, last_move=None,
            can_switch=[False], can_dynamax=True, dynamax_turns_left=0
        )
        battle_state.players.append(player2)
        
        # Should pass now
        assert self.serializer.validate_state(battle_state)
    
    def test_validate_state_invalid(self):
        """Test state validation with invalid states."""
        # Missing battle_id
        invalid_state = BattleState(
            battle_id="", format_id="gen9randombattle", turn=1,
            weather=None, weather_turns_left=0, terrain=None, terrain_turns_left=0,
            field_effects={}, players=[], battle_log=[], timestamp="2025-07-30T12:00:00",
            metadata={}
        )
        assert not self.serializer.validate_state(invalid_state)
        
        # No players
        invalid_state.battle_id = "test-battle"
        assert not self.serializer.validate_state(invalid_state)


class TestBattleStateManager:
    """Test battle state file management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.serializer = Mock(spec=BattleStateSerializer)
        self.manager = BattleStateManager(
            serializer=self.serializer,
            storage_dir=self.temp_dir
        )
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_sample_battle_state(self):
        """Create a sample battle state for testing."""
        pokemon = PokemonState(
            species="Pikachu", nickname=None, level=50, gender=None,
            hp=100, max_hp=100, status=None, stats={}, base_stats={},
            moves=[], ability=None, item=None, types=[], boosts={},
            volatile_status=[], position=0, active=True
        )
        
        player = PlayerState(
            player_id="p1", username="TestPlayer", team=[pokemon],
            active_pokemon=0, side_conditions={}, last_move=None,
            can_switch=[False], can_dynamax=True, dynamax_turns_left=0
        )
        
        return BattleState(
            battle_id="test-battle", format_id="gen9randombattle", turn=1,
            weather=None, weather_turns_left=0, terrain=None, terrain_turns_left=0,
            field_effects={}, players=[player, player], battle_log=[],
            timestamp="2025-07-30T12:00:00", metadata={}
        )
    
    def test_save_and_load_state(self):
        """Test saving and loading battle states."""
        # Create mock battle and expected state
        mock_battle = Mock()
        expected_state = self.create_sample_battle_state()
        
        # Configure serializer mock
        self.serializer.serialize_state.return_value = expected_state
        self.serializer.validate_state.return_value = True
        
        # Test save
        filepath = self.manager.save_state(mock_battle, "test_battle.json")
        assert Path(filepath).exists()
        assert "test_battle.json" in filepath
        
        # Test load
        loaded_state = self.manager.load_state(filepath)
        assert loaded_state.battle_id == "test-battle"
        assert loaded_state.format_id == "gen9randombattle"
        
        # Verify serializer was called
        self.serializer.serialize_state.assert_called_once_with(mock_battle)
        self.serializer.validate_state.assert_called_once()
    
    def test_auto_generate_filename(self):
        """Test automatic filename generation."""
        mock_battle = Mock()
        expected_state = self.create_sample_battle_state()
        
        self.serializer.serialize_state.return_value = expected_state
        
        # Save without filename
        filepath = self.manager.save_state(mock_battle)
        
        # Check that filename was generated
        filename = Path(filepath).name
        assert filename.startswith("test-battle_")
        assert filename.endswith(".json")
    
    def test_list_saved_states(self):
        """Test listing saved states."""
        # Initially empty
        states = self.manager.list_saved_states()
        assert len(states) == 0
        
        # Create some test files
        (Path(self.temp_dir) / "battle1_123.json").write_text('{"test": 1}')
        (Path(self.temp_dir) / "battle2_456.json").write_text('{"test": 2}')
        (Path(self.temp_dir) / "not_json.txt").write_text('not json')
        
        # List all states
        states = self.manager.list_saved_states()
        assert len(states) == 2
        assert "battle1_123.json" in states
        assert "battle2_456.json" in states
        assert "not_json.txt" not in states
        
        # Filter by battle_id
        states = self.manager.list_saved_states("battle1")
        assert len(states) == 1
        assert "battle1_123.json" in states
    
    def test_delete_state(self):
        """Test deleting saved states."""
        # Create test file
        test_file = Path(self.temp_dir) / "delete_test.json"
        test_file.write_text('{"test": true}')
        assert test_file.exists()
        
        # Delete existing file
        success = self.manager.delete_state("delete_test.json")
        assert success is True
        assert not test_file.exists()
        
        # Try to delete non-existent file
        success = self.manager.delete_state("nonexistent.json")
        assert success is False


class TestBattleCommunicatorStateOperations:
    """Test battle state operations in communicators."""
    
    @pytest.mark.asyncio
    async def test_communicator_save_state(self):
        """Test battle state saving via communicator."""
        communicator = Mock(spec=BattleCommunicator)
        
        # Mock save_battle_state method
        expected_response = {
            "type": "battle_state_saved",
            "battle_id": "test-battle",
            "state_id": "test-battle_123456",
            "success": True
        }
        communicator.save_battle_state.return_value = asyncio.Future()
        communicator.save_battle_state.return_value.set_result(expected_response)
        
        # Test save operation
        result = await communicator.save_battle_state("test-battle")
        assert result["success"] is True
        assert result["battle_id"] == "test-battle"
        
        communicator.save_battle_state.assert_called_once_with("test-battle")
    
    @pytest.mark.asyncio
    async def test_communicator_restore_state(self):
        """Test battle state restoration via communicator."""
        communicator = Mock(spec=BattleCommunicator)
        
        # Mock restore_battle_state method
        communicator.restore_battle_state.return_value = asyncio.Future()
        communicator.restore_battle_state.return_value.set_result(True)
        
        # Test restore operation
        state_data = {"battle_id": "test-battle", "turn": 5}
        success = await communicator.restore_battle_state("test-battle", state_data)
        assert success is True
        
        communicator.restore_battle_state.assert_called_once_with("test-battle", state_data)
    
    @pytest.mark.asyncio
    async def test_communicator_get_state(self):
        """Test getting battle state via communicator."""
        communicator = Mock(spec=BattleCommunicator)
        
        # Mock get_battle_state method
        expected_state = {
            "battle_id": "test-battle",
            "turn": 3,
            "players": ["p1", "p2"]
        }
        communicator.get_battle_state.return_value = asyncio.Future()
        communicator.get_battle_state.return_value.set_result(expected_state)
        
        # Test get operation
        state = await communicator.get_battle_state("test-battle")
        assert state["battle_id"] == "test-battle"
        assert state["turn"] == 3
        
        communicator.get_battle_state.assert_called_once_with("test-battle")


class TestPokemonEnvStateIntegration:
    """Test PokemonEnv integration with battle state management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Import here to avoid circular imports
        from src.env.pokemon_env import PokemonEnv
        from src.state.state_observer import StateObserver
        from src.action import action_helper
        
        # Create minimal test environment
        state_observer = Mock(spec=StateObserver)
        state_observer.get_observation_dimension.return_value = 100
        
        self.env = PokemonEnv(
            state_observer=state_observer,
            action_helper=action_helper,
            battle_mode="local"
        )
        
        # Override state manager with temp directory
        self.env._state_manager.storage_dir = Path(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_env_state_manager_initialization(self):
        """Test that PokemonEnv properly initializes state management."""
        assert hasattr(self.env, '_battle_serializer')
        assert hasattr(self.env, '_state_manager')
        assert self.env._state_manager.storage_dir.exists()
    
    def test_save_battle_state_no_battle(self):
        """Test saving state when no battle exists."""
        with pytest.raises(RuntimeError, match="Battle state save failed"):
            self.env.save_battle_state("player_0")
    
    def test_load_nonexistent_battle_state(self):
        """Test loading non-existent state file."""
        with pytest.raises(FileNotFoundError):
            self.env.load_battle_state("nonexistent.json")
    
    def test_list_empty_battle_states(self):
        """Test listing states when none exist."""
        states = self.env.list_saved_battle_states()
        assert len(states) == 0
    
    def test_delete_nonexistent_battle_state(self):
        """Test deleting non-existent state."""
        success = self.env.delete_battle_state("nonexistent.json")
        assert success is False
    
    def test_get_battle_state_info(self):
        """Test getting battle state information."""
        info = self.env.get_battle_state_info()
        assert "serializer_type" in info
        assert "storage_directory" in info
        assert "saved_states_count" in info
        assert "current_battles" in info
        assert "battle_mode" in info
        
        assert info["serializer_type"] == "PokeEnvBattleSerializer"
        assert info["battle_mode"] == "local"
        assert info["saved_states_count"] == 0
    
    @pytest.mark.asyncio
    async def test_save_state_via_communicator_no_player(self):
        """Test saving state via communicator when no player exists."""
        with pytest.raises(RuntimeError, match="Communicator state save failed"):
            await self.env.save_battle_state_via_communicator("nonexistent_player")
    
    @pytest.mark.asyncio
    async def test_restore_state_via_communicator_no_player(self):
        """Test restoring state via communicator when no player exists."""
        state_data = {"battle_id": "test"}
        with pytest.raises(RuntimeError, match="Communicator state restore failed"):
            await self.env.restore_battle_state_via_communicator("nonexistent_player", state_data)


class TestErrorHandling:
    """Test error handling in serialization system."""
    
    def test_serializer_error_handling(self):
        """Test error handling in serializer."""
        serializer = PokeEnvBattleSerializer()
        
        # Test with None battle
        with pytest.raises(Exception):
            serializer.serialize_state(None)
        
        # Test with invalid battle object
        invalid_battle = Mock()
        invalid_battle.battle_tag = None  # Invalid
        
        with pytest.raises(Exception):
            serializer.serialize_state(invalid_battle)
    
    def test_manager_error_handling(self):
        """Test error handling in state manager."""
        temp_dir = tempfile.mkdtemp()
        
        try:
            serializer = Mock(spec=BattleStateSerializer)
            manager = BattleStateManager(serializer, temp_dir)
            
            # Test save with serializer error
            serializer.serialize_state.side_effect = Exception("Serialization failed")
            
            with pytest.raises(Exception, match="Failed to save battle state"):
                manager.save_state(Mock())
            
            # Test load with invalid JSON
            invalid_file = Path(temp_dir) / "invalid.json"
            invalid_file.write_text("invalid json content")
            
            with pytest.raises(Exception, match="Failed to load battle state"):
                manager.load_state(str(invalid_file))
                
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.integration
class TestSerializationIntegration:
    """Integration tests for complete serialization workflow."""
    
    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.serializer = PokeEnvBattleSerializer()
        self.manager = BattleStateManager(self.serializer, self.temp_dir)
    
    def teardown_method(self):
        """Clean up integration test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_full_serialization_workflow(self):
        """Test complete save/load workflow."""
        # Create mock battle with realistic data
        mock_pokemon = Mock()
        mock_pokemon.species = "Pikachu"
        mock_pokemon.current_hp_fraction = 0.75
        mock_pokemon.max_hp = 100
        mock_pokemon.status = None
        mock_pokemon.moves = []
        mock_pokemon.types = [Mock(name="Electric")]
        mock_pokemon.types[0].name = "Electric"
        
        mock_player = Mock()
        mock_player.team = {"pikachu": mock_pokemon}
        mock_player.active_pokemon = mock_pokemon
        mock_player.username = "TestTrainer"
        
        mock_battle = Mock()
        mock_battle.battle_tag = "integration-test-battle"
        mock_battle.turn = 10
        mock_battle.weather = None
        mock_battle.terrain = None
        mock_battle.p1 = mock_player
        mock_battle.p2 = mock_player
        mock_battle.history = ["Turn 1", "Turn 2", "...Turn 10"]
        
        # Save the battle state
        filepath = self.manager.save_state(mock_battle)
        assert Path(filepath).exists()
        
        # Load and validate
        loaded_state = self.manager.load_state(filepath)
        assert loaded_state.battle_id == "integration-test-battle"
        assert loaded_state.turn == 10
        assert len(loaded_state.players) == 2
        
        # Verify JSON format
        with open(filepath, 'r') as f:
            json_data = json.load(f)
        
        assert json_data["battle_id"] == "integration-test-battle"
        assert json_data["turn"] == 10
        assert "players" in json_data
        assert "timestamp" in json_data
        
        # Test list and delete
        states = self.manager.list_saved_states()
        assert len(states) == 1
        
        success = self.manager.delete_state(Path(filepath).name)
        assert success is True
        assert not Path(filepath).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])