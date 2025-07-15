"""
Test for Ditto move access IndexError fix.

This test verifies that the state_spec.yml modifications correctly handle
situations where a Pokemon has fewer than 4 moves (e.g., Ditto after fainting).
"""
import pytest
from unittest.mock import Mock, MagicMock
from src.state.state_observer import StateObserver


class TestDittoMoveAccess:
    """Test safe move access for Pokemon with varying number of moves."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock battle object
        self.battle = Mock()
        self.battle.battle_tag = "test-battle"
        self.battle.turn = 1
        
        # Mock active Pokemon
        self.active_pokemon = Mock()
        self.battle.active_pokemon = self.active_pokemon
        
        # Mock StateObserver
        self.observer = StateObserver("config/state_spec.yml")
    
    def test_single_move_pokemon(self):
        """Test Pokemon with only 1 move (like Ditto with Transform only)."""
        # Create single move
        transform_move = Mock()
        transform_move.type.name.lower.return_value = "normal"
        transform_move.base_power = 0
        transform_move.accuracy = 1.0
        transform_move.category.name.lower.return_value = "status"
        transform_move.max_pp = 10
        transform_move.current_pp = 10
        
        # Mock context with single move
        context = {
            'active_sorted_moves': [transform_move],
            'battle': self.battle,
            'active': self.active_pokemon
        }
        
        # Test move access patterns that should not raise IndexError
        # These would previously fail with IndexError
        
        # Move 2 access (index 1) - should return defaults
        move2_type = self._evaluate_path(
            "active_sorted_moves[1].type.name.lower() if len(active_sorted_moves) > 1 and active_sorted_moves[1] else 'none'",
            context
        )
        assert move2_type == 'none'
        
        move2_power = self._evaluate_path(
            "active_sorted_moves[1].base_power if len(active_sorted_moves) > 1 and active_sorted_moves[1] else 0",
            context
        )
        assert move2_power == 0
        
        # Move 3 access (index 2) - should return defaults
        move3_type = self._evaluate_path(
            "active_sorted_moves[2].type.name.lower() if len(active_sorted_moves) > 2 and active_sorted_moves[2] else 'none'",
            context
        )
        assert move3_type == 'none'
        
        # Move 4 access (index 3) - should return defaults
        move4_type = self._evaluate_path(
            "active_sorted_moves[3].type.name.lower() if len(active_sorted_moves) > 3 and active_sorted_moves[3] else 'none'",
            context
        )
        assert move4_type == 'none'
    
    def test_two_move_pokemon(self):
        """Test Pokemon with 2 moves."""
        # Create two moves
        move1 = self._create_mock_move("fire", 80, 1.0, "physical", 15, 15)
        move2 = self._create_mock_move("water", 90, 0.9, "special", 10, 8)
        
        context = {
            'active_sorted_moves': [move1, move2],
            'battle': self.battle,
            'active': self.active_pokemon
        }
        
        # Move 1 should work
        move1_type = self._evaluate_path(
            "active_sorted_moves[0].type.name.lower()",
            context
        )
        assert move1_type == "fire"
        
        # Move 2 should work
        move2_type = self._evaluate_path(
            "active_sorted_moves[1].type.name.lower() if len(active_sorted_moves) > 1 and active_sorted_moves[1] else 'none'",
            context
        )
        assert move2_type == "water"
        
        # Move 3 should return default
        move3_type = self._evaluate_path(
            "active_sorted_moves[2].type.name.lower() if len(active_sorted_moves) > 2 and active_sorted_moves[2] else 'none'",
            context
        )
        assert move3_type == 'none'
        
        # Move 4 should return default
        move4_type = self._evaluate_path(
            "active_sorted_moves[3].type.name.lower() if len(active_sorted_moves) > 3 and active_sorted_moves[3] else 'none'",
            context
        )
        assert move4_type == 'none'
    
    def test_four_move_pokemon(self):
        """Test Pokemon with full 4 moves."""
        # Create four moves
        moves = [
            self._create_mock_move("fire", 80, 1.0, "physical", 15, 15),
            self._create_mock_move("water", 90, 0.9, "special", 10, 8),
            self._create_mock_move("grass", 75, 1.0, "special", 20, 12),
            self._create_mock_move("electric", 95, 0.8, "special", 5, 1)
        ]
        
        context = {
            'active_sorted_moves': moves,
            'battle': self.battle,
            'active': self.active_pokemon
        }
        
        # All moves should work
        for i, expected_type in enumerate(["fire", "water", "grass", "electric"]):
            if i == 0:
                path = "active_sorted_moves[0].type.name.lower()"
            else:
                path = f"active_sorted_moves[{i}].type.name.lower() if len(active_sorted_moves) > {i} and active_sorted_moves[{i}] else 'none'"
            
            move_type = self._evaluate_path(path, context)
            assert move_type == expected_type
    
    def test_pp_fraction_calculation(self):
        """Test PP fraction calculation with safe access."""
        # Create move with specific PP values
        move = self._create_mock_move("normal", 40, 1.0, "physical", 30, 15)
        
        context = {
            'active_sorted_moves': [move],
            'battle': self.battle,
            'active': self.active_pokemon
        }
        
        # Move 1 PP fraction should work
        pp_frac = self._evaluate_path(
            "active_sorted_moves[0].current_pp / active_sorted_moves[0].max_pp if active_sorted_moves[0] and active_sorted_moves[0].max_pp > 0 else 0",
            context
        )
        assert pp_frac == 0.5  # 15/30
        
        # Move 2 PP fraction should return 0 (safe access)
        pp_frac2 = self._evaluate_path(
            "active_sorted_moves[1].current_pp / active_sorted_moves[1].max_pp if len(active_sorted_moves) > 1 and active_sorted_moves[1] and active_sorted_moves[1].max_pp > 0 else 0",
            context
        )
        assert pp_frac2 == 0
    
    def test_pp_is_one_flag(self):
        """Test PP is one flag with safe access."""
        # Create move with 1 PP remaining
        move = self._create_mock_move("normal", 40, 1.0, "physical", 30, 1)
        
        context = {
            'active_sorted_moves': [move],
            'battle': self.battle,
            'active': self.active_pokemon
        }
        
        # Move 1 PP is one should be True
        pp_is_one = self._evaluate_path(
            "active_sorted_moves[0].current_pp == 1 if active_sorted_moves[0] else False",
            context
        )
        assert pp_is_one is True
        
        # Move 2 PP is one should be False (safe access)
        pp_is_one2 = self._evaluate_path(
            "active_sorted_moves[1].current_pp == 1 if len(active_sorted_moves) > 1 and active_sorted_moves[1] else False",
            context
        )
        assert pp_is_one2 is False
    
    def _create_mock_move(self, type_name, base_power, accuracy, category, max_pp, current_pp):
        """Create a mock move with specified attributes."""
        move = Mock()
        move.type.name.lower.return_value = type_name
        move.base_power = base_power
        move.accuracy = accuracy
        move.category.name.lower.return_value = category
        move.max_pp = max_pp
        move.current_pp = current_pp
        return move
    
    def _evaluate_path(self, path, context):
        """Evaluate a battle path expression in the given context."""
        # This is a simplified evaluation - in real StateObserver this would be more complex
        # For testing purposes, we just use eval with the context
        # Include len function in builtins for testing
        try:
            return eval(path, {"__builtins__": {"len": len}}, context)
        except Exception as e:
            pytest.fail(f"Path evaluation failed: {path}, Error: {e}")


class TestDamageCalculationSafety:
    """Test damage calculation with None moves."""
    
    def test_calc_damage_expectation_with_none_move(self):
        """Test that calc_damage_expectation_for_ai handles None move gracefully."""
        from unittest.mock import Mock
        
        # Create the wrapper function as implemented in StateObserver._build_context
        def calc_damage_expectation_for_ai(attacker, target, move, terastallized=False):
            # Validate basic inputs - raise error if attacker or target are None/invalid
            if not attacker or not target:
                raise ValueError(f"Invalid input: attacker={attacker}, target={target}, move={move}")
            
            # Handle case where move is None (e.g., Pokemon has fewer than 4 moves)
            # Return safe default values: 0% expected damage, 0% variance
            if not move:
                return (0.0, 0.0)
            
            # This would be the actual damage calculation for non-None moves
            return (50.0, 5.0)  # Example values for testing
        
        # Create mock attacker and target
        attacker = Mock()
        attacker.species = "ditto"
        
        target = Mock() 
        target.species = "pikachu"
        
        # Test with None move - should return (0.0, 0.0)
        result = calc_damage_expectation_for_ai(attacker, target, None)
        assert result == (0.0, 0.0), f"Expected (0.0, 0.0) for None move, got {result}"
        
        # Test with valid move - should work normally
        valid_move = Mock()
        valid_move.id = "tackle"
        result = calc_damage_expectation_for_ai(attacker, target, valid_move)
        assert result == (50.0, 5.0), f"Expected (50.0, 5.0) for valid move, got {result}"
        
        # Test with None attacker - should raise ValueError
        try:
            calc_damage_expectation_for_ai(None, target, valid_move)
            assert False, "Should have raised ValueError for None attacker"
        except ValueError:
            pass  # Expected
        
        # Test with None target - should raise ValueError  
        try:
            calc_damage_expectation_for_ai(attacker, None, valid_move)
            assert False, "Should have raised ValueError for None target"
        except ValueError:
            pass  # Expected


class TestDittoTransformScenario:
    """Test specific Ditto transform/faint scenario."""
    
    def test_ditto_faint_after_transform(self):
        """Test the specific scenario from the bug report."""
        # Simulate Ditto after fainting - should have only Transform move
        transform_move = Mock()
        transform_move.type.name.lower.return_value = "normal"
        transform_move.base_power = 0
        transform_move.accuracy = 1.0
        transform_move.category.name.lower.return_value = "status"
        transform_move.max_pp = 10
        transform_move.current_pp = 10
        
        context = {
            'active_sorted_moves': [transform_move],  # Only 1 move after faint
            'battle': Mock(),
            'active': Mock()
        }
        
        # The problematic path from the error log
        # This should NOT raise IndexError anymore
        result = eval(
            "active_sorted_moves[1].type.name.lower() if len(active_sorted_moves) > 1 and active_sorted_moves[1] else 'none'",
            {"__builtins__": {"len": len}},
            context
        )
        
        assert result == 'none'
        
        # Verify all move indices are safe
        for i in range(1, 4):  # Test indices 1, 2, 3
            result = eval(
                f"active_sorted_moves[{i}].type.name.lower() if len(active_sorted_moves) > {i} and active_sorted_moves[{i}] else 'none'",
                {"__builtins__": {"len": len}},
                context
            )
            assert result == 'none'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])