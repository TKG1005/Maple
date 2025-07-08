import unittest
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.damage.calculator import DamageCalculator
from src.damage.data_loader import DataLoader

class TestDamageCalculator(unittest.TestCase):

    def setUp(self):
        """Set up a DataLoader and DamageCalculator for all tests."""
        # Assuming the script is run from the project root, so config path is correct
        self.data_loader = DataLoader(config_path='config')
        self.calculator = DamageCalculator(self.data_loader)

    def test_basic_physical_damage(self):
        """Test a basic physical damage calculation."""
        attacker = {
            'name': 'Pikachu',
            'level': 50,
            'stats': {'attack': 110, 'sp_attack': 100},
            'types': ['でんき'] # Added types
        }
        defender = {
            'name': 'Chansey',
            'level': 50,
            'stats': {'defense': 60, 'sp_defense': 150, 'hp': 300},
            'types': ['ノーマル'] # Added types
        }
        
        result = self.calculator.calculate_damage_range(attacker, defender, 'たいあたり')

        # Expected calculation: 
        # base = floor(floor((50 * 2 / 5) + 2) * 40 * 110 / 60 / 50) + 2 = floor(22 * 40 * 110 / 60 / 50) + 2 = floor(32.26) + 2 = 34
        # max_damage = 34
        # min_damage = floor(34 * 0.85) = 28
        self.assertEqual(result['max_damage'], 34)
        self.assertEqual(result['min_damage'], 28)

    def test_basic_special_damage(self):
        """Test a basic special damage calculation."""
        attacker = {
            'name': 'Pikachu',
            'level': 50,
            'stats': {'attack': 100, 'sp_attack': 120},
            'types': ['でんき'] # Added types
        }
        defender = {
            'name': 'Gyarados',
            'level': 50,
            'stats': {'defense': 100, 'sp_defense': 130, 'hp': 200},
            'types': ['みず', 'ひこう'] # Added types
        }
        result = self.calculator.calculate_damage_range(attacker, defender, '10まんボルト')

        # Expected calculation:
        # base = floor(floor((50 * 2 / 5) + 2) * 90 * 120 / 130 / 50) + 2 = floor(22 * 90 * 120 / 130 / 50) + 2 = floor(36.55) + 2 = 38
        # max_damage = 38
        # min_damage = floor(38 * 0.85) = 32
        self.assertEqual(result['max_damage'], 38)
        self.assertEqual(result['min_damage'], 32)

    def test_type_effectiveness_damage(self):
        """Test damage calculation with type effectiveness."""
        attacker = {
            'name': 'Pikachu',
            'level': 50,
            'stats': {'attack': 100, 'sp_attack': 120},
            'types': ['でんき'] # Added types
        }
        defender = {
            'name': 'Charmander',
            'level': 50,
            'stats': {'defense': 50, 'sp_defense': 50, 'hp': 100},
            'types': ['ほのお']
        }
        result = self.calculator.calculate_damage_range(attacker, defender, 'ハイドロポンプ')

        # Expected calculation:
        # base = floor(floor((50 * 2 / 5) + 2) * 110 * 120 / 50 / 50) + 2 = floor(22 * 110 * 120 / 50 / 50) + 2 = floor(116.16) + 2 = 118
        # type_multiplier = 2.0 (Water vs Fire)
        # final_base_damage = floor(118 * 2.0) = 236
        # max_damage = 236
        # min_damage = floor(236 * 0.85) = 200
        self.assertEqual(result['max_damage'], 236)
        self.assertEqual(result['min_damage'], 200)
        self.assertEqual(result['type_effectiveness'], 2.0)

if __name__ == '__main__':
    unittest.main()
