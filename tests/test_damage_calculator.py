
import unittest
import pandas as pd
from src.damage.data_loader import DataLoader
from src.damage.calculator import DamageCalculator

class TestDamageCalculator(unittest.TestCase):

    def setUp(self):
        # Create a dummy DataLoader
        class MockDataLoader:
            def __init__(self):
                self.pokemon_stats = pd.DataFrame({
                    'name': ['Pikachu', 'Charmander'],
                    'type1': ['Electric', 'Fire'],
                    'type2': [None, None]
                })
                self.moves = pd.DataFrame({
                    'name': ['Thunderbolt', 'Flamethrower'],
                    'type': ['Electric', 'Fire'],
                    'power': [90, 90],
                    'category': ['Special', 'Special']
                })
                self.type_chart = pd.DataFrame({
                    'attacking_type': ['Electric', 'Fire'],
                    'defending_type': ['Fire', 'Electric'],
                    'multiplier': [1.0, 1.0]
                })
        self.data_loader = MockDataLoader()
        self.calculator = DamageCalculator(self.data_loader)

    def test_calculate_damage_range_no_stab_no_effectiveness(self):
        attacker = {'name': 'Pikachu', 'attack': 55, 'level': 50}
        defender = {'name': 'Charmander', 'defense': 43, 'max_hp': 100}
        move = {'name': 'Thunderbolt'}
        field_state = {}

        result = self.calculator.calculate_damage_range(attacker, defender, move, field_state)
        # This is a placeholder, a more accurate expected value should be calculated
        self.assertIsNotNone(result['damage_range'])
        self.assertIsNotNone(result['hp_percentage'])
        self.assertIsNotNone(result['knockout_count'])

    def test_calculate_damage_range_hp_knockout(self):
        attacker = {'name': 'Pikachu', 'attack': 250, 'level': 50}
        defender = {'name': 'Charmander', 'defense': 50, 'max_hp': 200}
        move = {'name': 'Thunderbolt'}
        field_state = {}

        result = self.calculator.calculate_damage_range(attacker, defender, move, field_state)
        # Assuming a damage range that results in 1HKO
        self.assertIn(result['knockout_count'], ["確定1発", "乱数1発"])
        self.assertGreater(result['hp_percentage'][0], 0)

        attacker = {'name': 'Pikachu', 'attack': 20, 'level': 50}
        defender = {'name': 'Charmander', 'defense': 50, 'max_hp': 200}
        move = {'name': 'Thunderbolt'}
        field_state = {}

        result = self.calculator.calculate_damage_range(attacker, defender, move, field_state)
        # Assuming a damage range that results in 2HKO or more
        self.assertIn(result['knockout_count'], ["乱数2発", "乱数3発", "確定複数発"])

    def test_simulate_move_effect_hit(self):
        attacker = {'name': 'Pikachu', 'attack': 55, 'level': 50}
        defender = {'name': 'Charmander', 'defense': 43}
        move = {'name': 'Thunderbolt', 'accuracy': 100, 'effect_prob': 0}
        field_state = {}

        result = self.calculator.simulate_move_effect(attacker, defender, move, field_state)
        self.assertTrue(result['hit'])
        self.assertGreater(result['damage'], 0)
        self.assertFalse(result['critical_hit'])
        self.assertFalse(result['additional_effect']['triggered'])

    def test_simulate_move_effect_miss(self):
        attacker = {'name': 'Pikachu', 'attack': 55, 'level': 50}
        defender = {'name': 'Charmander', 'defense': 43}
        move = {'name': 'Thunderbolt', 'accuracy': 0, 'effect_prob': 0}
        field_state = {}

        result = self.calculator.simulate_move_effect(attacker, defender, move, field_state)
        self.assertFalse(result['hit'])
        self.assertEqual(result['damage'], 0)

    def test_simulate_move_effect_additional_effect(self):
        attacker = {'name': 'Pikachu', 'attack': 55, 'level': 50}
        defender = {'name': 'Charmander', 'defense': 43}
        move = {'name': 'Thunderbolt', 'accuracy': 100, 'effect_prob': 100, 'effect_type': 'Paralysis'}
        field_state = {}

        result = self.calculator.simulate_move_effect(attacker, defender, move, field_state)
        self.assertTrue(result['hit'])
        self.assertTrue(result['additional_effect']['triggered'])
        self.assertEqual(result['additional_effect']['content'], 'Paralysis')

if __name__ == '__main__':
    unittest.main()
