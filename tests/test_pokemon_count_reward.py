import unittest
from unittest.mock import MagicMock

from src.rewards.pokemon_count import PokemonCountReward


class TestPokemonCountReward(unittest.TestCase):
    
    def setUp(self):
        self.reward = PokemonCountReward()
    
    def test_reset_does_nothing(self):
        """Reset should do nothing as this reward is stateless"""
        self.reward.reset()
        self.reward.reset(None)
        # Should not raise any exceptions
    
    def test_battle_not_finished_returns_zero(self):
        """When battle is not finished, should return 0"""
        battle = MagicMock()
        battle.finished = False
        
        result = self.reward.calc(battle)
        self.assertEqual(result, 0.0)
    
    def test_equal_pokemon_count_returns_zero(self):
        """When both players have equal pokemon (victory), should return 0"""
        battle = MagicMock()
        battle.finished = True
        battle.won = True  # Victory but no bonus for equal count
        
        # Create mock pokemon for both teams (2 each, all alive)
        my_pokemon = [MagicMock(), MagicMock()]
        opp_pokemon = [MagicMock(), MagicMock()]
        
        for mon in my_pokemon:
            mon.fainted = False
        for mon in opp_pokemon:
            mon.fainted = False
        
        battle.team = {f"mon{i}": mon for i, mon in enumerate(my_pokemon)}
        battle.opponent_team = {f"mon{i}": mon for i, mon in enumerate(opp_pokemon)}
        
        result = self.reward.calc(battle)
        self.assertEqual(result, 0.0)
    
    def test_one_pokemon_difference_returns_zero(self):
        """When difference is 1 pokemon (victory), should return 0"""
        battle = MagicMock()
        battle.finished = True
        battle.won = True  # Victory but no bonus for 1 pokemon difference
        
        # My team: 2 alive, opponent team: 1 alive
        my_pokemon = [MagicMock(), MagicMock()]
        opp_pokemon = [MagicMock()]
        
        for mon in my_pokemon:
            mon.fainted = False
        for mon in opp_pokemon:
            mon.fainted = False
        
        battle.team = {f"mon{i}": mon for i, mon in enumerate(my_pokemon)}
        battle.opponent_team = {f"mon{i}": mon for i, mon in enumerate(opp_pokemon)}
        
        result = self.reward.calc(battle)
        self.assertEqual(result, 0.0)
    
    def test_two_pokemon_difference_positive(self):
        """When I have 2 more pokemon (victory), should return +2"""
        battle = MagicMock()
        battle.finished = True
        battle.won = True  # Victory
        
        # My team: 3 alive, opponent team: 1 alive
        my_pokemon = [MagicMock(), MagicMock(), MagicMock()]
        opp_pokemon = [MagicMock()]
        
        for mon in my_pokemon:
            mon.fainted = False
        for mon in opp_pokemon:
            mon.fainted = False
        
        battle.team = {f"mon{i}": mon for i, mon in enumerate(my_pokemon)}
        battle.opponent_team = {f"mon{i}": mon for i, mon in enumerate(opp_pokemon)}
        
        result = self.reward.calc(battle)
        self.assertEqual(result, 2.0)
    
    def test_two_pokemon_difference_negative(self):
        """When opponent has 2 more pokemon (defeat), should return 0 (no penalty)"""
        battle = MagicMock()
        battle.finished = True
        battle.won = False  # Defeat - no penalty should be applied
        
        # My team: 1 alive, opponent team: 3 alive
        my_pokemon = [MagicMock()]
        opp_pokemon = [MagicMock(), MagicMock(), MagicMock()]
        
        for mon in my_pokemon:
            mon.fainted = False
        for mon in opp_pokemon:
            mon.fainted = False
        
        battle.team = {f"mon{i}": mon for i, mon in enumerate(my_pokemon)}
        battle.opponent_team = {f"mon{i}": mon for i, mon in enumerate(opp_pokemon)}
        
        result = self.reward.calc(battle)
        self.assertEqual(result, 0.0)
    
    def test_three_pokemon_difference_positive(self):
        """When I have 3 more pokemon (victory), should return +5"""
        battle = MagicMock()
        battle.finished = True
        battle.won = True  # Victory
        
        # My team: 3 alive, opponent team: 0 alive
        my_pokemon = [MagicMock(), MagicMock(), MagicMock()]
        opp_pokemon = []
        
        for mon in my_pokemon:
            mon.fainted = False
        
        battle.team = {f"mon{i}": mon for i, mon in enumerate(my_pokemon)}
        battle.opponent_team = {f"mon{i}": mon for i, mon in enumerate(opp_pokemon)}
        
        result = self.reward.calc(battle)
        self.assertEqual(result, 5.0)
    
    def test_three_pokemon_difference_negative(self):
        """When opponent has 3 more pokemon (defeat), should return 0 (no penalty)"""
        battle = MagicMock()
        battle.finished = True
        battle.won = False  # Defeat - no penalty should be applied
        
        # My team: 0 alive, opponent team: 3 alive
        my_pokemon = []
        opp_pokemon = [MagicMock(), MagicMock(), MagicMock()]
        
        for mon in opp_pokemon:
            mon.fainted = False
        
        battle.team = {f"mon{i}": mon for i, mon in enumerate(my_pokemon)}
        battle.opponent_team = {f"mon{i}": mon for i, mon in enumerate(opp_pokemon)}
        
        result = self.reward.calc(battle)
        self.assertEqual(result, 0.0)
    
    def test_handles_fainted_pokemon_correctly(self):
        """Should correctly count only non-fainted pokemon"""
        battle = MagicMock()
        battle.finished = True
        
        # My team: 2 total (1 alive, 1 fainted)
        # Opponent team: 3 total (2 alive, 1 fainted)
        my_pokemon = [MagicMock(), MagicMock()]
        opp_pokemon = [MagicMock(), MagicMock(), MagicMock()]
        
        my_pokemon[0].fainted = False  # alive
        my_pokemon[1].fainted = True   # fainted
        opp_pokemon[0].fainted = False  # alive
        opp_pokemon[1].fainted = False  # alive
        opp_pokemon[2].fainted = True   # fainted
        
        battle.team = {f"mon{i}": mon for i, mon in enumerate(my_pokemon)}
        battle.opponent_team = {f"mon{i}": mon for i, mon in enumerate(opp_pokemon)}
        
        # 1 vs 2 pokemon = -1 difference = 0 reward
        result = self.reward.calc(battle)
        self.assertEqual(result, 0.0)
    
    def test_handles_missing_fainted_attribute(self):
        """Should handle pokemon without fainted attribute gracefully"""
        battle = MagicMock()
        battle.finished = True
        
        # Create pokemon without fainted attribute
        my_pokemon = [MagicMock(), MagicMock()]
        opp_pokemon = [MagicMock()]
        
        # Remove fainted attribute to simulate missing attribute
        for mon in my_pokemon + opp_pokemon:
            if hasattr(mon, 'fainted'):
                del mon.fainted
        
        battle.team = {f"mon{i}": mon for i, mon in enumerate(my_pokemon)}
        battle.opponent_team = {f"mon{i}": mon for i, mon in enumerate(opp_pokemon)}
        
        # Should assume all pokemon are alive when fainted attribute is missing
        # 2 vs 1 = +1 difference = 0 reward
        result = self.reward.calc(battle)
        self.assertEqual(result, 0.0)
    
    def test_defeat_returns_zero_regardless_of_difference(self):
        """When battle is lost, should always return 0 regardless of pokemon difference"""
        battle = MagicMock()
        battle.finished = True
        battle.won = False  # Defeat
        
        # Test various defeat scenarios
        scenarios = [
            # (my_count, opp_count, expected_reward)
            (0, 1, 0.0),  # Close defeat
            (0, 2, 0.0),  # 2 pokemon difference defeat  
            (0, 3, 0.0),  # 3 pokemon difference defeat
            (1, 3, 0.0),  # 2 pokemon difference defeat with some remaining
        ]
        
        for my_count, opp_count, expected in scenarios:
            with self.subTest(my_count=my_count, opp_count=opp_count):
                my_pokemon = [MagicMock() for _ in range(my_count)]
                opp_pokemon = [MagicMock() for _ in range(opp_count)]
                
                for mon in my_pokemon:
                    mon.fainted = False
                for mon in opp_pokemon:
                    mon.fainted = False
                
                battle.team = {f"mon{i}": mon for i, mon in enumerate(my_pokemon)}
                battle.opponent_team = {f"mon{i}": mon for i, mon in enumerate(opp_pokemon)}
                
                result = self.reward.calc(battle)
                self.assertEqual(result, expected)
    
    def test_victory_rewards_by_difference(self):
        """Test victory rewards for various pokemon count differences"""
        battle = MagicMock()
        battle.finished = True
        battle.won = True  # Victory
        
        # Test various victory scenarios
        scenarios = [
            # (my_count, opp_count, expected_reward)
            (1, 1, 0.0),  # Equal count victory
            (2, 1, 0.0),  # 1 pokemon difference victory
            (3, 1, 2.0),  # 2 pokemon difference victory
            (3, 0, 5.0),  # 3 pokemon difference victory
            (6, 2, 5.0),  # 4 pokemon difference victory (capped at 5.0)
        ]
        
        for my_count, opp_count, expected in scenarios:
            with self.subTest(my_count=my_count, opp_count=opp_count):
                my_pokemon = [MagicMock() for _ in range(my_count)]
                opp_pokemon = [MagicMock() for _ in range(opp_count)]
                
                for mon in my_pokemon:
                    mon.fainted = False
                for mon in opp_pokemon:
                    mon.fainted = False
                
                battle.team = {f"mon{i}": mon for i, mon in enumerate(my_pokemon)}
                battle.opponent_team = {f"mon{i}": mon for i, mon in enumerate(opp_pokemon)}
                
                result = self.reward.calc(battle)
                self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()