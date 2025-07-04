import unittest
import numpy as np
from unittest.mock import Mock
from poke_env.environment.battle import AbstractBattle
from poke_env.environment.pokemon import Pokemon
from poke_env.environment.move import Move
from poke_env.environment.pokemon_type import PokemonType
from src.state.type_matchup_extractor import TypeMatchupFeatureExtractor

class TestTypeMatchupFeatureExtractor(unittest.TestCase):

    def setUp(self):
        self.extractor = TypeMatchupFeatureExtractor(gen=9)

    def _create_dummy_pokemon(self, type1, type2=None, tera_type=None, moves=None):
        pokemon = Mock(spec=Pokemon)
        pokemon.type_1 = type1
        pokemon.type_2 = type2 if type2 else None
        pokemon.is_terastallized = False
        if tera_type:
            pokemon.is_terastallized = True
        pokemon.tera_type = tera_type
        pokemon.moves = {m.id: m for m in moves} if moves else {}
        pokemon.active = True
        pokemon.fainted = False
        return pokemon

    def _create_dummy_move(self, move_id, move_type):
        move = Mock(spec=Move)
        move.id = move_id
        move.type = move_type
        return move

    def test_extractor_output_shape(self):
        # Arrange
        battle = Mock(spec=AbstractBattle)
        
        my_active_moves = [
            self._create_dummy_move("flamethrower", PokemonType.FIRE),
            self._create_dummy_move("thunderbolt", PokemonType.ELECTRIC),
            self._create_dummy_move("surf", PokemonType.WATER),
            self._create_dummy_move("earthquake", PokemonType.GROUND),
        ]
        my_active = self._create_dummy_pokemon(PokemonType.FIRE, moves=my_active_moves)
        
        opp_active = self._create_dummy_pokemon(PokemonType.GRASS)
        
        battle.active_pokemon = my_active
        battle.opponent_active_pokemon = opp_active
        battle.team = {my_active.species: my_active}
        battle.opponent_team = {opp_active.species: opp_active}

        # Act
        features = self.extractor.extract(battle)

        # Assert
        self.assertEqual(features.shape, (18,))

    def test_fire_vs_grass(self):
        # Arrange
        battle = Mock(spec=AbstractBattle)
        
        my_active_moves = [self._create_dummy_move("flamethrower", PokemonType.FIRE)]
        my_active = self._create_dummy_pokemon(PokemonType.FIRE, moves=my_active_moves)
        
        opp_active = self._create_dummy_pokemon(PokemonType.GRASS)
        
        battle.active_pokemon = my_active
        battle.opponent_active_pokemon = opp_active
        battle.team = {my_active.species: my_active}
        battle.opponent_team = {opp_active.species: opp_active}

        # Act
        features = self.extractor.extract(battle)

        # Assert
        # move1_vs_opp_active (index 0) should be 2.0 (Fire vs Grass)
        self.assertAlmostEqual(features[0], 2.0, places=2)

    def test_no_opponent_pokemon(self):
        # Arrange
        battle = Mock(spec=AbstractBattle)
        
        my_active_moves = [self._create_dummy_move("flamethrower", PokemonType.FIRE)]
        my_active = self._create_dummy_pokemon(PokemonType.FIRE, moves=my_active_moves)
        
        battle.active_pokemon = my_active
        battle.opponent_active_pokemon = None
        battle.team = {my_active.species: my_active}
        battle.opponent_team = {}

        # Act
        features = self.extractor.extract(battle)

        # Assert
        # All matchups should be 0.0 (log2(1.0) = 0.0)
        self.assertTrue(np.all(features == 0.0))

if __name__ == '__main__':
    unittest.main()
