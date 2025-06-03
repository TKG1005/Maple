import sys
import pathlib
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


@pytest.mark.slow
def test_connect_local_showdown_turn1():
    numpy = pytest.importorskip("numpy")
    gymnasium = pytest.importorskip("gymnasium")
    poke_env = pytest.importorskip("poke_env")

    from env.pokemon_env import PokemonEnv
    from poke_env.player import Player
    from poke_env.server_configuration import LocalhostServerConfiguration

    class TurnObserver:
        def get_observation_dimension(self):
            return 1

        def observe(self, battle):
            return numpy.array([battle.turn], dtype=numpy.int32)

    class DummyActionHelper:
        def action_index_to_order(self, idx):
            return idx

    class RandomOpponent(Player):
        def choose_move(self, battle):
            return self.choose_random_move(battle)

    opponent = RandomOpponent(
        battle_format="gen9ou",
        server_configuration=LocalhostServerConfiguration,
    )

    env = PokemonEnv(opponent, TurnObserver(), DummyActionHelper())

    obs, info = env.reset()

    assert obs[0] >= 1
    assert "battle_tag" in info

