from types import SimpleNamespace, ModuleType
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Create a minimal numpy stub for testing without the real package
np_stub = ModuleType("numpy")
np_stub.float32 = "float32"
np_stub.int8 = "int8"
np_stub.ndarray = list
np_stub.random = SimpleNamespace(default_rng=lambda seed=None: None)
np_stub.zeros = lambda shape, dtype=None: [0] * shape
np_stub.array = lambda data, dtype=None: list(data)
sys.modules.setdefault("numpy", np_stub)
sys.modules.setdefault("numpy.random", np_stub.random)
gym_stub = ModuleType("gymnasium")
class DummyBox:
    def __init__(self, *args, **kwargs):
        self.low = kwargs.get("low")
        self.high = kwargs.get("high")
        self.shape = kwargs.get("shape")

class DummyDiscrete:
    def __init__(self, n):
        self.n = n

gym_stub.spaces = SimpleNamespace(Box=DummyBox, Dict=dict, Discrete=DummyDiscrete)
gym_stub.Env = object
sys.modules.setdefault("gymnasium", gym_stub)
sys.modules.setdefault("gymnasium.spaces", gym_stub.spaces)
yaml_stub = ModuleType("yaml")
yaml_stub.safe_load = lambda x: {}
sys.modules.setdefault("yaml", yaml_stub)
poke_env_stub = ModuleType("poke_env")
concurrency = ModuleType("poke_env.concurrency")
concurrency.POKE_LOOP = None
poke_env_stub.concurrency = concurrency
environment = ModuleType("poke_env.environment")
abstract_battle = ModuleType("poke_env.environment.abstract_battle")
class _AbstractBattle: ...
abstract_battle.AbstractBattle = _AbstractBattle
environment.abstract_battle = abstract_battle
player_mod = ModuleType("poke_env.player")
class _Player: ...
player_mod.Player = _Player
poke_env_stub.player = player_mod
poke_env_stub.environment = environment
sys.modules.setdefault("poke_env", poke_env_stub)
sys.modules.setdefault("poke_env.concurrency", concurrency)
sys.modules.setdefault("poke_env.environment", environment)
sys.modules.setdefault("poke_env.environment.abstract_battle", abstract_battle)
sys.modules.setdefault("poke_env.player", player_mod)
import numpy as np

from src.env.pokemon_env import PokemonEnv


class DummyObserver:
    def get_observation_dimension(self):
        return 1

    def observe(self, battle):
        return np.zeros(1, dtype=np.float32)


class DummyActionHelper:
    def get_available_actions(self, battle):
        mask = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
        mapping = {8: ("switch", 0), 9: ("switch", 1)}
        return mask, mapping

    def get_available_actions_with_details(self, battle):
        mask = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8)
        mapping = {
            8: {"type": "switch", "name": "a", "id": "pika"},
            9: {"type": "switch", "name": "b", "id": "bulba"},
        }
        return mask, mapping


class DummyBattle:
    def __init__(self):
        self.available_switches = [SimpleNamespace(species="pika"), SimpleNamespace(species="bulba")]


class DummyBattleForce(DummyBattle):
    def __init__(self):
        super().__init__()
        self.force_switch = True



def make_env():
    return PokemonEnv(state_observer=DummyObserver(), action_helper=DummyActionHelper(), opponent_player=None)


def test_get_action_mask_filters_switches():
    env = make_env()
    env._current_battles = {"player_0": DummyBattle()}
    env._selected_species["player_0"] = {"pika"}
    mask, mapping = env.get_action_mask("player_0")
    assert mask[8] == 1
    assert mask[9] == 0
    assert mapping == {8: ("switch", 0), 9: ("switch", 1)}


def test_get_action_mask_with_details():
    env = make_env()
    env._current_battles = {"player_0": DummyBattle()}
    env._selected_species["player_0"] = {"pika"}
    mask, details = env.get_action_mask("player_0", with_details=True)
    assert mask[8] == 1
    assert mask[9] == 0
    assert details[8]["type"] == "switch"
    assert details[9]["id"] == "bulba"


def test_get_action_mask_force_switch():
    env = make_env()
    env._current_battles = {"player_0": DummyBattleForce()}
    env._selected_species["player_0"] = {"pika", "bulba"}
    mask, mapping = env.get_action_mask("player_0")
    assert mask[8] == 1
    assert mask[9] == 1
    assert mapping == {8: ("switch", 0), 9: ("switch", 1)}

