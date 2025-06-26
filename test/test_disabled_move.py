from types import SimpleNamespace, ModuleType
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# minimal numpy stub
np_stub = ModuleType("numpy")
np_stub.float32 = "float32"
np_stub.int8 = "int8"
np_stub.ndarray = list
np_stub.random = SimpleNamespace(default_rng=lambda seed=None: None)
np_stub.zeros = lambda shape, dtype=None: [0] * shape
np_stub.array = lambda data, dtype=None: list(data)
sys.modules.setdefault("numpy", np_stub)
sys.modules.setdefault("numpy.random", np_stub.random)

# gym stub
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

# yaml stub
yaml_stub = ModuleType("yaml")
yaml_stub.safe_load = lambda f: {}
sys.modules.setdefault("yaml", yaml_stub)

# poke_env stubs
poke_env_stub = ModuleType("poke_env")
concurrency = ModuleType("poke_env.concurrency")
concurrency.POKE_LOOP = None
poke_env_stub.concurrency = concurrency

environment = ModuleType("poke_env.environment")
abstract_battle = ModuleType("poke_env.environment.abstract_battle")
class _AbstractBattle: ...
abstract_battle.AbstractBattle = _AbstractBattle
environment.abstract_battle = abstract_battle
move_mod = ModuleType("poke_env.environment.move")
class _Move: ...
move_mod.Move = _Move
battle_mod = ModuleType("poke_env.environment.battle")
class _Battle: ...
battle_mod.Battle = _Battle
pokemon_mod = ModuleType("poke_env.environment.pokemon")
class _Pokemon: ...
pokemon_mod.Pokemon = _Pokemon
environment.move = move_mod
environment.battle = battle_mod
environment.pokemon = pokemon_mod

player_mod = ModuleType("poke_env.player")
class _Player: ...
player_mod.Player = _Player
player_player_mod = ModuleType("poke_env.player.player")
player_player_mod.Player = _Player
bo_mod = ModuleType("poke_env.player.battle_order")
bo_mod.BattleOrder = object
poke_env_stub.player = player_mod
poke_env_stub.environment = environment

sys.modules.setdefault("poke_env", poke_env_stub)
sys.modules.setdefault("poke_env.concurrency", concurrency)
sys.modules.setdefault("poke_env.environment", environment)
sys.modules.setdefault("poke_env.environment.abstract_battle", abstract_battle)
sys.modules.setdefault("poke_env.environment.move", move_mod)
sys.modules.setdefault("poke_env.environment.battle", battle_mod)
sys.modules.setdefault("poke_env.environment.pokemon", pokemon_mod)
sys.modules.setdefault("poke_env.player", player_mod)
sys.modules.setdefault("poke_env.player.player", player_player_mod)
sys.modules.setdefault("poke_env.player.battle_order", bo_mod)

import numpy as np
from src.env.pokemon_env import PokemonEnv
from src.action import action_helper


class DummyObserver:
    def get_observation_dimension(self):
        return 1

    def observe(self, battle):
        return np.zeros(1, dtype=np.float32)


def make_env():
    return PokemonEnv(state_observer=DummyObserver(), action_helper=action_helper, opponent_player=None)


class DummyBattle:
    def __init__(self):
        m0 = SimpleNamespace(id="a", current_pp=0)
        m1 = SimpleNamespace(id="b", current_pp=5)
        m2 = SimpleNamespace(id="c", current_pp=5)
        m3 = SimpleNamespace(id="d", current_pp=5)
        self.active_pokemon = SimpleNamespace(moves={0: m0, 1: m1, 2: m2, 3: m3})
        self.available_moves = [m1, m2, m3]
        self.available_switches = []
        self.force_switch = False
        self.can_tera = True


def test_pp0_move_disabled_and_mask():
    env = make_env()
    battle = DummyBattle()
    mapping = action_helper.get_action_mapping(battle)
    assert len(mapping) >= 4
    assert mapping[0][2] is True
    mask = env._build_action_mask(mapping)
    assert len(mask) == len(mapping)
    assert mask[0] == 0
