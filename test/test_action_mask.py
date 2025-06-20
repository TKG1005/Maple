import sys
import types
from types import SimpleNamespace

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - fallback minimal numpy stub
    np = types.ModuleType("numpy")
    def array(seq, dtype=None):
        return list(seq)
    def zeros(n, dtype=None):
        return [0] * n
    class _RNG:
        def __init__(self, seed=None):
            import random
            self._rand = random.Random(seed)
        def choice(self, seq, size=None, replace=True, p=None):
            if size is None:
                return self._rand.choice(list(seq))
            return [self._rand.choice(list(seq)) for _ in range(size)]
    np.array = array
    np.zeros = zeros
    np.int8 = int
    np.float32 = float
    np.random = types.SimpleNamespace(default_rng=lambda seed=None: _RNG(seed))
    sys.modules['numpy'] = np

try:
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover - minimal gymnasium stub
    gym = types.ModuleType("gymnasium")
    class Box:
        def __init__(self, low, high, shape=None, dtype=float):
            self.low = low
            self.high = high
            self.shape = shape
            self.dtype = dtype
        def sample(self):
            return 0
    class Dict(dict):
        pass
    class Discrete:
        def __init__(self, n):
            self.n = n
        def sample(self):
            return 0
    gym.Env = object
    gym.spaces = types.SimpleNamespace(Box=Box, Dict=Dict, Discrete=Discrete)
    sys.modules['gymnasium'] = gym
    sys.modules['gymnasium.spaces'] = gym.spaces

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - minimal yaml stub
    yaml = types.ModuleType("yaml")
    def safe_load(stream):
        return {}
    yaml.safe_load = safe_load
    sys.modules['yaml'] = yaml

try:
    import poke_env.concurrency  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - minimal poke_env stub
    import asyncio
    poke_env = types.ModuleType("poke_env")
    concurrency = types.ModuleType("concurrency")
    concurrency.POKE_LOOP = asyncio.new_event_loop()
    player_mod = types.ModuleType("player")
    class Player:
        DEFAULT_CHOICE_CHANCE = 0.0
        def __init__(self, *args, **kwargs):
            pass
    player_mod.Player = Player
    environment = types.ModuleType("environment")
    abstract_battle = types.ModuleType("abstract_battle")
    class AbstractBattle: ...
    abstract_battle.AbstractBattle = AbstractBattle
    environment.abstract_battle = abstract_battle
    poke_env.concurrency = concurrency
    poke_env.player = player_mod
    poke_env.environment = environment
    sys.modules['poke_env'] = poke_env
    sys.modules['poke_env.concurrency'] = concurrency
    sys.modules['poke_env.player'] = player_mod
    sys.modules['poke_env.environment'] = environment
    sys.modules['poke_env.environment.abstract_battle'] = abstract_battle

from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.env.pokemon_env import PokemonEnv

class DummyActionHelper:
    @staticmethod
    def get_available_actions(battle):
        mask = [0] * 10
        if not battle.force_switch:
            for i in range(min(4, len(battle.available_moves))):
                mask[i] = 1
            if battle.can_tera:
                for i in range(min(4, len(battle.available_moves))):
                    mask[4 + i] = 1
        for i in range(min(2, len(battle.available_switches))):
            mask[8 + i] = 1
        mapping = {}
        if not battle.force_switch:
            for i in range(min(4, len(battle.available_moves))):
                mapping[i] = ("move", i)
            if battle.can_tera:
                for i in range(min(4, len(battle.available_moves))):
                    mapping[4 + i] = ("terastal", i)
        for i in range(min(2, len(battle.available_switches))):
            mapping[8 + i] = ("switch", i)
        return mask, mapping

class DummyObserver:
    def get_observation_dimension(self):
        return 1
    def observe(self, battle):
        return np.array([0], dtype=np.float32)

class DummyMove:
    def __init__(self, mid):
        self.id = mid

class DummyPokemon:
    def __init__(self, species):
        self.species = species

def make_env():
    return PokemonEnv(state_observer=DummyObserver(), action_helper=DummyActionHelper, opponent_player=None)

def test_mask_respects_team_selection():
    env = make_env()
    battle = SimpleNamespace(
        available_moves=[DummyMove("tackle")],
        available_switches=[DummyPokemon("A"), DummyPokemon("B")],
        force_switch=False,
        can_tera=True,
    )
    env._current_battles = {"player_0": battle}
    env._selected_species["player_0"] = {"B"}
    mask, mapping = env.get_action_mask("player_0")
    assert mapping[8] == ("switch", 0)
    assert mapping[9] == ("switch", 1)
    assert mask[8] == 0  # A not selected
    assert mask[9] == 1  # B selected


def test_mask_force_switch():
    env = make_env()
    battle = SimpleNamespace(
        available_moves=[DummyMove("tackle")],
        available_switches=[DummyPokemon("A"), DummyPokemon("B")],
        force_switch=True,
        can_tera=True,
    )
    env._current_battles = {"player_0": battle}
    env._selected_species["player_0"] = {"A"}
    mask, mapping = env.get_action_mask("player_0")
    # move slots disabled
    assert sum(mask[:4]) == 0
    # only selected switch allowed
    assert mask[8] == 1
    assert mask[9] == 0
