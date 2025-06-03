import sys
import types
import pathlib
import pytest

# Ensure src is in path
ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

# Fixture to provide dummy numpy and gymnasium modules if missing
@pytest.fixture(autouse=True)
def dummy_modules(monkeypatch):
    try:
        import numpy  # type: ignore
    except Exception:
        numpy = types.ModuleType("numpy")
        class DummyRNG:
            def __init__(self, seed=None):
                pass
        numpy.random = types.SimpleNamespace(default_rng=lambda seed=None: DummyRNG())
        numpy.float32 = float
        numpy.zeros = lambda shape, dtype=float: [0] * shape[0]
        monkeypatch.setitem(sys.modules, "numpy", numpy)

    try:
        import gymnasium  # type: ignore
    except Exception:
        gymnasium = types.ModuleType("gymnasium")
        class DummyEnv:
            pass
        class DummyBox:
            def __init__(self, low, high, shape, dtype):
                self.low = low
                self.high = high
                self.shape = shape
                self.dtype = dtype
            def contains(self, x):
                return len(x) == self.shape[0]
        gymnasium.Env = DummyEnv
        gymnasium.spaces = types.SimpleNamespace(Box=DummyBox)
        monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)

    yield


def test_import_pokemon_env():
    from env.pokemon_env import PokemonEnv
    assert PokemonEnv is not None


def test_env_instantiation():
    from env.pokemon_env import PokemonEnv

    class DummyObserver:
        def __init__(self, dim=3):
            self.dim = dim
        def get_observation_dimension(self):
            return self.dim
        def observe(self, battle=None):
            return [0] * self.dim

    class DummyActionHelper:
        def action_index_to_order(self, idx):
            return idx

    env = PokemonEnv(object(), DummyObserver(), DummyActionHelper())
    assert env.observation_space.shape == (3,)


def test_observation_space_contains():
    from env.pokemon_env import PokemonEnv

    class DummyObserver:
        def __init__(self, dim=4):
            self.dim = dim
        def get_observation_dimension(self):
            return self.dim
        def observe(self, battle=None):
            return [0] * self.dim

    class DummyActionHelper:
        def action_index_to_order(self, idx):
            return idx

    env = PokemonEnv(object(), DummyObserver(4), DummyActionHelper())
    dummy_state = [0] * 4
    assert env.observation_space.contains(dummy_state)
