from types import ModuleType, SimpleNamespace
import sys
import asyncio
import logging
from threading import Thread
from pathlib import Path

# Ensure repo root on path
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Minimal stubs
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
gym_stub.spaces = SimpleNamespace(Box=lambda *a, **k: None, Dict=dict, Discrete=lambda n: None)
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
sys.modules.setdefault("poke_env", poke_env_stub)
sys.modules.setdefault("poke_env.concurrency", concurrency)

from src.env.pokemon_env import PokemonEnv


def test_race_get_returns_last_message():
    loop = asyncio.new_event_loop()
    t = Thread(target=loop.run_forever)
    t.start()
    try:
        import src.env.pokemon_env as penv
        old_loop = penv.POKE_LOOP
        penv.POKE_LOOP = loop
        env = PokemonEnv.__new__(PokemonEnv)
        env.timeout = 1.0
        env._logger = logging.getLogger(__name__)
        q = asyncio.Queue()
        event = asyncio.Event()
        asyncio.run_coroutine_threadsafe(q.put(1), loop).result()
        asyncio.run_coroutine_threadsafe(q.put(2), loop).result()
        result = env._race_get(q, event)
        assert result == 2
        assert q.empty()
        penv.POKE_LOOP = old_loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        t.join()
