import sys
from pathlib import Path
from types import SimpleNamespace

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "copy_of_poke-env"))

# Dummy modules for optional dependencies
sys.modules.setdefault(
    "numpy",
    SimpleNamespace(random=SimpleNamespace(default_rng=lambda seed=None: None), float32=float),
)

class DummySpace:
    def __init__(self, *a, **k):
        pass

gym_spaces = SimpleNamespace(Box=DummySpace, Discrete=DummySpace, Dict=lambda *a, **k: None)
sys.modules.setdefault("gymnasium", SimpleNamespace(Env=object, spaces=gym_spaces))
sys.modules.setdefault("gymnasium.spaces", gym_spaces)

sys.modules.setdefault("yaml", SimpleNamespace(safe_load=lambda *a, **k: {}))
sys.modules.setdefault("poke_env.concurrency", SimpleNamespace(POKE_LOOP=None))

# Minimal EnvPlayer stub to satisfy PokemonEnv import
sys.modules.setdefault("src.env.env_player", SimpleNamespace(EnvPlayer=object))

from src.env.pokemon_env import PokemonEnv


class DummyObserver:
    def get_observation_dimension(self):
        return 1


def test_compute_rewards_win():
    env = PokemonEnv(
        opponent_player=object(),
        state_observer=DummyObserver(),
        action_helper=object(),
    )

    battle = SimpleNamespace(finished=True, won=True)
    rewards = env._compute_rewards(battle)
    assert rewards[env.agent_ids[0]] == 1.0
    assert rewards[env.agent_ids[1]] == -1.0
