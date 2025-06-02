import os
import numpy as np
import types

import pytest

from src.environments import pokemon_env
from src.action import action_helper
from src.state.state_observer import StateObserver


class DummyBackend:
    def __init__(self, env_ref, opponent_player, state_observer, action_helper, **kwargs):
        self.env_ref = env_ref
        self.obs_dim = state_observer.get_observation_dimension()
        self.step_count = 0

    def sync_reset(self, seed=None, options=None):
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        info = {"dummy": True}
        return obs, info

    def sync_step(self, action_idx):
        self.step_count += 1
        obs = np.full(self.obs_dim, self.step_count, dtype=np.float32)
        reward = float(action_idx)
        terminated = True
        truncated = False
        info = {"step": self.step_count}
        return obs, reward, terminated, truncated, info

    def sync_close(self):
        pass

    def render(self):
        pass


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setattr(pokemon_env, "_AsyncPokemonBackend", DummyBackend)
    spec_path = os.path.join(os.path.dirname(__file__), "..", "config", "state_spec.yml")
    observer = StateObserver(spec_path)
    dummy_opponent = object()
    env = pokemon_env.PokemonEnv(
        opponent_player=dummy_opponent,
        state_observer=observer,
        action_helper=action_helper,
        battle_format="gen9ou",
        team_pascal=None,
        player_username="TestPlayer",
    )
    yield env
    env.close()


def test_pokemon_env_single_episode(env):
    obs, info = env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape[0] == env.observation_space.shape[0]

    obs, reward, terminated, truncated, info = env.step(0)
    assert isinstance(obs, np.ndarray)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert terminated is True
    assert truncated is False

