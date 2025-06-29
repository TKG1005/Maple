import sys
from pathlib import Path
import pytest

pytest.importorskip("numpy")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.env.pokemon_env import PokemonEnv
from src.rewards import RewardBase


class DummyObserver:
    def get_observation_dimension(self):
        return 1

    def observe(self, battle):
        return [0.0]


class DummyActionHelper:
    def get_available_actions_with_details(self, battle):
        return [1], {0: ("move", 0, False)}


class DummyComposite(RewardBase):
    def __init__(self):
        self.last_values = {"sub": 0.0}

    def reset(self, battle=None):
        pass

    def calc(self, battle):
        self.last_values["sub"] = 0.5
        return 0.5


def test_env_calc_composite_reward():
    env = PokemonEnv(DummyObserver(), DummyActionHelper(), reward="composite", reward_config_path="dummy")
    env._composite_rewards = {"player_0": DummyComposite(), "player_1": DummyComposite()}
    env._sub_reward_logs = {"player_0": {}, "player_1": {}}

    reward = env._calc_reward(object(), "player_0")
    assert reward == 0.5
    assert env._sub_reward_logs["player_0"]["sub"] == 0.5
