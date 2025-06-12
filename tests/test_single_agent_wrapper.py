import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root))

from src.env.wrappers import SingleAgentCompatibilityWrapper


class DummyEnv:
    def __init__(self):
        self.agent_ids = ("player_0", "player_1")
        self.action_space = {"player_0": "a0", "player_1": "a1"}
        self.observation_space = {"player_0": "o0", "player_1": "o1"}
        self.reset = MagicMock(return_value=("obs", {}))
        self.step = MagicMock(return_value=("obs", [0], 1.0, True, {}))


def test_wrapper_delegates_calls():
    env = DummyEnv()
    wrapper = SingleAgentCompatibilityWrapper(env)
    assert getattr(env, "single_agent_mode") is True
    assert wrapper.action_space == "a0"
    assert wrapper.observation_space == "o0"

    obs, info = wrapper.reset()
    env.reset.assert_called_once()
    assert obs == "obs"
    assert info == {}

    result = wrapper.step(1)
    env.step.assert_called_with({"player_0": 1})
    assert result[0] == "obs"

