import asyncio
from types import SimpleNamespace
import pytest

pytest.importorskip("numpy")
pytest.importorskip("gymnasium")
import numpy as np

from src.env import pokemon_env
from src.env.pokemon_env import PokemonEnv


class DummyObserver:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def get_observation_dimension(self) -> int:
        return self.dim

    def observe(self, battle) -> np.ndarray:
        return np.zeros(self.dim, dtype=np.float32)


class DummyActionHelper:
    def action_index_to_order(self, *args, **kwargs):
        return "order"

    def get_available_actions_with_details(self, battle):
        return np.zeros(10, dtype=np.int8), {}


class DummyBattle:
    def __init__(self) -> None:
        self.turn = 1
        self.finished = False
        self.won = False
        self.available_moves = []
        self.available_switches = []
        self.can_tera = False
        self.force_switch = False
        self.team = {}
        self.opponent_team = {}
        self.active_pokemon = None
        self.opponent_active_pokemon = None
        self.battle_tag = "dummy"


class DummyEnvPlayer:
    def __init__(self, env, player_id, *_, **__):
        self._env = env
        self.player_id = player_id
        self._waiting = asyncio.Event()
        self._trying_again = asyncio.Event()
        async def stop_listening():
            return None
        self.ps_client = SimpleNamespace(stop_listening=stop_listening)

    async def battle_against(self, other, n_battles=1):
        battle = DummyBattle()
        await self._env._battle_queues[self.player_id].put(battle)
        await self._env._battle_queues[other.player_id].put(battle)
        self._waiting.set()
        other._trying_again.set()


async def dummy_run_battle(self):
    battle = DummyBattle()
    await self._battle_queues["player_0"].put(battle)
    await self._battle_queues["player_1"].put(battle)


def setup_dummy_env(monkeypatch):
    monkeypatch.setattr(pokemon_env, "EnvPlayer", DummyEnvPlayer)
    monkeypatch.setattr(PokemonEnv, "_run_battle", dummy_run_battle)
    return PokemonEnv(
        opponent_player=None,
        state_observer=DummyObserver(3),
        action_helper=DummyActionHelper(),
    )


def test_reset_step_close(monkeypatch):
    env = setup_dummy_env(monkeypatch)
    obs, info = env.reset()

    assert set(obs.keys()) == {"player_0", "player_1"}
    assert obs["player_0"].shape == (3,)
    assert info.get("request_teampreview")

    observations, rewards, terms, truncs, infos = env.step({"player_0": 0, "player_1": 0})
    assert observations["player_0"].shape == (3,)
    assert rewards == {"player_0": 0.0, "player_1": 0.0}
    assert terms == {"player_0": False, "player_1": False}
    assert truncs == {"player_0": False, "player_1": False}

    env.close()
    assert env._action_queues == {}
    assert env._battle_queues == {}
