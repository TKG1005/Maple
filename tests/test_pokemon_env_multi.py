import sys
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_pokemon_env_multi(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "copy_of_poke-env"))

    # ------------------------------------------------------------------
    # Patch modules required for MapleAgentPlayer
    # ------------------------------------------------------------------
    class DummyBattleOrder:
        pass

    class DummyPlayer:
        def __init__(self, **kwargs):
            pass

        def create_order(self, *args, **kwargs):
            return DummyBattleOrder()

    dummy_battle_cls = MagicMock(name="AbstractBattle")

    monkeypatch.setitem(
        sys.modules,
        "poke_env.player",
        SimpleNamespace(
            Player=DummyPlayer,
            battle_order=SimpleNamespace(BattleOrder=DummyBattleOrder),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "poke_env.environment.battle",
        SimpleNamespace(Battle=dummy_battle_cls, AbstractBattle=dummy_battle_cls),
    )

    class DummyAgent:
        def select_action(self, state, mask):
            return 0

    monkeypatch.setitem(
        sys.modules,
        "src.agents.MapleAgent",
        SimpleNamespace(MapleAgent=DummyAgent),
    )

    class DummyObserver:
        def __init__(self, path):
            pass

        def observe(self, battle):
            return [0]

        def get_observation_dimension(self):
            return 1

    monkeypatch.setitem(
        sys.modules,
        "src.state.state_observer",
        SimpleNamespace(StateObserver=DummyObserver),
    )

    def fake_get_available_actions(battle):
        return [1] + [0] * 9, {}

    def fake_get_available_actions_with_details(battle):
        return [1] + [0] * 9, {}

    def fake_action_index_to_order(self, battle, idx):
        return DummyBattleOrder()

    action_helper_mod = SimpleNamespace(
        get_available_actions=fake_get_available_actions,
        get_available_actions_with_details=fake_get_available_actions_with_details,
        action_index_to_order=fake_action_index_to_order,
    )
    monkeypatch.setitem(sys.modules, "src.action.action_helper", action_helper_mod)
    monkeypatch.setitem(
        sys.modules, "src.action", SimpleNamespace(action_helper=action_helper_mod)
    )

    MapleAgentPlayer = importlib.import_module(
        "src.agents.maple_agent_player"
    ).MapleAgentPlayer
    player0 = MapleAgentPlayer(maple_agent=DummyAgent(), start_listening=False)
    player1 = MapleAgentPlayer(maple_agent=DummyAgent(), start_listening=False)

    # ------------------------------------------------------------------
    # Patch modules required for PokemonEnv import
    # ------------------------------------------------------------------
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
    sys.modules.setdefault("src.env.env_player", SimpleNamespace(EnvPlayer=object))

    from src.env.pokemon_env import PokemonEnv

    # Stub out heavy methods
    def dummy_reset(self, *args, **kwargs):
        return {"player_0": [0], "player_1": [0]}

    def dummy_step(self, action_dict):
        obs = {"player_0": [0], "player_1": [0]}
        reward = {"player_0": 0.0, "player_1": 0.0}
        term = {"player_0": False, "player_1": False}
        trunc = {"player_0": False, "player_1": False}
        info = {"player_0": {}, "player_1": {}}
        return obs, reward, term, trunc, info

    monkeypatch.setattr(PokemonEnv, "reset", dummy_reset)
    monkeypatch.setattr(PokemonEnv, "step", dummy_step)
    monkeypatch.setattr(PokemonEnv, "close", lambda self: None)

    env = PokemonEnv(
        opponent_player=player1,
        state_observer=DummyObserver("x"),
        action_helper=action_helper_mod,
        agent_players={"player_0": player0, "player_1": player1},
    )

    observation = env.reset()
    for _ in range(3):
        observation, reward, term, trunc, info = env.step(
            {"player_0": 0, "player_1": 0}
        )
        assert isinstance(observation, dict) and "player_0" in observation
    env.close()
