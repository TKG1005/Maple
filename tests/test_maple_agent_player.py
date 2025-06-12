import sys
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


def test_choose_move_returns_battleorder(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "copy_of_poke-env"))

    class DummyBattleOrder:  # return type stub
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

    monkeypatch.setitem(
        sys.modules,
        "src.state.state_observer",
        SimpleNamespace(StateObserver=DummyObserver),
    )

    def fake_get_available_actions(battle):
        return [1] + [0] * 9, {}

    def fake_action_index_to_order(self, battle, idx):
        return DummyBattleOrder()

    action_helper_mod = SimpleNamespace(
        get_available_actions=fake_get_available_actions,
        action_index_to_order=fake_action_index_to_order,
    )
    monkeypatch.setitem(sys.modules, "src.action.action_helper", action_helper_mod)
    monkeypatch.setitem(
        sys.modules, "src.action", SimpleNamespace(action_helper=action_helper_mod)
    )

    MapleAgentPlayer = importlib.import_module(
        "src.agents.maple_agent_player"
    ).MapleAgentPlayer
    player = MapleAgentPlayer(maple_agent=DummyAgent(), start_listening=False)

    fake_order = DummyBattleOrder()
    monkeypatch.setattr(player._observer, "observe", MagicMock(return_value=[0]))
    monkeypatch.setattr(
        player._helper,
        "get_available_actions",
        MagicMock(return_value=([1] + [0] * 9, {})),
    )
    monkeypatch.setattr(
        player._helper, "action_index_to_order", MagicMock(return_value=fake_order)
    )

    battle = dummy_battle_cls
    result = player.choose_move(battle)
    assert isinstance(result, DummyBattleOrder)
