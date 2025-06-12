from __future__ import annotations

from typing import Any

from poke_env.player import Player
from poke_env.environment.battle import Battle

from .MapleAgent import MapleAgent


class MapleAgentPlayer(Player):
    """poke-env の :class:`Player` API で :class:`MapleAgent` を利用するラッパー。"""

    def __init__(self, maple_agent: "MapleAgent", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.maple_agent = maple_agent

    def choose_move(self, battle: Battle) -> Any:
        pass

    def choose_team(self, battle: Battle) -> Any:
        pass


__all__ = ["MapleAgentPlayer"]
