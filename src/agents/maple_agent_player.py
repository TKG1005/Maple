from __future__ import annotations

from typing import Any
from pathlib import Path
import logging

from poke_env.player import Player
from poke_env.environment.battle import Battle

from .MapleAgent import MapleAgent
from src.state.state_observer import StateObserver
from src.action import action_helper


class MapleAgentPlayer(Player):
    """poke-env の :class:`Player` API で :class:`MapleAgent` を利用するラッパー。"""

    def __init__(self, maple_agent: "MapleAgent", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.maple_agent = maple_agent
        # StateObserver と ActionHelper (モジュール) を保持
        state_spec = Path(__file__).resolve().parents[2] / "config" / "state_spec.yml"
        self._observer = StateObserver(str(state_spec))
        self._helper = action_helper
        self._logger = logging.getLogger(__name__)

    async def _handle_battle_request(
        self,
        battle: Battle,
        from_teampreview_request: bool = False,
        maybe_default_order: bool = False,
    ):
        """Handle battle requests with additional debug output."""

        if from_teampreview_request:
            message = self.teampreview(battle)
            self._logger.debug(
                "%s for %s", message, battle.battle_tag
            )
            await self.ps_client.send_message(message, battle.battle_tag)
            return

        return await super()._handle_battle_request(
            battle,
            from_teampreview_request=from_teampreview_request,
            maybe_default_order=maybe_default_order,
        )

    def choose_move(self, battle: Battle) -> Any:
        # battle から状態ベクトルと利用可能アクションを取得
        state = self._observer.observe(battle)
        mask, mapping = self._helper.get_available_actions(battle)
        

        # MapleAgent で行動インデックスを選択
        idx = self.maple_agent.select_action(state, mask)
        

        # インデックスを BattleOrder に変換
        try:
            order = self._helper.action_index_to_order(self, battle, idx)
        except Exception:
            raise
        
        return order

    async def choose_team(self, battle) -> str:
        """チームプレビューに応答し、先頭3匹を選出する。"""

        # TODO: MapleAgent.choose_team に差し替え予定
        team = "/team 123"
        return team

    def teampreview(self, battle: Battle) -> str:
        """チームプレビューでの選択を :class:`MapleAgent` に委譲する。"""
        obs = self._observer.observe(battle)
        team_cmd = self.maple_agent.choose_team(obs)
        return team_cmd


__all__ = ["MapleAgentPlayer"]
