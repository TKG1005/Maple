from __future__ import annotations

import random
from typing import Any, Awaitable

from poke_env.environment.abstract_battle import AbstractBattle
from poke_env.player import Player


class EnvPlayer(Player):
    """poke_env Player subclass controlled via an action queue."""

    def __init__(self, env: Any, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._env = env

    async def choose_move(self, battle):
        """Return the order chosen by the external agent via :class:`PokemonEnv`."""

        # PokemonEnv に最新の battle オブジェクトを送信
        await self._env._battle_queue.put(battle)
        

        # PokemonEnv.step からアクションが投入されるまで待機
        action_idx: int = await self._env._action_queue.get()

        # 取得したインデックスを BattleOrder に変換して返す
        return self._env.action_helper.action_index_to_order(self, battle, action_idx)

    # Player クラスの _handle_battle_request をオーバーライド
    async def _handle_battle_request(
        self,
        battle: AbstractBattle,
        from_teampreview_request: bool = False,
        maybe_default_order: bool = False,
    ):
        print(
            f"[DBG]_handle_battle_request() called batte.teampreview={battle.teampreview},from_teampreview_request = {from_teampreview_request}"
        )

        if from_teampreview_request:
            # チームプレビュー開始通知は無視する
            return

        if maybe_default_order and (
            "illusion" in [p.ability for p in battle.team.values()]
            or random.random() < self.DEFAULT_CHOICE_CHANCE
        ):
            message = self.choose_default_move().message
        elif battle.teampreview:
            # チームプレビュー処理: 初回のみ MapleAgent に委任
            if not hasattr(self, "_teampreview_done"):
                self._teampreview_done = set()

            if battle.battle_tag not in self._teampreview_done:
                self._teampreview_done.add(battle.battle_tag)
                team_msg = None
                if getattr(self._env, "_agent", None) is not None:
                    team_msg = self._env._agent.teampreview(battle)
                    if isinstance(team_msg, Awaitable):
                        team_msg = await team_msg
                if team_msg is None:
                    team_msg = self.teampreview(battle)
                message = team_msg
            else:
                choice = self.choose_move(battle)
                if isinstance(choice, Awaitable):
                    choice = await choice
                message = choice.message
        else:
            if maybe_default_order:
                self._trying_again.set()
            choice = self.choose_move(battle)
            if isinstance(choice, Awaitable):
                choice = await choice
            message = choice.message

        await self.ps_client.send_message(message, battle.battle_tag)
