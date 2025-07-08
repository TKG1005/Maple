from __future__ import annotations

import numpy as np
from typing import Any
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory

from src.agents.MapleAgent import MapleAgent
from src.action.action_helper import get_available_actions_with_details


class MaxDamageBot(MapleAgent):
    """最大威力の技を選択するボット。poke-envのMaxBasePowerPlayerと同等のロジック。"""

    def _get_current_battle(self):
        """現在のバトル状態を取得する。"""
        player_id = self._get_player_id()
        env_player = self.env._env_players.get(player_id)
        if env_player and hasattr(env_player, 'current_battle'):
            return env_player.current_battle
        return None

    def _find_max_damage_move(self, battle, action_mask, action_details):
        """最大威力の攻撃技を見つけて返す。"""
        max_power = -1
        best_action = None
        
        for action_idx, details in action_details.items():
            # マスクで無効な行動はスキップ
            if not action_mask[action_idx]:
                continue
                
            # 通常の技のみ考慮
            if details["type"] != "move" or details["id"] == "struggle":
                continue
            
            # 技オブジェクトを取得
            move = None
            for available_move in battle.available_moves:
                if available_move.id == details["id"]:
                    move = available_move
                    break
            
            # 攻撃技の場合、威力を比較
            if move and move.category != MoveCategory.STATUS:
                power = move.base_power or 0
                if power > max_power:
                    max_power = power
                    best_action = action_idx
        
        return best_action

    def select_action(self, observation: Any, action_mask: Any) -> int:
        """
        最大威力の攻撃技を選択する。
        攻撃技がない場合はランダムに有効な行動を選択する。
        """
        # 現在のバトル状態を取得
        battle = self._get_current_battle()
        if not battle:
            # バトル情報が取得できない場合はランダム選択
            valid_actions = np.where(action_mask)[0]
            return int(self.env.rng.choice(valid_actions)) if len(valid_actions) > 0 else 0

        # 利用可能な行動の詳細を取得
        try:
            _, action_details = get_available_actions_with_details(battle)
        except Exception:
            # エラーが発生した場合はランダム選択
            valid_actions = np.where(action_mask)[0]
            return int(self.env.rng.choice(valid_actions)) if len(valid_actions) > 0 else 0

        # 最大威力の技を探す
        best_action = self._find_max_damage_move(battle, action_mask, action_details)
        
        if best_action is not None:
            return best_action

        # 攻撃技がない場合は有効な行動からランダム選択
        valid_actions = np.where(action_mask)[0]
        return int(self.env.rng.choice(valid_actions)) if len(valid_actions) > 0 else 0

    def act(self, observation: Any, action_mask: Any) -> int:
        """select_actionと同じ処理を行う。"""
        return self.select_action(observation, action_mask)