from __future__ import annotations

import numpy as np
from typing import Any
from poke_env.data import GenData
from poke_env.environment.move import Move
from poke_env.environment.move_category import MoveCategory

from .MapleAgent import MapleAgent
from src.action.action_helper import get_available_actions_with_details


class RuleBasedPlayer(MapleAgent):
    """ルールベースで行動を選択するエージェント。"""

    def __init__(self, env, gen: int = 9):
        super().__init__(env)
        self.type_chart = GenData.from_gen(gen).type_chart

    def _get_current_battle(self):
        """現在のバトル状態を取得する。"""
        player_id = self._get_player_id()
        env_player = self.env._env_players.get(player_id)
        if env_player and hasattr(env_player, 'current_battle'):
            return env_player.current_battle
        return None

    def _get_damage_multiplier(self, move: Move, target_pokemon) -> float:
        """技の対象への型相性倍率を計算する。"""
        if not target_pokemon or not move.type:
            return 1.0
        
        if move.category == MoveCategory.STATUS:
            return 1.0

        if hasattr(target_pokemon, 'is_terastallized') and target_pokemon.is_terastallized:
            return move.type.damage_multiplier(target_pokemon.tera_type, None, type_chart=self.type_chart)
        
        return move.type.damage_multiplier(target_pokemon.type_1, target_pokemon.type_2, type_chart=self.type_chart)

    def _find_super_effective_moves(self, battle, action_mask, action_details):
        """効果抜群の技をリストアップし、威力順にソートして返す。"""
        super_effective_moves = []
        opponent = battle.opponent_active_pokemon
        
        if not opponent:
            return []

        for action_idx, details in action_details.items():
            # マスクで無効な行動はスキップ
            if not action_mask[action_idx]:
                continue
                
            # 通常の技のみ考慮（テラスタル技は除く）
            if details["type"] != "move" or details["id"] == "struggle":
                continue
            
            # 技オブジェクトを取得
            move = None
            for available_move in battle.available_moves:
                if available_move.id == details["id"]:
                    move = available_move
                    break
            
            if not move or move.category == MoveCategory.STATUS:
                continue
            
            # 型相性を計算
            multiplier = self._get_damage_multiplier(move, opponent)
            
            if multiplier > 1.0:  # 効果抜群
                power = move.base_power or 0
                super_effective_moves.append((action_idx, move, power, multiplier))
        
        # 威力順にソート（降順）
        return sorted(super_effective_moves, key=lambda x: x[2], reverse=True)

    def _find_attacking_moves(self, battle, action_mask, action_details):
        """利用可能な攻撃技をリストアップして返す。"""
        attacking_moves = []
        
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
            
            if move and move.category != MoveCategory.STATUS:
                attacking_moves.append(action_idx)
        
        return attacking_moves

    def select_action(self, observation: Any, action_mask: Any) -> int:
        """
        ルールに基づいて行動を選択する。
        ルール1: 相手に効果抜群の利用可能な技があれば、その中で最も威力の高い技を選択。
        ルール2: 効果抜群の技がなければ、利用可能な攻撃技からランダムに選択。
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

        # ルール1: 効果抜群の技を探す
        super_effective_moves = self._find_super_effective_moves(battle, action_mask, action_details)
        
        if super_effective_moves:
            # 最も威力の高い効果抜群技を選択
            chosen_action, chosen_move, power, multiplier = super_effective_moves[0]
            return chosen_action

        # ルール2: 効果抜群技がなければ攻撃技からランダム選択
        attacking_moves = self._find_attacking_moves(battle, action_mask, action_details)
        
        if attacking_moves:
            return int(self.env.rng.choice(attacking_moves))

        # 攻撃技もない場合は有効な行動からランダム選択
        valid_actions = np.where(action_mask)[0]
        return int(self.env.rng.choice(valid_actions)) if len(valid_actions) > 0 else 0

    def act(self, observation: Any, action_mask: Any) -> int:
        return self.select_action(observation, action_mask)


__all__ = ["RuleBasedPlayer"]