from __future__ import annotations

from . import RewardBase


class SwitchPenaltyReward(RewardBase):
    """交代行動が連続で続く場合にペナルティを与える報酬クラス。
    
    7回以上連続で交代行動をした場合、その後1回ごとに指定されたペナルティを与える。
    交代以外の行動を選んだ場合はカウンターをリセットする。
    """

    def __init__(self, penalty: float = -1.0, threshold: int = 7) -> None:
        """
        Parameters
        ----------
        penalty : float
            連続交代のペナルティ値（負の値）
        threshold : int
            ペナルティを開始する連続交代回数の閾値
        """
        self.penalty = penalty
        self.threshold = threshold
        self.consecutive_switches = 0
        self._last_active_pokemon = None
        self._last_turn = None

    def reset(self, battle: object | None = None) -> None:
        """内部状態をリセットする。"""
        self.consecutive_switches = 0
        self._last_active_pokemon = None
        self._last_turn = None

    def _is_switch_action(self, battle: object) -> bool:
        """交代行動かどうかを判定する。"""
        try:
            # 現在のターン数を取得
            current_turn = getattr(battle, 'turn', 0)
            
            # 現在のアクティブポケモンを取得
            current_active = None
            if hasattr(battle, 'active_pokemon') and battle.active_pokemon:
                current_active = getattr(battle.active_pokemon, 'species', None)
            
            # 初回の場合は交代ではない
            if self._last_active_pokemon is None:
                self._last_active_pokemon = current_active
                self._last_turn = current_turn
                return False
            
            # ターンが進んでいない場合は判定しない
            if current_turn == self._last_turn:
                return False
            
            # アクティブポケモンが変わった場合は交代と判定
            is_switch = (current_active != self._last_active_pokemon and 
                        current_active is not None and 
                        self._last_active_pokemon is not None)
            
            # 状態を更新
            self._last_active_pokemon = current_active
            self._last_turn = current_turn
            
            return is_switch
            
        except Exception:
            # エラーが発生した場合は安全に交代ではないとする
            return False

    def calc(self, battle: object) -> float:
        """報酬を計算して返す。"""
        is_switch = self._is_switch_action(battle)
        
        # 行動タイプに基づいてカウンターを更新
        if is_switch:
            self.consecutive_switches += 1
        else:
            # 交代以外の行動（技使用）でリセット
            self.consecutive_switches = 0
        
        # 閾値を超えた連続交代に対してペナルティを与える
        if self.consecutive_switches > self.threshold:
            return float(self.penalty)
        
        return 0.0


__all__ = ["SwitchPenaltyReward"]