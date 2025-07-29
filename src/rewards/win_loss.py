from __future__ import annotations

from typing import Any

from . import RewardBase


class WinLossReward(RewardBase):
    """勝敗に基づく報酬を提供するクラス。
    
    バトルの勝敗に基づいて絶対的な報酬を与える。
    勝利時は大きな正の報酬、敗北時は大きな負の報酬を与える。
    """

    def __init__(self, win_reward: float = 30.0, loss_penalty: float = -30.0) -> None:
        """勝敗報酬を初期化する。
        
        Args:
            win_reward: 勝利時の報酬（正の値）
            loss_penalty: 敗北時のペナルティ（負の値）
        """
        self.win_reward = float(win_reward)
        self.loss_penalty = float(loss_penalty)

    def reset(self, battle: Any | None = None) -> None:
        """リセット処理（特に状態を持たないため何もしない）。"""
        pass

    def calc(self, battle: Any) -> float:
        """バトルの勝敗に基づいて報酬を計算する。
        
        Args:
            battle: バトルオブジェクト
            
        Returns:
            勝利時は win_reward、敗北時は loss_penalty、継続中は 0.0
        """
        # バトルが終了していない場合は報酬なし
        if not getattr(battle, "finished", False):
            return 0.0
            
        # 勝敗に基づく報酬
        if getattr(battle, "won", False):
            return self.win_reward
        else:
            return self.loss_penalty


__all__ = ["WinLossReward"]