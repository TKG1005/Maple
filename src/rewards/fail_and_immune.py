from __future__ import annotations

from . import RewardBase


class FailAndImmuneReward(RewardBase):
    """無効行動時のペナルティを与える報酬クラス。"""

    def __init__(self, penalty: float = -0.02) -> None:
        self.penalty = penalty

    def reset(self, battle: object | None = None) -> None:
        """内部状態をリセットする。"""
        pass  # この報酬クラスは状態を持たない

    def calc(self, battle: object) -> float:
        """報酬を計算して返す。
        
        battle.last_fail_actionまたはbattle.last_immune_actionがTrueの場合、ペナルティを返す。
        """
        fail_action = hasattr(battle, 'last_fail_action') and getattr(battle, 'last_fail_action', False)
        immune_action = hasattr(battle, 'last_immune_action') and getattr(battle, 'last_immune_action', False)
        
        if fail_action or immune_action:
            return float(self.penalty)
        return 0.0


__all__ = ["FailAndImmuneReward"]