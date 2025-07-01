from __future__ import annotations

from . import RewardBase


class TurnPenaltyReward(RewardBase):
    """ターン経過にペナルティを与える報酬クラスの雛形。"""

    def __init__(self, penalty: float = -0.01) -> None:
        self.penalty = penalty
        self.turn_count = 0

    def reset(self, battle: object | None = None) -> None:
        """内部状態をリセットする。"""
        self.turn_count = 0

    def __call__(self, battle: object) -> float:
        """報酬を計算して返す。"""
        self.turn_count += 1
        return float(self.penalty)

    def calc(self, battle: object) -> float:  # pragma: no cover - thin wrapper
        """Alias for compatibility with :class:`RewardBase`."""
        return self.__call__(battle)


__all__ = ["TurnPenaltyReward"]
