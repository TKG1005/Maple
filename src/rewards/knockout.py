from __future__ import annotations

from . import RewardBase


class KnockoutReward(RewardBase):
    """撃破と被撃破に基づいて報酬を計算するクラス。"""

    def reset(self, battle: object | None = None) -> None:
        """内部状態をリセットする。"""
        pass

    def calc(self, battle: object) -> float:
        """報酬を計算して返す。"""
        return 0.0


__all__ = ["KnockoutReward"]
