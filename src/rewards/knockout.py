from __future__ import annotations

from typing import Dict

from . import RewardBase


class KnockoutReward(RewardBase):
    """撃破と被撃破に基づいて報酬を計算するクラス。"""

    ENEMY_KO_BONUS = 1.0
    SELF_KO_PENALTY = -0.5

    def __init__(self) -> None:
        # 前ターンの HP と生存状態を記録しておく
        self.prev_my_hp: Dict[int, int] = {}
        self.prev_opp_hp: Dict[int, int] = {}
        self.prev_my_alive: Dict[int, bool] = {}
        self.prev_opp_alive: Dict[int, bool] = {}

    def reset(self, battle: object | None = None) -> None:
        """内部状態をリセットする。"""
        pass

    def calc(self, battle: object) -> float:
        """報酬を計算して返す。"""
        return 0.0


__all__ = ["KnockoutReward"]
