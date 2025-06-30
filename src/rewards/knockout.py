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
        self.prev_my_hp.clear()
        self.prev_opp_hp.clear()
        self.prev_my_alive.clear()
        self.prev_opp_alive.clear()

        if battle is not None:
            for mon in getattr(battle, "team", {}).values():
                self.prev_my_hp[id(mon)] = getattr(mon, "current_hp", 0) or 0
                self.prev_my_alive[id(mon)] = not getattr(mon, "fainted", False)
            for mon in getattr(battle, "opponent_team", {}).values():
                self.prev_opp_hp[id(mon)] = getattr(mon, "current_hp", 0) or 0
                self.prev_opp_alive[id(mon)] = not getattr(mon, "fainted", False)

    def calc(self, battle: object) -> float:
        """報酬を計算して返す。"""
        enemy_kos = 0
        self_kos = 0

        for mon in getattr(battle, "team", {}).values():
            cur_hp = getattr(mon, "current_hp", 0) or 0
            alive = not getattr(mon, "fainted", cur_hp <= 0)
            prev_alive = self.prev_my_alive.get(id(mon), alive)
            if prev_alive and not alive:
                self_kos += 1
            self.prev_my_hp[id(mon)] = cur_hp
            self.prev_my_alive[id(mon)] = alive

        for mon in getattr(battle, "opponent_team", {}).values():
            cur_hp = getattr(mon, "current_hp", 0) or 0
            alive = not getattr(mon, "fainted", cur_hp <= 0)
            prev_alive = self.prev_opp_alive.get(id(mon), alive)
            if prev_alive and not alive:
                enemy_kos += 1
            self.prev_opp_hp[id(mon)] = cur_hp
            self.prev_opp_alive[id(mon)] = alive

        reward = (
            enemy_kos * self.ENEMY_KO_BONUS
            + self_kos * self.SELF_KO_PENALTY
        )
        return float(reward)


__all__ = ["KnockoutReward"]
