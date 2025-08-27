from __future__ import annotations

from typing import Dict

from . import RewardBase


class KnockoutReward(RewardBase):
    """撃破と被撃破に基づいて報酬を計算するクラス。

    Parameters
    ----------
    enemy_ko_bonus : float
        相手ポケモンを倒したときのボーナス
    self_ko_penalty : float
        自分のポケモンが倒れたときのペナルティ（負の値）
    """

    def __init__(self, *, enemy_ko_bonus: float = 1.0, self_ko_penalty: float = -1.0) -> None:
        # 前ターンの HP と生存状態を記録しておく
        self.enemy_ko_bonus = float(enemy_ko_bonus)
        self.self_ko_penalty = float(self_ko_penalty)
        self.prev_my_hp: Dict[int, int] = {}
        self.prev_opp_hp: Dict[int, int] = {}
        self.prev_my_alive: Dict[int, bool] = {}
        self.prev_opp_alive: Dict[int, bool] = {}
        self.enemy_ko = 0
        self.self_ko = 0

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

        reward = enemy_kos * self.enemy_ko_bonus + self_kos * self.self_ko_penalty
        return float(reward)



__all__ = ["KnockoutReward"]
