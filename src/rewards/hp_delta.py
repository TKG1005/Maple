from __future__ import annotations

from typing import Dict

from . import RewardBase


class HPDeltaReward(RewardBase):
    """HP の増減に基づいて報酬を計算するクラス。"""

    SELF_DAMAGE_COEF = -0.01
    SELF_HEAL_COEF = 0.01
    ENEMY_DAMAGE_COEF = 0.01
    ENEMY_HEAL_COEF = -0.01

    def __init__(self) -> None:
        self.prev_my_hp: Dict[int, int] = {}
        self.prev_opp_hp: Dict[int, int] = {}
        self.self_damage = 0.0
        self.self_heal = 0.0
        self.enemy_damage = 0.0
        self.enemy_heal = 0.0

    def reset(self, battle: object | None = None) -> None:
        """状態をリセットする。``battle`` が与えられれば HP を初期化する。"""

        self.prev_my_hp.clear()
        self.prev_opp_hp.clear()
        self.self_damage = 0.0
        self.self_heal = 0.0
        self.enemy_damage = 0.0
        self.enemy_heal = 0.0

        if battle is not None:
            for mon in getattr(battle, "team", {}).values():
                self.prev_my_hp[id(mon)] = getattr(mon, "current_hp", 0) or 0
            for mon in getattr(battle, "opponent_team", {}).values():
                self.prev_opp_hp[id(mon)] = getattr(mon, "current_hp", 0) or 0

    def calc(self, battle: object) -> float:
        """前ターンからの HP 変化量に応じて報酬を計算する。"""

        self_damage = 0.0
        self_heal = 0.0
        enemy_damage = 0.0
        enemy_heal = 0.0

        for mon in getattr(battle, "team", {}).values():
            cur = getattr(mon, "current_hp", 0) or 0
            max_hp = getattr(mon, "max_hp", 1) or 1
            prev = self.prev_my_hp.get(id(mon), cur)
            delta = cur - prev
            if delta < 0:
                self_damage += -delta / max_hp
            elif delta > 0:
                self_heal += delta / max_hp
            self.prev_my_hp[id(mon)] = cur

        for mon in getattr(battle, "opponent_team", {}).values():
            cur = getattr(mon, "current_hp", 0) or 0
            max_hp = getattr(mon, "max_hp", 1) or 1
            prev = self.prev_opp_hp.get(id(mon), cur)
            delta = cur - prev
            if delta < 0:
                enemy_damage += -delta / max_hp
            elif delta > 0:
                enemy_heal += delta / max_hp
            self.prev_opp_hp[id(mon)] = cur

        self.self_damage = self_damage
        self.self_heal = self_heal
        self.enemy_damage = enemy_damage
        self.enemy_heal = enemy_heal

        reward = (
            self_damage * self.SELF_DAMAGE_COEF
            + self_heal * self.SELF_HEAL_COEF
            + enemy_damage * self.ENEMY_DAMAGE_COEF
            + enemy_heal * self.ENEMY_HEAL_COEF
        )
        return float(reward)


__all__ = ["HPDeltaReward"]
