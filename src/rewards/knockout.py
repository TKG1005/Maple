from __future__ import annotations

from typing import Dict, Any

from . import RewardBase


class KnockoutReward(RewardBase):
    """撃破/被撃破に基づいて報酬を計算するクラス。"""

    SELF_FAINT_COEF = -0.5
    ENEMY_FAINT_COEF = 1.0

    def __init__(self) -> None:
        # ポケモンIDセット
        self.my_ids: Dict[int, bool] = {}
        self.opp_ids: Dict[int, bool] = {}

        # 直前ターンのHPと生存状態
        self.prev_my_hp: Dict[int, int] = {}
        self.prev_opp_hp: Dict[int, int] = {}
        self.prev_my_alive: Dict[int, bool] = {}
        self.prev_opp_alive: Dict[int, bool] = {}

        self.self_fainted = 0.0
        self.enemy_fainted = 0.0

    def reset(self, battle: Any | None = None) -> None:
        self.my_ids.clear()
        self.opp_ids.clear()
        self.prev_my_hp.clear()
        self.prev_opp_hp.clear()
        self.prev_my_alive.clear()
        self.prev_opp_alive.clear()
        self.self_fainted = 0.0
        self.enemy_fainted = 0.0

        if battle is not None:
            for mon in getattr(battle, "team", {}).values():
                ident = id(mon)
                self.my_ids[ident] = True
                self.prev_my_hp[ident] = getattr(mon, "current_hp", 0) or 0
                self.prev_my_alive[ident] = not getattr(mon, "fainted", False)
            for mon in getattr(battle, "opponent_team", {}).values():
                ident = id(mon)
                self.opp_ids[ident] = True
                self.prev_opp_hp[ident] = getattr(mon, "current_hp", 0) or 0
                self.prev_opp_alive[ident] = not getattr(mon, "fainted", False)

    def calc(self, battle: Any) -> float:  # pragma: no cover - not implemented
        return 0.0


__all__ = ["KnockoutReward"]
