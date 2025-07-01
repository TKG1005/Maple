from __future__ import annotations

from . import RewardBase


class TurnPenaltyReward(RewardBase):
    """ターン経過にペナルティを与える報酬クラスの雛形。"""

    def __init__(self, penalty: float = -0.01) -> None:
        self.penalty = penalty
        self.turn_count = 0
        self._last_turn: int | None = None

    def reset(self, battle: object | None = None) -> None:
        """内部状態をリセットする。"""
        self.turn_count = 0
        if battle is not None:
            self._last_turn = getattr(battle, "turn", 0) - 1
        else:
            self._last_turn = None

    def __call__(self, battle: object) -> float:
        """報酬を計算して返す。"""
        current_turn = getattr(battle, "turn", None)
        if current_turn is None or self._last_turn is None:
            self.turn_count += 1
            self._last_turn = current_turn
            return float(self.penalty)

        if current_turn > self._last_turn:
            self.turn_count += 1
            self._last_turn = current_turn
            return float(self.penalty)
        self._last_turn = current_turn
        return 0.0

    def calc(self, battle: object) -> float:  # pragma: no cover - thin wrapper
        """Alias for compatibility with :class:`RewardBase`."""
        return self.__call__(battle)


__all__ = ["TurnPenaltyReward"]
