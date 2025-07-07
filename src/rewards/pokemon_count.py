from __future__ import annotations

from . import RewardBase


class PokemonCountReward(RewardBase):
    """対戦終了時の残りポケモン数の差に基づいて報酬を計算するクラス。
    
    ポケモンの差が1匹なら0点、2匹なら2点、3匹なら5点を報酬またはペナルティとして付与。
    """

    def __init__(self) -> None:
        pass

    def reset(self, battle: object | None = None) -> None:
        """内部状態をリセットする。このクラスは状態を持たないため何もしない。"""
        pass

    def calc(self, battle: object) -> float:
        """対戦終了時の残りポケモン数の差に基づいて報酬を計算する。"""
        
        # 対戦が終了していない場合は報酬なし
        if not getattr(battle, "finished", False):
            return 0.0
        
        # 自分の残りポケモン数をカウント
        my_remaining = 0
        for mon in getattr(battle, "team", {}).values():
            if not getattr(mon, "fainted", False):
                my_remaining += 1
        
        # 相手の残りポケモン数をカウント
        opp_remaining = 0
        for mon in getattr(battle, "opponent_team", {}).values():
            if not getattr(mon, "fainted", False):
                opp_remaining += 1
        
        # ポケモン数の差を計算（正の値は自分が有利）
        pokemon_diff = my_remaining - opp_remaining
        
        # 差に基づいて報酬を計算
        if abs(pokemon_diff) <= 1:
            return 0.0
        elif abs(pokemon_diff) == 2:
            return 2.0 if pokemon_diff > 0 else -2.0
        elif abs(pokemon_diff) >= 3:
            return 5.0 if pokemon_diff > 0 else -5.0
        
        return 0.0


__all__ = ["PokemonCountReward"]