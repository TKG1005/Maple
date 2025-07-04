from __future__ import annotations

import re
from typing import List, Tuple
import numpy as np


def parse_opponent_mix(mix_str: str) -> List[Tuple[str, float]]:
    """
    対戦相手比率文字列をパースして、(相手タイプ, 比率)のリストを返す。
    
    Parameters
    ----------
    mix_str : str
        "random:0.3,max:0.3,self:0.4"形式の文字列
        
    Returns
    -------
    List[Tuple[str, float]]
        [(相手タイプ, 比率), ...]のリスト
        
    Raises
    ------
    ValueError
        不正な形式や無効な相手タイプが含まれている場合
    """
    valid_types = {"random", "max", "rule", "self"}
    
    if not mix_str.strip():
        raise ValueError("Empty opponent mix string")
    
    # カンマで分割
    pairs = mix_str.split(",")
    result = []
    
    for pair in pairs:
        pair = pair.strip()
        if ":" not in pair:
            raise ValueError(f"Invalid format in '{pair}'. Expected 'type:ratio'")
        
        type_name, ratio_str = pair.split(":", 1)
        type_name = type_name.strip()
        ratio_str = ratio_str.strip()
        
        # タイプ名の検証
        if type_name not in valid_types:
            raise ValueError(f"Invalid opponent type '{type_name}'. Valid types: {valid_types}")
        
        # 比率の変換
        try:
            ratio = float(ratio_str)
        except ValueError:
            raise ValueError(f"Invalid ratio '{ratio_str}' for type '{type_name}'")
        
        if ratio < 0:
            raise ValueError(f"Negative ratio {ratio} for type '{type_name}'")
        
        result.append((type_name, ratio))
    
    # 比率の正規化
    total_ratio = sum(ratio for _, ratio in result)
    if total_ratio == 0:
        raise ValueError("Total ratio is zero")
    
    normalized_result = [(type_name, ratio / total_ratio) for type_name, ratio in result]
    
    return normalized_result


class OpponentPool:
    """複数の対戦相手タイプから重み付きランダムで選択するクラス。"""
    
    def __init__(self, opponent_mix: List[Tuple[str, float]], rng: np.random.Generator = None):
        """
        Parameters
        ----------
        opponent_mix : List[Tuple[str, float]]
            [(相手タイプ, 比率), ...]のリスト
        rng : np.random.Generator, optional
            乱数生成器。Noneの場合はデフォルトを使用
        """
        self.opponent_types = [pair[0] for pair in opponent_mix]
        self.weights = [pair[1] for pair in opponent_mix]
        self.rng = rng or np.random.default_rng()
        
        # 重みの正規化
        total_weight = sum(self.weights)
        if total_weight == 0:
            raise ValueError("Total weight is zero")
        self.weights = [w / total_weight for w in self.weights]
    
    def sample_opponent_type(self) -> str:
        """重み付きランダムで対戦相手タイプを選択する。"""
        return str(self.rng.choice(self.opponent_types, p=self.weights))