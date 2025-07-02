from __future__ import annotations

import numpy as np
from typing import Any

from .MapleAgent import MapleAgent


class RandomAgent(MapleAgent):
    """ランダムに有効な行動を選択するシンプルなエージェント。"""

    def select_action(self, observation: Any, action_mask: Any) -> int:
        """有効な行動の中からランダムに選択する。
        
        Parameters
        ----------
        observation : Any
            環境からの観測（未使用）
        action_mask : Any
            有効な行動のマスク（numpy配列）
            
        Returns
        -------
        int
            選択された行動のインデックス
        """
        # action_maskから有効な行動のインデックスを取得
        valid_actions = np.where(action_mask)[0]
        
        if len(valid_actions) == 0:
            # 有効な行動がない場合は0を返す（通常は起こらない）
            return 0
            
        # ランダムに1つ選択
        chosen_action = self.env.rng.choice(valid_actions)
        return int(chosen_action)


__all__ = ["RandomAgent"]