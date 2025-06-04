# my_simple_player.py の中身

import numpy as np
from poke_env.player import Player
# 行動選択のために Move や Pokemon オブジェクトが必要になることがあるので、
# 基本的な型情報もインポートしておくと便利 (必須ではないが良い習慣)
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
from src.action.action_helper import get_available_actions, action_index_to_order

class MySimplePlayer(Player):
    
    def choose_move(self, battle):
        # 1) 利用可能アクションのマスク＆マッピング取得
        mask, mapping = get_available_actions(battle)

        # 2) マスクベクトルをログ出力
        print(f"Turn:{battle.turn},Available action mask:{mask}")

        # 3) マッピングが空ならランダムチョイス
        if not mapping:
            return self.choose_random_move(battle)

        # 4) ランダムに１つの action_index を選択
        idx = np.random.choice(list(mapping.keys()))
        print("Chosen action index:", idx)

        # 5) poke‐env が理解できるオブジェクトに変換して返却
        return action_index_to_order(self, battle, idx)
    
# --- クラス定義はここまで ---