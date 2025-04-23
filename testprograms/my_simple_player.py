# my_simple_player.py の中身

from poke_env.player import Player
# 行動選択のために Move や Pokemon オブジェクトが必要になることがあるので、
# 基本的な型情報もインポートしておくと便利 (必須ではないが良い習慣)
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
import random # ランダム選択を使う場合 (今回の例では使わないが、参考のため)

class MySimplePlayer(Player):
    
    def choose_move(self, battle):
        # 1) 技が出せる
        if battle.available_moves:
            return self.create_order(battle.available_moves[0])

        # 2) 強制交代 or 通常交代
        if battle.force_switch and battle.available_switches:
            return self.create_order(battle.available_switches[0])

        if battle.available_switches:
            return self.create_order(battle.available_switches[0])

        # 3) どうにもならない（降参など）
        return self.choose_random_move(battle)  # choose_default_move() は避ける

# --- クラス定義はここまで ---