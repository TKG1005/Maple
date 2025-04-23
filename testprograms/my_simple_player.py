# my_simple_player.py の中身

from poke_env.player import Player
# 行動選択のために Move や Pokemon オブジェクトが必要になることがあるので、
# 基本的な型情報もインポートしておくと便利 (必須ではないが良い習慣)
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon
import random # ランダム選択を使う場合 (今回の例では使わないが、参考のため)

class MySimplePlayer(Player):
    """
    これは私たちが作る最初のカスタムプレイヤーです。
    choose_move メソッドを実装しました。
    """
    def choose_move(self, battle):
        # battle オブジェクトは現在の対戦状況を表します
        
        #　対戦が継続しているかどうかの確認
        if battle.finished:
            print("対戦終了済みです。行動を選択しません。")
            return None

        # 1. 使える技があるか確認
        if battle.available_moves:
            # 使える技があれば、リストの最初の技を選択
            best_move = battle.available_moves[0]
            print(f"{self.username} が選択した行動: 技 '{best_move.id}'") # ログ出力 (任意)
            return self.create_order(best_move) # 選択した技で行動順序を作成して返す

        # 2. 使える技がない場合、交代できるか確認
        elif battle.available_switches:
            # 交代できるポケモンがいれば、リストの最初のポケモンを選択
            best_switch = battle.available_switches[0]
            print(f"{self.username} が選択した行動: 交代 '{best_switch.species}'") # ログ出力 (任意)
            return self.create_order(best_switch) # 選択した交代で行動順序を作成して返す

        # 3. 技も使えず、交代もできない場合 (通常は発生しないはずだが念のため)
        #    仕方ないのでデフォルトの行動 (通常は技リストの先頭、PPがあれば) を試みる
        #    あるいは、強制的に降参するなどの処理も考えられる
        else:
            print(f"{self.username}: 取れる行動がありません！デフォルト行動を試みます。") # ログ出力 (任意)
            return self.choose_default_move() # デフォルト行動を返す

# --- クラス定義はここまで ---