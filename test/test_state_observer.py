import asyncio
import numpy as np
import os
import sys
import yaml
import time
import csv
from datetime import datetime

from pprint import pprint # 結果を整形して表示したい場合

from poke_env.player import RandomPlayer, Player
from poke_env.environment.battle import Battle

# プロジェクトルートをシステムパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# StateObserver をプロジェクトの構造に合わせてインポート
# (Maple/src/state/state_observer.py を想定)
try:
    from src.state.state_observer import StateObserver
except ModuleNotFoundError:
    print("エラー: src.state.state_observer が見つかりません。")
    print("プロジェクトルートから実行しているか、PYTHONPATHを確認してください。")
    exit()
except Exception as e:
    print(f"StateObserver のインポート中にエラーが発生しました: {e}")
    exit()


class ObservingRandomPlayer(RandomPlayer):
    def __init__(self, observer_yaml_path: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.observer = StateObserver(observer_yaml_path)
            print(f"[{self.username}] StateObserver initialized successfully from: {observer_yaml_path}")
            # StateObserver の spec から期待される次元数を計算して表示
            if self.observer and self.observer.spec:
                total_dims = 0
                for group_name, features in self.observer.spec.items():
                    for feature_name, meta in features.items():
                        encoder_type = meta.get('encoder', 'identity')
                        if encoder_type == 'onehot':
                            # 'classes' リストの長さが次元数になる
                            # YAMLでは文字列で'[1,0,0]'のように入っている場合があるのでevalで評価
                            default_classes_str = meta.get('default', '[]') # defaultがない場合の処理を追加
                            try:
                                default_classes = eval(default_classes_str) if isinstance(default_classes_str, str) else default_classes_str
                                num_classes = len(meta.get('classes', default_classes if isinstance(default_classes, list) else []))
                                total_dims += num_classes
                            except Exception: # eval失敗など
                                # 暫定的に onehot のクラス数不明時は1次元としておくか、エラーを出すか。
                                # ここでは state_spec.yml の classes が正しく定義されている前提
                                print(f"Warning: Could not determine class length for onehot encoder '{group_name}.{feature_name}'. Assuming 0 dimensions from default for now.")
                                num_classes = len(meta.get('classes', [])) # specにclassesがあればそれを使う
                                total_dims += num_classes

                        else: # identity, linear_scale などは1次元
                            total_dims += 1
                print(f"[{self.username}] Expected state vector dimension based on spec: {total_dims}")

        except FileNotFoundError:
            print(f"[{self.username}] Error: StateObserver config file not found at {observer_yaml_path}. Please check the path.")
            raise
        except Exception as e:
            print(f"[{self.username}] Error initializing StateObserver: {e}")
            raise

    def choose_move(self, battle: Battle):
        current_turn = battle.turn
        print(f"\n--- Turn {current_turn} (Player: {self.username}) ---")

        # StateObserver で状態を観測
        try:
            if hasattr(self, 'observer') and self.observer:
                observed_state = self.observer.observe(battle)
                #print(f"[{self.username}] Observed state (shape: {observed_state.shape}):")
                # np.set_printoptions(threshold=np.inf) # 全要素表示したい場合
                #pprint(observed_state.tolist()) # リストとして整形して表示
                #print(observed_state) # 通常のprint
                #print(f"[{self.username}] State vector dimension from observer output: {len(observed_state)}")

            else:
                print(f"[{self.username}] StateObserver not available.")

        except Exception as e:
            print(f"[{self.username}] Error during state observation: {e}")
            # 観察エラーが発生しても、親クラスの行動選択は試みる

        # 親クラス (RandomPlayer) の行動選択ロジックを呼び出す
        return super().choose_move(battle)

    # gen9randombattle では teampreview は通常呼び出されないが、念のためオーバーライドしておく
    def teampreview(self, battle: Battle):
        print(f"\n--- Teampreview (Player: {self.username}) ---")
        # StateObserver は主に詳細な戦闘中の状態を見るため、
        # チームプレビュー時点ではあまり有用な情報を出せない可能性がある。
        # ここではログ出力のみに留め、親クラスの処理を呼ぶ。
        try:
            if hasattr(self, 'observer') and self.observer:
                print(f"[{self.username}] Attempting to observe teampreview state (might be limited).")
                # observed_state_tp = self.observer.observe(battle) # チームプレビューでの観察を試す場合
                # print(f"[{self.username}] Teampreview observed state: {observed_state_tp}")
            else:
                print(f"[{self.username}] StateObserver not available for teampreview.")
        except Exception as e:
            print(f"[{self.username}] Error during teampreview state observation: {e}")

        return super().teampreview(battle)


async def main():
    # --- 設定 ---
    state_spec_path = "src/state/state_spec.yml"
    battle_format = "gen9randombattle" # ランダムバトル形式
    num_battles = 1

    # --- プレイヤーの準備 ---

    player1 = ObservingRandomPlayer(
        battle_format=battle_format,
        observer_yaml_path=state_spec_path,
        log_level=25 # INFOより少し詳細 (poke-envのログレベル)
                     # 20=DEBUG, 25=やや詳細なINFO, 30=INFO, 40=WARNING
    )

    # 対戦相手は通常のRandomPlayerでも良いし、もう一体のObservingRandomPlayerでも良い
    player2 = RandomPlayer(
        battle_format=battle_format,
        log_level=25
    )
    # もし両方観察したいなら:
    # player2 = ObservingRandomPlayer(
    #     player_configuration=player2_config,
    #     battle_format=battle_format,
    #     observer_yaml_path=state_spec_path,
    #     log_level=25
    # )

    # --- 対戦の実行 ---
    print(f"ランダム対戦を開始します ({player1.username} vs {player2.username})...")
    print(f"プレイヤー1 ({player1.username}) のStateObserverの出力を監視します。")

    for i in range(num_battles):
        print(f"\n===== Battle {i + 1} START =====")
        await player1.battle_against(player2, n_battles=1)
        print(f"===== Battle {i + 1} END =====")

        # 簡単な結果表示
        if player1.battles:
            last_battle_key = list(player1.battles.keys())[-1]
            battle_result: Battle = player1.battles[last_battle_key]
            final_turn = battle_result.turn
            winner = "Unknown"
            if battle_result.won:
                winner = player1.username
            elif battle_result.lost:
                winner = player2.username
            else: # 実際にはlostかwonになるはず
                winner = "Draw"
            print(f"\nBattle {i+1} Result: Winner - {winner}, Turns - {final_turn}")
        else:
            print(f"No battle result recorded for battle {i+1}.")

    print("\n全ての対戦が終了しました。")

if __name__ == "__main__":
    asyncio.run(main())