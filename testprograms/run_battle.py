import asyncio
# random_player.py から RandomPlayer クラスをインポート
# (random_player.py と同じフォルダにある場合)
# from random_player import RandomPlayer
# もし poke-env に含まれるものを使うなら:
from poke_env.player import RandomPlayer
from my_simple_player import MySimplePlayer
from poke_env.environment.battle import Battle


async def main():
    # --- プレイヤーの準備 ---
    # 2つの RandomPlayer インスタンスを作成します。
    # battle_format で対戦ルールを指定します。第9世代のランダム対戦を指定してみましょう。
    player1 = RandomPlayer(
        battle_format="gen9randombattle",
        # ローカルサーバーで動かす場合、通常 server_configuration の指定は不要
        # server_configuration=...,
        log_level= 20 # INFOレベルのログを出力する場合 (デバッグに便利)
    )
    player2 = RandomPlayer(
        battle_format="gen9randombattle",
        log_level = 20
    )

    # --- 対戦の実行 ---
    # player1 が player2 に1回対戦を挑みます。
    print("対戦を開始します...")
    await player1.battle_against(player2, n_battles=1)
    print("対戦が終了しました。")
        
    if player1.battles:
        battle_result = list(player1.battles.values())[-1]
        final_turn = battle_result.turn

        if battle_result.won:
            print(f"勝者: {player1.__class__.__name__} (player1) / 最終ターン: {final_turn}")
        elif battle_result.lost:
            print(f"勝者: {player2.__class__.__name__} (player2) / 最終ターン: {final_turn}")
        elif battle_result.tied:
            print(f"結果: 引き分け / 最終ターン: {final_turn}")
        else:
            print(f"結果: 不明 (勝敗判定不可) / 最終ターン: {final_turn}")
    else:
        print("結果: 対戦結果を取得できませんでした（battlesに何も記録されていません）")

if __name__ == "__main__":
    # 非同期関数 main() を実行します。
    asyncio.run(main())