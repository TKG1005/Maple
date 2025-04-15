import asyncio
# random_player.py から RandomPlayer クラスをインポート
# (random_player.py と同じフォルダにある場合)
# from random_player import RandomPlayer
# もし poke-env に含まれるものを使うなら:
from poke_env.player import RandomPlayer

async def main():
    # --- プレイヤーの準備 ---
    # 2つの RandomPlayer インスタンスを作成します。
    # battle_format で対戦ルールを指定します。第9世代のランダム対戦を指定してみましょう。
    player1 = RandomPlayer(
        battle_format="gen9randombattle",
        # ローカルサーバーで動かす場合、通常 server_configuration の指定は不要
        # server_configuration=...,
        # log_level=20 # INFOレベルのログを出力する場合 (デバッグに便利)
    )
    player2 = RandomPlayer(
        battle_format="gen9randombattle",
    )

    # --- 対戦の実行 ---
    # player1 が player2 に1回対戦を挑みます。
    print("対戦を開始します...")
    await player1.battle_against(player2, n_battles=1)
    print("対戦が終了しました。")

if __name__ == "__main__":
    # 非同期関数 main() を実行します。
    asyncio.get_event_loop().run_until_complete(main())