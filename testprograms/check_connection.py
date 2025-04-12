import asyncio
import os # osモジュールをインポート
from poke_env.player import Player, RandomPlayer
# from poke_env.server_configuration import ServerConfiguration # 元の行
from poke_env.server_configuration import LocalhostServerConfiguration # 変更

# ローカルサーバーへの接続設定
# LocalhostServerConfiguration を使用
# デフォルトで "localhost:8000" に接続し、認証情報は None になります
LocalServer = LocalhostServerConfiguration

class SimplePlayer(Player):
    def choose_move(self, battle):
        if battle.available_moves:
            move = self.choose_random_move(battle)
            print(f"{self.username} chose move: {move}")
            return move
        elif battle.available_switches:
            switch = self.choose_random_switch(battle)
            print(f"{self.username} chose switch: {switch}")
            return switch
        else:
            print(f"{self.username} has to pass.")
            return self.choose_default_move(battle)

async def main():
    print("ローカルサーバーへの接続を開始します...")

    # 接続テスト用のプレイヤーを作成
    # サーバー設定を LocalhostServerConfiguration に変更
    test_player = SimplePlayer(
        battle_format="gen9randombattle",
        # server_configuration=LocalServer, # ここはクラスそのものを渡すか、インスタンスを渡すか要確認。通常インスタンス
        server_configuration=LocalServer(), # インスタンス化して渡すのが一般的
        log_level=20
    )

    # 相手プレイヤーも同様に
    opponent = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalServer(), # インスタンス化して渡す
    )

    try:
        await test_player.challenge(opponent.username, num_battles=1)
        print("接続に成功し、対戦を開始しました！")
        print("Pokemon Showdownサーバーのターミナルと、ブラウザ(localhost:8000)で対戦状況を確認してみてください。")
        print("スクリプトは対戦終了後に自動で終了します。")

    except Exception as e:
        print(f"接続または対戦開始中にエラーが発生しました: {e}")
        print("ローカルサーバーが正しく起動しているか確認してください。")

    finally:
        await test_player.disconnect()
        await opponent.disconnect()
        print("プレイヤーを切断しました。")


if __name__ == "__main__":
    # Windows環境向けの非同期イベントループポリシー設定
    if os.name == 'nt':
         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.get_event_loop().run_until_complete(main())