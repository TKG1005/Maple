import asyncio  # 非同期I/O操作 (async/await) のためのライブラリ
import os       # オペレーティングシステム機能 (Windowsのイベントループポリシー設定で使用)
from poke_env.player import Player, RandomPlayer  # poke_envライブラリからプレイヤー関連クラスをインポート
from poke_env import ServerConfiguration         # poke_envライブラリからサーバー設定クラスをインポート
import time     # 時間関連の機能 (待機処理で使用)
import traceback # エラー発生時に詳細なトレースバック情報を表示するためにインポート

# --- サーバー設定 ---
# ローカル環境で起動しているPokemon Showdownサーバーへの接続情報を定義します。
# ServerConfigurationクラスのインスタンスを作成します。
LocalServer = ServerConfiguration(
    # サーバーのアドレスとポート番号をWebSocket形式 (ws://) で指定します。
    # デフォルトは localhost の 8000番ポートです。
    "ws://localhost:8000/showdown/websocket",
    # ローカルサーバーで認証が不要な場合は None を指定します。
    None
)

# --- 自作プレイヤークラス (SimplePlayer) ---
# poke_env の基本 Player クラスを継承して、独自のプレイヤーを作成します。
# ここでは、技や交代をランダムに選択するだけのシンプルなプレイヤーを定義します。
class SimplePlayer(Player):
    # サーバーからのメッセージを内部的に処理するメソッド (オーバーライド可能)
    async def _handle_message(self, message: str):
        # デバッグ用に受信メッセージの一部を表示したい場合は、以下のコメントを解除します。
        # print(f"[{self.username or 'Connecting...'}] Recv: {message[:100]}")
        # 必ず親クラスの _handle_message を呼び出して、基本的な処理を行わせます。
        await super()._handle_message(message)

    # 行動選択メソッド (Playerクラスを継承する場合、このメソッドの実装が必須です)
    # 対戦中にサーバーから行動選択を求められたときに呼び出されます。
    # battle オブジェクトには、現在の対戦状況に関する情報が含まれています。
    def choose_move(self, battle):
        if battle.available_moves:
            # moveまたはbattleorderを取得
            action = self.choose_random_move(battle)
            
            # actionが技の場合 (Moveオブジェクト)
            if hasattr(action, 'id'):
                print(f"[{self.username}] 行動選択: 技 '{action.id}' を選択しました。")
            # actionがBattleOrderの場合 (例: テラスタル化込みの行動)
            else:
                print(f"[{self.username}] 行動選択: 複合アクション '{action}' を選択しました。")

            return action
        
        elif battle.available_switches:
            switch = self.choose_random_switch(battle)
            print(f"[{self.username}] 行動選択: 交代 '{switch.species}' を選択しました。")
            return switch
        
        else:
            print(f"[{self.username}] 行動選択: 有効な行動がないため、パスします。")
            return self.choose_default_move(battle)


# --- チャレンジを受ける側のプレイヤー (Opponent) の処理 ---
# この非同期関数は、相手プレイヤー (ここではRandomPlayer) の動作を定義します。
# 主な役割は、サーバーに接続し、他のプレイヤーからの対戦申し込み (チャレンジ) を待つことです。
async def opponent_routine(player: Player):
    player_name = player.__class__.__name__ # ログ表示用にクラス名を取得
    print(f"Opponent ({player_name}) がチャレンジ待機を開始します...")
    try:
        # accept_challenges メソッドを呼び出します。
        # これにより、サーバーへの接続が試行され、接続後はチャレンジを待ち受けます。
        # 第一引数: チャレンジを受け付ける相手のユーザー名。Noneの場合は誰からでも受け付けます。
        # 第二引数: 受け付けるチャレンジの最大回数。ここでは1回だけ受け付けたら終了します。
        await player.accept_challenges(None, 1)
        # チャレンジを受け付けたか、何らかの理由で待機が終了すると、この行に到達します。
        # player.username は接続後にサーバーから割り当てられます。
        print(f"Opponent ({player.username or player_name}) はチャレンジを受け付けました（または待機が終了しました）。")
    except asyncio.CancelledError:
        # このタスクが他の場所からキャンセルされた場合に発生します。
        print(f"Opponent ({player_name}) の待機処理がキャンセルされました。")
    except Exception as e:
        # accept_challenges 中に予期せぬエラーが発生した場合。
        print(f"Opponent ({player_name}) 処理中にエラーが発生しました: {e}")
        traceback.print_exc() # エラーの詳細情報をコンソールに出力します。
    finally:
        # tryブロックの処理が正常終了したか、エラーが発生したか、キャンセルされたかに関わらず、
        # 最後に必ず実行される後片付け処理です。
        print(f"Opponent ({player_name}) のルーチンを終了します。")
        # プレイヤーがまだサーバーに接続されている場合は、切断処理を試みます。
        # hasattrでメソッドの存在確認、player.connectedで接続状態を確認してから呼び出すのが安全です。
        if hasattr(player, 'disconnect') and player.connected:
             print(f"Opponent ({player_name}) をサーバーから切断します。")
             await player.disconnect()

# --- メインの非同期処理 ---
# このスクリプトの中心となる処理フローを定義します。
async def main():
    print("Maple Project - ローカルサーバー接続確認スクリプト")
    print("-" * 30) # 区切り線
    print("ローカルサーバーへの接続と、プレイヤー間のチャレンジ送信を確認します。")

    # --- プレイヤーオブジェクトの作成 ---
    # これから対戦（今回はチャレンジ送信まで）を行う2体のエージェントを作成します。
    print("プレイヤーオブジェクトを作成中...")
    # test_player: 上で定義した SimplePlayer クラスを使用。こちらがチャレンジを送る側。
    test_player = SimplePlayer(
        battle_format="gen9randombattle",  # 対戦形式 (今回はランダムバトル)
        server_configuration=LocalServer,  # 上で定義したローカルサーバー設定を使用
        log_level=20  # ログの詳細度 (INFOレベル)
    )
    # opponent: poke_env 標準の RandomPlayer を使用。チャレンジを受ける側。
    opponent = RandomPlayer(
        battle_format="gen9randombattle",
        server_configuration=LocalServer,
        log_level=20
    )

    # --- Opponent (受け手) の処理をバックグラウンドで開始 ---
    print("チャレンジ待機側のプレイヤー (Opponent) の処理をバックグラウンドで開始します...")
    # opponent_routine 関数を非同期タスクとして起動します。
    # これにより、opponent はチャレンジ待機状態に入りつつ、main 関数の処理は先に進みます。
    opponent_task = asyncio.create_task(opponent_routine(opponent))

    # --- test_player (送り手) の接続確認 ---
    print("チャレンジ送信側のプレイヤー (test_player) のサーバー接続を確認します...")
    # test_player はチャレンジを送る側なので、accept_challenges は呼びません。
    # Playerオブジェクトは、内部的にサーバーへの接続を試みます。
    # 接続が完了し、サーバーからユーザー名が割り当てられるまで待機します。
    start_time = time.time()  # 待機開始時刻を記録
    timeout_seconds = 20      # タイムアウトまでの秒数
    test_player_connected = False # 接続成功フラグ

    # タイムアウトするか接続が確認できるまでループします。
    while time.time() - start_time < timeout_seconds:
        # player.username 属性に値が入ったら、サーバーから名前が割り当てられたと判断します。
        # (logged_in 属性がないため、この方法で接続完了を判断します)
        if test_player.username:
            print(f"test_player がサーバーに接続し、ユーザー名 '{test_player.username}' を取得しました。")
            test_player_connected = True
            break # 接続できたのでループを抜けます。
        # 接続が完了するまで、0.5秒待機します。CPU負荷を下げるため。
        await asyncio.sleep(0.5)
    else:
        # whileループが break せずに終了した場合 (タイムアウトした場合)
        print(f"エラー: test_player の接続がタイムアウトしました ({timeout_seconds}秒)。")
        print("以下の点を確認してください:")
        print("  - ローカルの Pokemon Showdown サーバーが起動していますか？")
        print("  - サーバーアドレス/ポート番号 ('ws://localhost:8000/showdown/websocket') は正しいですか？")
        print("  - ファイアウォール等がローカル接続を妨げていませんか？")
        # タイムアウトしたので、バックグラウンドで動いている Opponent タスクもキャンセルします。
        print("Opponent タスクをキャンセルします...")
        opponent_task.cancel()
        await asyncio.sleep(1) # キャンセル処理が完了するのを少し待ちます。
        return # main 関数の処理をここで終了します。

    # --- Opponent のユーザー名取得待機 ---
    # Opponent側もサーバーに接続してユーザー名が設定されるまで少し待ちます。
    # チャレンジを送るためには相手の正確なユーザー名が必要です。
    print("Opponent のユーザー名が設定されるのを待ちます...")
    target_opponent = None # 送信先のユーザー名を格納する変数
    start_time_opp = time.time()
    timeout_opp_seconds = 10 # Opponentのユーザー名取得は少し短めに待つ
    while time.time() - start_time_opp < timeout_opp_seconds:
        if opponent.username: # opponent のユーザー名が設定されたか？
            target_opponent = opponent.username
            print(f"Opponent のユーザー名 '{target_opponent}' を取得しました。")
            break
        await asyncio.sleep(0.2)
    else:
        # タイムアウトした場合
        print(f"警告: Opponent のユーザー名を時間内 ({timeout_opp_seconds}秒) に取得できませんでした。")
        # 取得できなくても、続行を試みる (相手がゲスト名などの場合、取得が難しいことがある)

    # --- チャレンジ送信 ---
    # test_player が接続できていれば、チャレンジ送信を試みます。
    if test_player_connected:
        # 相手のユーザー名が取得できなかった場合の代替処理
        if not target_opponent:
            print("警告: Opponent のユーザー名が不明です。代わりに自分自身にチャレンジを送ります。")
            target_opponent = test_player.username # 自分自身にチャレンジを送る

        try:
            print(f"test_player ('{test_player.username}') から Opponent ('{target_opponent}') へチャレンジを送信します...")
            # send_challenges メソッドで対戦申し込みを送信します。
            await test_player.send_challenges(target_opponent, n_challenges=1)
            print("チャレンジを送信しました。")
            print("--------------------------------------------------")
            print(">>> 接続確認成功！ <<<")
            print("poke-env からローカルサーバーへの接続と基本的な通信が確認できました。")
            print("--------------------------------------------------")
            print("サーバーログやブラウザ (localhost:8000) でチャレンジ通知を確認してください。")
            # 確認のために少し待機します。
            await asyncio.sleep(5)

        except Exception as e:
            # send_challenges 中にエラーが発生した場合
            print(f"チャレンジ送信中にエラーが発生しました: {e}")
            print("--- エラー詳細 ---")
            traceback.print_exc()
            print("------------------")
        finally:
             # チャレンジ送信処理が終わったら、バックグラウンドのOpponentタスクを終了させます。
             if not opponent_task.done(): # タスクがまだ完了していなければ
                  print("Opponentタスクをキャンセルして終了させます...")
                  opponent_task.cancel()
                  await asyncio.sleep(1) # キャンセル処理を待ちます。

    # --- 終了処理 ---
    print("メインの処理を終了します...")
    # test_player がまだ接続されている場合は切断します。
    if hasattr(test_player, 'disconnect') and test_player.connected:
        print(f"test_player ('{test_player.username}') を切断します。")
        await test_player.disconnect()

    # Opponentタスクが終了するのを待ちます (正常終了 or キャンセル)。
    print("バックグラウンドの Opponentタスク の終了を待機しています...")
    try:
        await opponent_task # タスクの完了を待つ
    except asyncio.CancelledError:
        print("Opponentタスクはキャンセルによって終了しました。")
    print("すべての処理が完了しました。スクリプトを終了します。")


# --- 非同期関数実行のためのラッパー ---
# main() 関数は非同期関数 (async def) なので、直接呼び出すのではなく、
# asyncio のイベントループを通じて実行する必要があります。
async def run_main():
    # ここで main() を呼び出すことで、メインの非同期処理が開始されます。
    await main()

# --- スクリプト実行のエントリーポイント ---
# このファイルが直接実行された場合 (例: `python check_connection.py`) に、
# 以下の `if __name__ == "__main__":` ブロック内のコードが実行されます。
if __name__ == "__main__":
    # Windows 環境で asyncio に関する特定のエラーを防ぐための設定 (推奨)
    if os.name == 'nt': # 'nt' は Windows を示します
         asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # asyncio.run() を使って非同期関数 (run_main) を実行します。
    # これがプログラムの開始点となり、非同期処理全体を管理します。
    try:
        print("非同期処理を開始します...")
        asyncio.run(run_main())
    except RuntimeError as e:
        # asyncio.run() 自体に関するエラーなど、予期せぬ実行時エラーを捕捉します。
        print(f"スクリプトの実行中に予期せぬエラーが発生しました: {e}")
        traceback.print_exc()