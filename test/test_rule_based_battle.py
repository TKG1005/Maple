# test/test_rule_based_battle.py

import sys
import os
import asyncio
import csv
from datetime import datetime

# プロジェクトルートをシステムパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.rule_based_player import RuleBasedPlayer # RuleBasedPlayerをインポート
from poke_env.player import Player # Playerクラスをインポート（チーム設定のため）

# 対戦設定
N_BATTLES = 1000  # 対戦回数
BATTLE_FORMAT = "gen9randombattle"  # 対戦フォーマット (パーティ手動登録の場合は変更が必要)

# チーム定義 (手動で登録する場合)
# 必要に応じて、それぞれのプレイヤーに異なるチームを設定できます
# 例: TEAM_1_PATH = "config/team1.txt"
#     TEAM_2_PATH = "config/team2.txt"
# gen9randombattle を使用する場合は、チーム指定は不要です。
TEAM_PATH = "config/my_team.txt" # M2の時点では gen9randombattle のため、この行は実際には使われません

def load_team(team_path: str) -> str:
    """チームファイルを読み込む関数"""
    try:
        with open(team_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"Error: Team file not found at {team_path}")
        # gen9randombattle以外で手動チームを使う場合は、ここでプログラムを終了させるか、
        # デフォルトチームを返すなどの処理が必要です。
        # 今回はひとまずNoneを返し、Player初期化時にエラーとなるようにします。
        return None

async def main():
    start_time = asyncio.get_event_loop().time()

    # プレイヤーの初期化
    # パーティを手動で設定する場合 (BATTLE_FORMAT が "gen9ou" など特定ルールの場合)
    # team_str = load_team(TEAM_PATH)
    # if not team_str and BATTLE_FORMAT != "gen9randombattle":
    #     print(f"Team for {BATTLE_FORMAT} is required. Exiting.")
    #     return
    #
    # player1 = RuleBasedPlayer(
    #     battle_format=BATTLE_FORMAT,
    #     team=team_str, # gen9randombattle の場合は team=None でOK
    #     log_level=25
    # )
    # player2 = RuleBasedPlayer(
    #     battle_format=BATTLE_FORMAT,
    #     team=team_str, # gen9randombattle の場合は team=None でOK
    #     log_level=25
    # )

    # gen9randombattle を使用する場合 (チーム指定不要)
    player1 = RuleBasedPlayer(
        battle_format=BATTLE_FORMAT,
        log_level=25 # ログレベルは必要に応じて調整
    )
    player2 = RuleBasedPlayer(
        battle_format=BATTLE_FORMAT,
        log_level=25 # ログレベルは必要に応じて調整
    )

    print(f"Starting {N_BATTLES} battles between two RuleBasedPlayers...")

    # 対戦実行
    battle_results = [] # 各対戦の結果を格納するリスト

    for i in range(N_BATTLES):
        print(f"\n--- Starting Battle {i + 1}/{N_BATTLES} ---")
        # 新しいバトルごとにプレイヤーを初期化するか、状態をリセットする必要があるか確認
        # poke-envの battle_against は内部で新しい Battle オブジェクトを都度生成するので、
        # Playerインスタンスを使いまわしても基本的には問題ありません。
        # しかし、プレイヤーが内部状態（例：相手の過去の行動の記憶など）を持つ場合、
        # それをリセットする処理が必要になることがあります。
        # 今回のRuleBasedPlayerはバトルごとに状態がリセットされるため、そのままでOKです。

        # 1戦だけ実行
        await player1.battle_against(player2, n_battles=1)

        # 直近の対戦結果を取得
        # player1.battles は辞書で、キーがバトルID、値がBattleオブジェクト
        # battle_against を1戦ずつ実行しているので、最後のバトルが最新
        if player1.battles:
            # battles辞書の最後の要素を取得
            last_battle_id = list(player1.battles.keys())[-1]
            battle_info: Battle = player1.battles[last_battle_id]

            winner = "Unknown"
            if battle_info.won:
                winner = player1.username
            elif battle_info.lost: # player1が負けた場合、player2が勝ったことになる
                winner = player2.username
            elif battle_info.tied:
                winner = "Tie"
            
            turns = battle_info.turn
            battle_results.append({
                "battle_id": last_battle_id,
                "winner": winner,
                "turns": turns,
                "player1_team": [p.species for p in battle_info.team.values()] if battle_info.team else [], # チーム情報も記録(任意)
                "player2_team": [p.species for p in battle_info.opponent_team.values()] if battle_info.opponent_team else [] # チーム情報も記録(任意)
            })
            print(f"Battle {i + 1} finished. Winner: {winner}, Turns: {turns}")
        else:
            print(f"Battle {i + 1} result could not be determined for player1.")
            battle_results.append({
                "battle_id": f"battle_{i+1}_error",
                "winner": "Error",
                "turns": 0,
                "player1_team": [],
                "player2_team": []
            })
        
        # player1とplayer2のbattleオブジェクトは同一のものを指すはずなので、
        # player2側のバトル結果を参照する必要は通常ありません。

    print(f"\n--- Battle Results ---")
    print(f"Player 1 ({player1.username}) won {player1.n_won_battles} / {N_BATTLES} battles.")
    # RuleBasedPlayer同士なので、player2の勝利数は N_BATTLES - player1.n_won_battles - (引き分け数) となります。
    # poke-env は各プレイヤーのn_won_battlesを記録しますが、直接的な敗北数や引き分け数を
    # 集計するプロパティは持っていません。詳細な結果はbattleオブジェクトから取得する必要があります。
    # ここでは player1 の勝利数のみ表示します。タスク4.2で詳細な集計を行います。

    # --- 集計 ---
    p1_wins = sum(1 for r in battle_results if r["winner"] == player1.username)
    p2_wins = sum(1 for r in battle_results if r["winner"] == player2.username)
    ties = sum(1 for r in battle_results if r["winner"] == "Tie")
    errors = sum(1 for r in battle_results if r["winner"] == "Error")
    total_turns = sum(r["turns"] for r in battle_results if r["winner"] != "Error")
    valid_battles = N_BATTLES - errors

    avg_turns = total_turns / valid_battles if valid_battles > 0 else 0
    p1_win_rate = p1_wins / valid_battles if valid_battles > 0 else 0
    p2_win_rate = p2_wins / valid_battles if valid_battles > 0 else 0

    # --- コンソール出力 ---
    print(f"\n\n--- Overall Battle Statistics ({N_BATTLES} battles) ---")
    print(f"Player 1 ({player1.username}) Wins: {p1_wins} (Win Rate: {p1_win_rate:.2%})")
    print(f"Player 2 ({player2.username}) Wins: {p2_wins} (Win Rate: {p2_win_rate:.2%})")
    print(f"Ties: {ties}")
    if errors > 0:
        print(f"Errors/Undetermined: {errors}")
    print(f"Average Turns per Battle: {avg_turns:.2f}")
    print(f"Total execution time: {asyncio.get_event_loop().time() - start_time:.2f} seconds.")

    # --- CSV出力 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # resultsディレクトリがなければ作成
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"battle_metrics_{timestamp}.csv")

    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            if not battle_results: # 結果が空の場合はヘッダーだけ書き込むか、何もしない
                print("No battle results to write to CSV.")
                writer = csv.writer(file)
                writer.writerow(["battle_id", "winner", "turns", "player1_team", "player2_team"]) # ヘッダー行
                # サマリー情報も書き込む場合はここに追加
                writer.writerow([]) # 空行
                writer.writerow(["Summary", "Value"])
                writer.writerow([f"Player 1 ({player1.username}) Wins", p1_wins])
                writer.writerow([f"Player 1 ({player1.username}) Win Rate", f"{p1_win_rate:.2%}"])
                writer.writerow([f"Player 2 ({player2.username}) Wins", p2_wins])
                writer.writerow([f"Player 2 ({player2.username}) Win Rate", f"{p2_win_rate:.2%}"])
                writer.writerow(["Ties", ties])
                writer.writerow(["Average Turns", f"{avg_turns:.2f}"])
                writer.writerow(["Total Battles", N_BATTLES])
                writer.writerow(["Valid Battles (no errors)", valid_battles])

            else:
                fieldnames = battle_results[0].keys() # 最初の結果からヘッダーを取得
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for result in battle_results:
                    writer.writerow(result)
                
                # サマリー情報をCSVの末尾に追加 (オプション)
                writer.writerow({}) # 空行
                # DictWriterではなくwriterで追記
                csv_writer_obj = csv.writer(file)
                csv_writer_obj.writerow(["Summary", "Value"])
                csv_writer_obj.writerow([f"Player 1 ({player1.username}) Wins", p1_wins])
                csv_writer_obj.writerow([f"Player 1 ({player1.username}) Win Rate", f"{p1_win_rate:.2%}"])
                csv_writer_obj.writerow([f"Player 2 ({player2.username}) Wins", p2_wins])
                csv_writer_obj.writerow([f"Player 2 ({player2.username}) Win Rate", f"{p2_win_rate:.2%}"])
                csv_writer_obj.writerow(["Ties", ties])
                csv_writer_obj.writerow(["Average Turns", f"{avg_turns:.2f}"])
                csv_writer_obj.writerow(["Total Battles", N_BATTLES])
                csv_writer_obj.writerow(["Valid Battles (no errors)", valid_battles])

        print(f"Battle metrics saved to {csv_filename}")
    except IOError:
        print(f"Error: Could not write to CSV file {csv_filename}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")


    #受け入れ基準の確認
    print("\nAcceptance Criteria Check for Task 4.2:")
    criteria_met = True
    if not (p1_wins + p2_wins + ties + errors == N_BATTLES):
        print("- Count consistency: FAILED (Sum of outcomes does not match N_BATTLES)")
        criteria_met = False
    else:
        print("- Count consistency: PASSED")

    if os.path.exists(csv_filename) and os.path.getsize(csv_filename) > 0 :
         print(f"- CSV Output: PASSED (File '{csv_filename}' created and not empty)")
    else:
         print(f"- CSV Output: FAILED (File '{csv_filename}' not created or empty)")
         criteria_met = False
    
    # コンソール出力の確認は目視
    print("- Console Output: Check console for summary (e.g., '○試合中 ○勝○敗、平均ターン数 ○ターン')")

    if criteria_met:
        print("All acceptance criteria for Task 4.2 seem to be met based on script execution.")
    else:
        print("Some acceptance criteria for Task 4.2 may not be met. Please review logs and output.")



if __name__ == "__main__":
    asyncio.run(main())