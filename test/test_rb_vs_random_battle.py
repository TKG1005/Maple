# test/test_rb_vs_random_battle.py

import sys
import os
import asyncio
import csv
from datetime import datetime

# プロジェクトルートをシステムパスに追加
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.agents.rule_based_player import RuleBasedPlayer
from poke_env.player import RandomPlayer
from poke_env.environment.battle import Battle

# 対戦設定
N_BATTLES = 10
BATTLE_FORMAT = "gen9randombattle"

async def main():
    start_time = asyncio.get_event_loop().time()

    # --- プレイヤー1: RuleBasedPlayer ---
    # プレイヤー名は指定せず、デフォルトのものを利用
    player1 = RuleBasedPlayer(
        battle_format=BATTLE_FORMAT,
        log_level=25
    )
    # デフォルト名を取得しておく（ログや集計のため）
    # この名前は "Player Server xyzw" や "Player X" のようになることがあります
    player1_default_name = player1.username

    # --- プレイヤー2: RandomPlayer ---
    # プレイヤー名は指定せず、デフォルトのものを利用
    player2 = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        log_level=25
    )
    player2_default_name = player2.username

    print(f"Starting {N_BATTLES} battles between {player1_default_name} (RuleBased) and {player2_default_name} (Random)...")

    battle_results = []

    for i in range(N_BATTLES):
        print(f"\n--- Starting Battle {i + 1}/{N_BATTLES} ({player1_default_name} vs {player2_default_name}) ---")
        await player1.battle_against(player2, n_battles=1)

        if player1.battles:
            last_battle_id = list(player1.battles.keys())[-1]
            battle_info: Battle = player1.battles[last_battle_id]

            winner_username = "Unknown" # 勝者の実際のusernameを格納
            outcome_label = "Unknown"   # 表示用のラベル (RuleBased, Random, Tie)

            # Battleオブジェクトから実際のプレイヤー名を取得して比較
            p1_battle_actual_username = battle_info.players[0] if battle_info.players else player1_default_name
            p2_battle_actual_username = battle_info.players[1] if battle_info.players else player2_default_name

            if battle_info.won:
                winner_username = p1_battle_actual_username
                outcome_label = "RuleBasedPlayer" # player1がRuleBasedPlayerであるという前提
            elif battle_info.lost:
                winner_username = p2_battle_actual_username
                outcome_label = "RandomPlayer"    # player2がRandomPlayerであるという前提
            elif battle_info.tied:
                winner_username = "Tie"
                outcome_label = "Tie"
            
            turns = battle_info.turn
            battle_results.append({
                "battle_id": last_battle_id,
                "winner_username": winner_username, # 実際の勝者名
                "outcome_label": outcome_label,     # RuleBased/Random/Tie のラベル
                "turns": turns,
                "player1_team_species": [p.species for p in battle_info.team.values()] if battle_info.team else [],
                "player2_team_species": [p.species for p in battle_info.opponent_team.values()] if battle_info.opponent_team else []
            })
            print(f"Battle {i + 1} finished. Winner: {outcome_label} ({winner_username}), Turns: {turns}")
        else:
            print(f"Battle {i + 1} result could not be determined for {player1_default_name}.")
            battle_results.append({
                "battle_id": f"battle_{i+1}_error",
                "winner_username": "Error",
                "outcome_label": "Error",
                "turns": 0,
                "player1_team_species": [],
                "player2_team_species": []
            })

    # --- 集計 ---
    # outcome_label を使って集計
    p1_wins = sum(1 for r in battle_results if r["outcome_label"] == "RuleBasedPlayer")
    p2_wins = sum(1 for r in battle_results if r["outcome_label"] == "RandomPlayer")
    ties = sum(1 for r in battle_results if r["outcome_label"] == "Tie")
    errors = sum(1 for r in battle_results if r["outcome_label"] == "Error")
    
    total_turns = sum(r["turns"] for r in battle_results if r["outcome_label"] != "Error")
    valid_battles = N_BATTLES - errors

    avg_turns = total_turns / valid_battles if valid_battles > 0 else 0
    p1_win_rate = p1_wins / valid_battles if valid_battles > 0 else 0
    p2_win_rate = p2_wins / valid_battles if valid_battles > 0 else 0

    # --- コンソール出力 ---
    # 固定のラベルで表示
    print(f"\n\n--- Overall Battle Statistics ({N_BATTLES} battles: RuleBasedPlayer vs RandomPlayer) ---")
    print(f"RuleBasedPlayer ({player1_default_name}) Wins: {p1_wins} (Win Rate: {p1_win_rate:.2%})")
    print(f"RandomPlayer ({player2_default_name}) Wins: {p2_wins} (Win Rate: {p2_win_rate:.2%})")
    print(f"Ties: {ties}")
    if errors > 0:
        print(f"Errors/Undetermined: {errors}")
    print(f"Average Turns per Battle: {avg_turns:.2f}")
    print(f"Total execution time: {asyncio.get_event_loop().time() - start_time:.2f} seconds.")

    # --- CSV出力 ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"battle_metrics_rb_vs_random_defaultnames_{timestamp}.csv")

    try:
        with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
            if not battle_results:
                writer = csv.writer(file)
                # CSVヘッダーに outcome_label と winner_username を含める
                writer.writerow(["battle_id", "winner_username", "outcome_label", "turns", "player1_team_species", "player2_team_species"])
            else:
                fieldnames = battle_results[0].keys()
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                for result in battle_results:
                    writer.writerow(result)
            
            csv_writer_obj = csv.writer(file)
            if battle_results:
                 csv_writer_obj.writerow([])

            csv_writer_obj.writerow(["Summary", "Value"])
            csv_writer_obj.writerow([f"RuleBasedPlayer ({player1_default_name}) Wins", p1_wins])
            csv_writer_obj.writerow([f"RuleBasedPlayer ({player1_default_name}) Win Rate", f"{p1_win_rate:.2%}"])
            csv_writer_obj.writerow([f"RandomPlayer ({player2_default_name}) Wins", p2_wins])
            csv_writer_obj.writerow([f"RandomPlayer ({player2_default_name}) Win Rate", f"{p2_win_rate:.2%}"])
            csv_writer_obj.writerow(["Ties", ties])
            csv_writer_obj.writerow(["Average Turns", f"{avg_turns:.2f}"])
            csv_writer_obj.writerow(["Total Battles", N_BATTLES])
            csv_writer_obj.writerow(["Valid Battles (no errors)", valid_battles])

        print(f"Battle metrics saved to {csv_filename}")
    except IOError:
        print(f"Error: Could not write to CSV file {csv_filename}")
    except Exception as e:
        print(f"An unexpected error occurred during CSV writing: {e}")

    # (受け入れ基準の確認は同様)
    print("\nAcceptance Criteria Check for Task 4.4:")
    criteria_met = True
    if not (p1_wins + p2_wins + ties + errors == N_BATTLES):
        print(f"- Count consistency: FAILED (Sum of outcomes {p1_wins+p2_wins+ties+errors} does not match N_BATTLES {N_BATTLES})")
        criteria_met = False
    else:
        print("- Count consistency: PASSED")

    if os.path.exists(csv_filename) and os.path.getsize(csv_filename) > 0 :
         print(f"- CSV Output: PASSED (File '{csv_filename}' created and not empty)")
    else:
         print(f"- CSV Output: FAILED (File '{csv_filename}' not created or empty)")
         criteria_met = False
    
    if valid_battles > 0 and p1_win_rate > p2_win_rate :
        print(f"- RuleBasedPlayer Performance: PASSED (Win rate {p1_win_rate:.2%} is higher than RandomPlayer's {p2_win_rate:.2%})")
    elif valid_battles > 0:
        print(f"- RuleBasedPlayer Performance: CHECK (Win rate {p1_win_rate:.2%}. Expected to be higher than RandomPlayer's {p2_win_rate:.2%})")
    else:
        print(f"- RuleBasedPlayer Performance: NOT ENOUGH DATA (Win rate {p1_win_rate:.2%})")

    if criteria_met:
        print("Most acceptance criteria for Task 4.4 seem to be met. Review performance.")
    else:
        print("Some acceptance criteria for Task 4.4 may not be met. Please review logs and output.")


if __name__ == "__main__":
    asyncio.run(main())