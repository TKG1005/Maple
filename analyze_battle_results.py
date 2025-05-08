# analyze_battle_results.py

import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# スタイルを設定してグラフを見やすくする (任意)
plt.style.use('seaborn-v0_8-whitegrid') # Matplotlib 3.6以降で推奨されるスタイル
# plt.style.use('seaborn-whitegrid') # 古いバージョン用

def get_latest_csv_file(directory="results", prefix="battle_metrics_"):
    """指定されたディレクトリから最新のCSVファイルを取得する"""
    list_of_files = glob.glob(os.path.join(directory, f"{prefix}*.csv"))
    if not list_of_files:
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def plot_battle_results(csv_filepath):
    """CSVファイルから対戦結果を読み込み、グラフをプロットする"""
    if not csv_filepath or not os.path.exists(csv_filepath):
        print(f"Error: CSV file not found at {csv_filepath}")
        return

    try:
        # CSVファイルを読み込む。サマリー行はスキップするか、後でフィルタリングする。
        # まず全行読み込み、'battle_id'列が有効な行（対戦データ行）のみを抽出する。
        all_data_df = pd.read_csv(csv_filepath)
        
        # 'battle_id' 列が存在し、かつ 'battle_' で始まる行を対戦データとして抽出
        # または、'winner' 列が 'Error' でない行、かつ 'battle_id' がNaNでない行など
        # 今回のCSV構造では、サマリー行の battle_id は空か特殊な値になることを想定
        # battle_resultsリストから書き出しているので、battle_idが空になることはないはず
        # ただし、サマリー行はDictWriterではなくwriterで追記しているので、列数が異なる可能性がある
        # そのため、まずはエラーなく読み込める行数を特定し、そこから処理する。

        # サマリー情報が追記される前の、純粋な対戦結果の行数を特定
        # 'Summary'という行が現れるまでをデータフレームとして扱うのが安全
        summary_start_index = all_data_df[all_data_df.iloc[:, 0] == 'Summary'].index
        if not summary_start_index.empty:
            df = pd.read_csv(csv_filepath, nrows=summary_start_index[0])
        else: # サマリー行がない古いフォーマットの場合や、サマリーのみのファイル
            df = all_data_df[all_data_df['battle_id'].notna() & all_data_df['battle_id'].str.contains('battle_', na=False)]

        if df.empty:
            print("No valid battle data found in the CSV file after filtering.")
            return

        print(f"Successfully loaded {len(df)} battle records from {csv_filepath}")

    except pd.errors.EmptyDataError:
        print(f"Error: The CSV file {csv_filepath} is empty.")
        return
    except Exception as e:
        print(f"Error reading or parsing CSV file {csv_filepath}: {e}")
        return

    # プレイヤー名を取得 (最初の有効なバトルから取得、あるいは固定)
    # ここでは仮に Player1, Player2 とする。CSVにプレイヤー名があればそれを使うとより良い。
    # タスク4.2のCSVでは winner 列にプレイヤー名が入っている
    player_names = df['winner'].unique().tolist()
    player_names = [name for name in player_names if name not in ['Tie', 'Error', 'Unknown']]
    
    # プレイヤー名が特定できない場合はデフォルトを設定
    player1_name = player_names[0] if len(player_names) > 0 else "Player 1"
    player2_name = player_names[1] if len(player_names) > 1 else "Player 2"
    # もしプレイヤー名が固定なら以下のように設定
    # player1_name = "RuleBasedPlayer_1" # タスク4.2で設定した名前に合わせる
    # player2_name = "RuleBasedPlayer_2"


    # 1. プレイヤーごとの総勝利数の比較 (棒グラフ)
    wins_p1 = df[df['winner'] == player1_name].shape[0]
    wins_p2 = df[df['winner'] == player2_name].shape[0]
    ties = df[df['winner'] == 'Tie'].shape[0]

    plt.figure(figsize=(8, 6))
    categories = [player1_name, player2_name, 'Ties']
    counts = [wins_p1, wins_p2, ties]
    bars = plt.bar(categories, counts, color=['skyblue', 'lightcoral', 'lightgrey'])
    plt.title('Total Wins and Ties')
    plt.xlabel('Player / Outcome')
    plt.ylabel('Number of Battles')
    for bar in bars: # 棒グラフの上に数値を表示
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, int(yval), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join("results", "total_wins_ties_bar.png"))
    plt.show()


    # 2. 勝敗引き分けの割合 (円グラフ)
    labels = []
    sizes = []
    colors = []
    
    if wins_p1 > 0:
        labels.append(f'{player1_name} Wins')
        sizes.append(wins_p1)
        colors.append('skyblue')
    if wins_p2 > 0:
        labels.append(f'{player2_name} Wins')
        sizes.append(wins_p2)
        colors.append('lightcoral')
    if ties > 0:
        labels.append('Ties')
        sizes.append(ties)
        colors.append('lightgrey')

    if not sizes: # 有効な勝敗データがない場合
        print("No win/loss/tie data to plot for the pie chart.")
    else:
        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})
        plt.title('Battle Outcome Proportions')
        plt.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
        plt.tight_layout()
        plt.savefig(os.path.join("results", "battle_outcome_pie.png"))
        plt.show()

    # 3. 各対戦のターン数の推移 (折れ線グラフ)
    # battle_id をインデックスにするか、単純な連番にする
    df['battle_number'] = range(1, len(df) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(df['battle_number'], df['turns'], marker='o', linestyle='-', color='green')
    plt.title('Number of Turns per Battle')
    plt.xlabel('Battle Number')
    plt.ylabel('Number of Turns')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "turns_per_battle_line.png"))
    plt.show()

    # 4. ターン数の分布 (ヒストグラム)
    plt.figure(figsize=(10, 6))
    max_turns = df['turns'].max()
    min_turns = df['turns'].min()
    # ビンの数を調整。対戦数が少ない場合は特に注意。
    num_bins = min(20, max(1, max_turns - min_turns +1 )) if max_turns > min_turns else 10
    
    plt.hist(df['turns'], bins=num_bins, color='purple', edgecolor='black', alpha=0.7)
    plt.title('Distribution of Battle Turns')
    plt.xlabel('Number of Turns')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join("results", "battle_turns_histogram.png"))
    plt.show()

    print(f"Plots saved in '{os.path.join(os.getcwd(), 'results')}' directory.")

if __name__ == "__main__":
    latest_csv = get_latest_csv_file()
    if latest_csv:
        print(f"Analyzing latest CSV file: {latest_csv}")
        plot_battle_results(latest_csv)
    else:
        print("No CSV files found in the 'results' directory to analyze.")