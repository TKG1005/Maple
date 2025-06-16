# Maple Project M5 Backlog – 初回強化学習モデル学習と評価

> **目的**
> - `PokemonEnv` を利用し、最初の強化学習モデルを学習させる。
> - 学習済みモデルの対戦結果を評価し、基本性能を把握する。
> - `PokemonEnv_Specification.md` で定義されたインタフェースに従い、実験用スクリプトを整備する。

| # | ステップ名 | 目標 (WHAT) | 達成条件 (DONE) | テスト内容 (HOW) | 使用技術・ライブラリ (WITH) |
|---|-----------|-------------|----------------|-----------------|----------------------------|
| 1 | RLライブラリ選定と依存追加 | 学習に使用する RL ライブラリを決定し `requirements.txt` を更新 | `stable-baselines3` 等が追記され `pip install -r requirements.txt` が成功 | 依存インストール後、`python -c 'import stable_baselines3'` がエラーなく終了 | pip, stable-baselines3 |
| 2 | 学習スクリプト雛形作成 | `train_rl.py` を新規作成し `PokemonEnv` を単一エージェント用ラッパで起動 | `python train_rl.py --dry-run` が環境初期化のみ行い終了 | Gymnasium, wrappers.SingleAgentCompatibilityWrapper |
| 3 | ポリシーネットワーク定義 | PyTorch を用いた簡易 NN モデルを実装 | `model = PolicyNetwork(observation_space, action_space)` がインスタンス化可能 | PyTorch |
| 4 | エージェントクラス実装 | モデルとオプティマイザを保持する `RLAgent` を作成 | `RLAgent.select_action()` がマスクに従い確率分布を返す | PyTorch, numpy |
| 5 | 経験リプレイ導入 | `ReplayBuffer` を準備し遷移を保存 | `len(buffer)` が上限サイズを超えない | numpy |
| 6 | 学習ループ実装 | 環境から遷移を取得し `ReplayBuffer` からバッチ学習 | `train_rl.py --episodes 1` が1エピソード学習を完了 | asyncio, gymnasium |
| 7 | モデル保存処理 | 学習後に `.pt` ファイルとして重みを保存 | `--save model.pt` 指定でファイルが生成される | PyTorch `torch.save` |
| 8 | 評価スクリプト作成 | `evaluate_rl.py` を新規作成し保存済みモデルで対戦 | `python evaluate_rl.py --model model.pt --n 5` が5戦完走 | PokemonEnv, MapleAgent |
| 9 | 勝率・平均報酬計算 | 評価結果から統計値を算出し表示 | 実行ログに `win_rate:` と `avg_reward:` が出力される | Python logging |
|10| 学習パラメータ設定ファイル | ハイパーパラメータを YAML で管理 | `config/train_config.yml` 読み込み後、値が辞書として取得できる | PyYAML |
|11| TensorBoard ログ出力 | 学習過程を可視化する | `--tensorboard` オプション指定で `runs/` ディレクトリにログ生成 | tensorboardX または SB3 built-in |
|12| 途中経過チェックポイント | 一定エピソードごとにモデルを保存 | `--checkpoint-interval 10` で10エピソードごとにファイルが増える | PyTorch |
|13| ランダムポリシーとの比較 | `MapleAgent`(ランダム行動) と対戦させ指標を記録 | 勝率が50%前後になることを確認 | run_battle.py, numpy |
|14| 学習済みモデルの読み込みテスト | 保存済みファイルから `RLAgent` を復元 | `evaluate_rl.py --model saved.pt` がエラーなく実行 | PyTorch |
|15| コマンドライン引数整備 | 主要オプション (`--episodes`, `--lr` 等) を追加 | `python train_rl.py --help` がオプション一覧を表示 | argparse |
|16| 学習スクリプト自動テスト | エピソード数を絞った CI 用テストを追加 | `pytest -k test_train_one_episode` が PASS | pytest |
|17| 評価スクリプト自動テスト | 1 戦だけ実行して終了するテストを作成 | `pytest -k test_evaluate_once` が PASS | pytest |
|18| ドキュメント更新 | M5 開始方法と学習手順を `docs/M5_setup.md` に記述 | ドキュメントを参照し手順通り実行できる | Markdown |
|19| 実験ログ管理 | `logs/` ディレクトリを作り実験毎にファイル保存 | ログファイルに日時とパラメータが記録される | logging |
|20| 乱数シード制御 | 学習・評価とも `--seed` オプションで再現性確保 | 同じシード指定で結果がほぼ一致 | numpy RNG |
|21| 学習結果の図示 | 勝率推移などを matplotlib でプロット | `python plot_results.py logs/run1.json` が図を保存 | matplotlib |
|22| バトルリプレイ保存 | Showdown のリプレイログを保管 | 評価時に `replays/` フォルダへ HTML が出力される | poke-env `save_replay` |
|23| 複数モデル比較機能 | 複数の重みファイルをロードし対戦させる | `evaluate_rl.py --models a.pt b.pt` が両者の勝率を表示 | Python CLI |
|24| 学習時間計測 | 1 エピソード当たりの処理時間を計測 | ログに `time/episode:` が表示される | time, logging |
|25| M5 完了レビュー | すべてのステップを実施し学習モデルの性能を報告 | 指定テストが PASS し、勝率レポートを提出 | 総合確認 |

> **備考**
> - PokemonEnv_Specification.md の状態・行動取得 API を遵守して学習を行う。
> - Showdown サーバは `npx pokemon-showdown` でローカル起動し、`config/my_team.txt` のパーティを使用する。
> - 学習・評価スクリプトは長時間実行を避けるためエピソード数を小さく調整してテスト可能にする。
