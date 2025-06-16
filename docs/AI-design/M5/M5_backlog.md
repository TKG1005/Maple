# Maple Project M5 Backlog – 初回強化学習モデル学習と評価

> **目的**
> - `PokemonEnv` と `SingleAgentCompatibilityWrapper` を利用し、基本的な強化学習ループを構築する。
> - Self-Play 形式で初期モデルを学習し、勝率などの指標を取得できるようにする。

| # | ステップ名 | 目標 (WHAT) | 達成条件 (DONE) | テスト内容 (HOW) | 使用技術・ライブラリ (WITH) |
|---|-----------|-------------|----------------|-----------------|------------------------------|
| 1 | 強化学習ライブラリ追加 | `stable-baselines3` と `torch` を依存に加える | `requirements.txt` へ追記し `pip install -r requirements.txt` が成功 | 新規仮想環境でインストール確認 | pip, stable-baselines3, PyTorch |
| 2 | 学習用スクリプト雛形 | `train_rl.py` を作成し `PokemonEnv` を `SingleAgentCompatibilityWrapper` 経由で利用 | `python train_rl.py --dry-run` がエラーなく終了 | CLI 実行 | Python, Gymnasium |
| 3 | ハイパーパラメータ設定 | `config/train_config.yml` に学習ステップ数やモデル保存先を記述 | YAML を読み込み `train_rl.py` が設定値を表示 | 単体テストでパース確認 | PyYAML |
| 4 | Self-Play 対戦ループ | 2 体の `MapleAgent` が同一ポリシーを共有して対戦 | 1 エピソード学習が最後まで実行される | ログに勝敗が出力される | PokemonEnv, SB3 |
| 5 | ポリシーネットワーク実装 | 小規模 MLP を用いたポリシーを定義 | `train_rl.py` 実行時にモデルのパラメータ数が表示される | SB3 のネットワークサマリ確認 | PyTorch, SB3 |
| 6 | モデル保存と再読み込み | 学習後 `models/latest.zip` が生成され、再読み込み可能 | `--load models/latest.zip` で学習再開 | ファイル存在確認 | SB3 save/load |
| 7 | 学習ログ出力 | 収束状況を `logs/` 配下に TensorBoard 形式で保存 | `tensorboard --logdir logs` でグラフ確認 | ログディレクトリ作成確認 | TensorBoard |
| 8 | 簡易評価スクリプト | `evaluate_model.py` で保存済みモデルの勝率を測定 | 5 戦評価で平均報酬が表示される | スクリプト実行 | PokemonEnv, SB3 |
| 9 | 学習用ユニットテスト | `pytest` で `train_rl.py --dry-run` が実行されるテストを追加 | テストが PASS | pytest |
|10 | 評価ユニットテスト | `evaluate_model.py` がダミーモデルで実行できるか検証 | `pytest` で PASS | pytest |
|11 | ドキュメント更新 | `docs/M5_setup.md` に学習手順を記載 | Markdown リンクチェックが通る | 手順に沿ってモデル学習 | Markdown |
|12 | M5 完了レビュー | 10 戦評価で学習モデルがランダムより勝率が高いことを確認 | レビューコメントで承認 | 実戦テスト | |

> **備考**
> - `PokemonEnv_Specification.md` の観測・行動仕様を遵守し、学習スクリプトでも同じインターフェースを利用すること。
> - Self-Play 用に 2 エージェントを登録するが、初期実装では両者の重みを共有するシンプルな方式とする。
