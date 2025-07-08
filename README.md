# Maple

PokemonEnv を利用したポケモンバトル強化学習フレームワークです。

## 変更履歴

- 2025-06-13 `PokemonEnv.reset()` と `step()` に `return_masks` オプションを追加し、
  観測とあわせて利用可能な行動マスクを返すよう更新
- 2025-06-27 `EnvPlayer` が `battle.last_request` の変化を監視し、更新を確認してから
  `PokemonEnv` に `battle` オブジェクトを送信するよう改善
- 2025-06-28 `train_selfplay.py` にチェックポイント保存オプションを追加し、一定間隔で
  モデルを自動保存可能に
- 2025-06-29 対戦ログ比較用の `plot_compare.py` を新規追加し、学習結果をグラフで確認できるように
- 2025-06-29 `SingleAgentCompatibilityWrapper` の `reset()` と `step()` が `return_masks` を受け取り
  `PokemonEnv` の行動マスクと連携するよう更新
- 2025-07-01 PPO 対応手順をまとめた `docs/M7_setup.md` を追加し、`train_selfplay.py` の `--algo` オプションでアルゴリズムを切り替え可能に
- 2025-07-05 `HPDeltaReward` を追加し、`--reward hp_delta` オプションで使用可能に（後にPokemonCountRewardに置き換え）
- 2025-07-06 `train_selfplay.py` の報酬を `CompositeReward` ベースに変更し、
  `config/train_config.yml` の設定で他の報酬へ切り替え可能に
- 2025-07-02 `CLAUDE.md` ファイルを新規作成し、Claude Code用のコードベース解説と
  開発コマンド、アーキテクチャ概要をドキュメント化
- 2025-07-02 `FailAndImmuneReward` を実装し、無効行動（失敗・無効技）時の
  ペナルティ機能を追加。`config/reward.yaml` の `fail_immune` で有効化可能
- 2025-07-07 `train_selfplay.py` に `--load-model` オプションを追加し、チェックポイントから
  学習を再開可能に。ファイル名から自動的にエピソード番号を抽出
- 2025-07-07 `train_selfplay.py` と `evaluate_rl.py` に `--team random` オプションを追加し、
  ランダムチーム機能を実装。各プレイヤーが独立してランダムチームを選択
- 2025-07-07 `evaluate_rl.py` でNameTakenErrorを修正し、一意なプレイヤー名生成により
  Pokemon Showdownサーバーとの接続問題を解決
- 2025-07-07 `PokemonCountReward` を実装し、対戦終了時の残数差による報酬システムを追加。
  `HPDeltaReward` による総HP報酬システムを削除し、よりシンプルな終了時スコアリングに変更
