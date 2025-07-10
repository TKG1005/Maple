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
- 2025-07-09 **緊急修正**: 自己対戦システムの重大な問題を修正し、真の自己対戦学習を実現
  - 両エージェントが同じネットワークを共有していた問題を修正
  - 単一モデル収束アプローチを実装（主エージェントが学習、対戦相手は凍結コピー）
  - 学習率を0.002から0.0005に最適化して学習安定性を向上
- 2025-07-09 **新機能**: 包括的な報酬正規化システムを実装
  - `RewardNormalizer` クラスによる実行統計ベースの正規化
  - `WindowedRewardNormalizer` による滑動窓正規化（代替手法）
  - エージェント別の独立した正規化器で学習安定性を向上
  - `PokemonEnv` で自動的に報酬正規化を適用、`normalize_rewards` パラメータで制御可能
- 2025-07-10 **重要アップデート**: LSTM隠れ状態管理とシーケンシャル学習を修正
  - `RLAgent` にバッチ処理対応の隠れ状態管理を実装
  - エピソード境界での隠れ状態リセット機能を追加
  - LSTM/Attentionネットワークでの学習安定性を大幅改善
- 2025-07-10 **新機能**: 設定ファイルシステムと勝率ベース対戦相手更新を実装
  - YAMLベースの包括的なパラメータ管理システム
  - 勝率閾値（60%）による効率的な対戦相手更新機能
  - `config/train_config_quick.yml`（高速テスト）、`config/train_config_long.yml`（本格訓練）等の設定テンプレート
  - コマンドライン引数の大幅簡素化とパラメータ管理の改善
- 2025-07-10 **重要アップデート**: LSTM競合解消とGPU対応を実装
  - **LSTM並列実行問題の根本解決**: 隠れ状態を各エージェントで独立管理し、並列環境での競合を完全解消
  - **包括的GPU対応**: NVIDIA CUDA、Apple MPS、CPU自動検出とデバイス間の最適化
  - **ステートレス設計**: ネットワークが隠れ状態を内部保持せず、戻り値として返す設計に変更
  - **マルチプラットフォーム対応**: Windows（CUDA）、macOS（MPS）、Linux（CUDA/CPU）で統一的な動作
  - **パフォーマンス向上**: GPU加速による大幅な学習速度向上とメモリ効率化

## 新しい使用方法（設定ファイルシステム）

### 設定ファイルベースの訓練

```bash
# テスト・短時間駆動（10エピソード、混合対戦相手）
python train_selfplay.py --config config/train_config.yml

# 長時間学習（1000エピソード、セルフプレイ、Attentionネットワーク）
python train_selfplay.py --config config/train_config_long.yml
```

### 設定ファイルとコマンドライン引数の組み合わせ

```bash
# 設定ファイル + 個別パラメータ上書き
python train_selfplay.py --config config/train_config.yml --episodes 20 --lr 0.001

# 勝率閾値を変更してセルフプレイ
python train_selfplay.py --config config/train_config.yml --win-rate-threshold 0.7
```

### 利用可能な設定ファイル

- `config/train_config.yml`: テスト・短時間駆動用（100エピソード、混合対戦相手、LSTMネットワーク、GPU対応）
- `config/train_config_long.yml`: 長時間学習用（1000エピソード、セルフプレイ、Attentionネットワーク、GPU最適化）

### 勝率ベース対戦相手更新システム

セルフプレイ時に勝率が設定した閾値（デフォルト60%）を超えた場合のみ対戦相手ネットワークを更新します：

```yaml
# config/train_config.yml
win_rate_threshold: 0.6  # 60%の勝率で更新
win_rate_window: 50      # 最近50戦の勝率を監視
```

このシステムにより：
- 過度なネットワークコピーを削減
- 学習効率の向上
- 安定した対戦相手との継続的な学習

## GPU対応とデバイス選択

### 自動デバイス検出

Mapleは利用可能なハードウェアを自動検出し、最適なデバイスを選択します：

```bash
# 自動検出（推奨）
python train_selfplay.py --config config/train_config.yml
# 検出順序: CUDA > Apple MPS > CPU

# 手動デバイス指定
python train_selfplay.py --device cuda    # NVIDIA GPU強制
python train_selfplay.py --device mps     # Apple Silicon GPU強制
python train_selfplay.py --device cpu     # CPU強制
```

### 対応デバイス

- **NVIDIA CUDA**: Windows/Linux環境でのCUDA対応GPU
- **Apple MPS**: Apple Silicon Mac（M1/M2/M3）のMetal Performance Shaders
- **CPU**: 全プラットフォームでのフォールバック実行

### GPU最適化設定

GPU使用時の推奨設定（自動的に適用）：

```yaml
# GPU最適化された設定例
batch_size: 2048      # 大きなバッチサイズでGPU効率向上
buffer_capacity: 4096 # メモリ使用量と性能のバランス
parallel: 10          # 並列環境数（GPU性能に応じて調整）
```

## LSTM並列実行の改善

### 問題の解決

以前のバージョンでは、LSTM/Attentionネットワークを並列環境で使用する際に隠れ状態の競合が発生していました。この問題を根本的に解決：

- **エージェント別状態管理**: 各RLAgentが独自の隠れ状態を管理
- **ステートレスネットワーク**: ネットワーク内部に状態を保持しない設計
- **エピソード境界リセット**: 適切な隠れ状態のクリア処理
- **スレッドセーフ**: 並列実行時の安全性を保証

### 使用例

```bash
# LSTM + 並列実行 + GPU加速
python train_selfplay.py \
  --config config/train_config.yml \
  --parallel 10 \
  --device mps \
  --episodes 100
```
