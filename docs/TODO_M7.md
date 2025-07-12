# M7 TODO リスト

以下は `docs/AI-design/M7/M7_backlog.md` に基づいた実装タスクの一覧です。
進捗管理のためチェックボックス形式で列挙します。

## 報酬シェイピング
- [x] R-1 HP差スコア報酬クラス追加
- [x] R-2 撃破/被撃破 ±1 実装
- [x] R-3 報酬合成マネージャ
- [ ] R-4 状態異常付与時の報酬クラス追加
- [ ] R-5 積み技成功時の微加点
- [ ] R-6 交代直後にタイプ有利になった場合の加点
- [x] R-7 毎ターン経過ペナルティ (TurnPenaltyReward として実装完了)
- [x] R-8 最終味方数-相手数で差分報酬 (PokemonCountReward として実装完了)
- [ ] R-9 報酬合成管理の強化
- [x] R-10 無効行動時のボーナス・ペナルティ (FailAndImmuneReward として実装完了)

## 対戦相手の多様化
- [x] B-1 RandomBot 実装 (実装完了)
- [x] B-2 MaxDamageBot 実装 (実装完了)
- [x] B-3 学習相手スケジューラ (OpponentPool および --opponent-mix として実装完了)

## ネットワーク拡張
- [x] N-1 MLP 2 層化 (network_factory.py で実装完了)
- [x] N-2 LSTM ヘッダ追加 (network_factory.py で実装完了)
- [x] N-3 アテンション試験フック (network_factory.py で実装完了)

## 探索戦略強化
- [x] E-1 PPO エントロピー係数 config 化 (config/train_config.yml で実装完了)
- [ ] E-2 ε-greedy wrapper 実装

## 評価 & ロギング
- [ ] V-1 TensorBoard スカラー整理
- [ ] V-2 CSV エクスポートユーティリティ

## 緊急修正 (2025-07-09)
- [x] **自己対戦システムの修正**: 両エージェントが同じネットワークを共有していた問題を修正
- [x] **単一モデル収束**: 主エージェントが学習し、対戦相手は凍結コピーを使用する設計に変更
- [x] **学習率最適化**: 0.002 → 0.0005 に変更して学習安定性を向上
- [x] **報酬正規化実装**: RewardNormalizer クラスによる実行統計ベースの正規化システム
- [x] **アルゴリズム対応**: PPO/REINFORCE でオプティマイザーなしのエージェントをサポート
- [x] **設定ファイル更新**: train_config.yml と reward.yaml の最適化

## 重要アップデート (2025-07-10)
- [x] **LSTM隠れ状態管理修正**: シーケンシャル学習の実現
  - バッチ処理対応の隠れ状態管理を RLAgent に実装
  - エピソード境界での隠れ状態リセット機能を追加
  - LSTM/Attentionネットワークでの学習安定性を大幅改善
- [x] **設定ファイルシステム実装**: 包括的なパラメータ管理
  - YAMLベースの設定ファイル（test/long）
  - コマンドライン引数の大幅簡素化
  - 全訓練パラメータの設定ファイル対応
- [x] **勝率ベース対戦相手更新システム**: 効率的なセルフプレイ学習
  - 勝率閾値（デフォルト60%）による条件付き対戦相手更新
  - 対戦相手スナップショット管理システム
  - 過度なネットワークコピー削減による学習効率向上
- [x] **ネットワーク互換性修正**: 全ネットワーク対応
  - 基本・LSTM・Attentionネットワークの統一インターフェース
  - 条件分岐による forward メソッド互換性確保

## 評価 & ロギング
- [ ] V-1 TensorBoard スカラー整理
- [ ] V-2 CSV エクスポートユーティリティ
- [ ] V-3 行動多様性メトリクス

## アルゴリズム追加
- [ ] A-1 A2C 実装
- [ ] A-2 ハイパラ検索スクリプト

## ポケモン種族名Embedding (2025-07-12)
- [x] **EmbeddingInitializer実装**: 種族値による初期化機能
- [x] **EmbeddingPolicyNetwork実装**: 基本的なEmbedding統合
- [x] **EmbeddingValueNetwork実装**: Value network対応
- [x] **Network Factory統合**: 既存システムとの統合
- [x] **Configuration設定追加**: YAML設定でのEmbedding制御
- [x] **Unit Test作成**: Embedding層の動作確認（17テストケース）
- [x] **freeze_base_stats機能**: gradient masking実装
- [x] **League Training実装**: 破滅的忘却対策

## 並列処理最適化 (2025-07-12)
- [x] **TensorBoard分析実行**: parallel=5,10,20,30の効率測定
- [x] **最適設定特定**: parallel=5が最高効率（0.76 battles/sec）
- [x] **設定ファイル最適化**: train_config.ymlをparallel=5に設定

## CI / 自動化
- [ ] C-1 GitHub Actions スモーク
- [x] C-2 Codex/LLM 用 TODO.md (CLAUDE.md として実装完了)
- [ ] C-3 Pre-commit Black + ruff
