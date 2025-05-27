# Maple Project Step-by-Step Development Plan

## Milestone 1: 基礎環境構築
- Task 1.1: Python環境のセットアップ
- Task 1.2: poke-env、Gymnasium、関連ライブラリのインストール
- Task 1.3: 基本的な環境動作テスト（poke-envのexampleコード実行）

## Milestone 2: 基本対戦実行
- Task 2.1: poke-envを用いた対戦実行可能確認
- Task 2.2: 手動登録パーティーによる基本対戦の確認

## Milestone 3: 状態・行動空間定義とルールベースAI
- Task 3.1: 状態観測クラス (StateObserver) の設計・実装
- Task 3.2: 行動ヘルパークラス (ActionHelper) の設計・実装
- Task 3.3: シンプルなルールベースAIの実装と動作確認

## Milestone 4: 強化学習基本ループ実装
- Task 4.1: PokemonEnvクラスの設計 (Gymnasium互換)
- Task 4.2: PokemonEnvの基本メソッド (reset, step) 実装とテスト
- Task 4.3: ランダムエージェントの作成
- Task 4.4: ランダムエージェントとPokemonEnvの学習ループ作成・動作確認

## Milestone 5: 初回強化学習モデル学習と評価
- Task 5.1: 簡単なRLアルゴリズム (例: Q-learning) の導入
- Task 5.2: 初回の強化学習実施（トレーニングループ構築）
- Task 5.3: 学習モデルの基本的評価と改善点抽出

## Milestone 6: 自己対戦学習環境構築
- Task 6.1: 2つのクライアントプログラム間の自己対戦環境の設定
- Task 6.2: 学習済みモデル同士の自己対戦実施・データ収集

## Milestone 7: RLモデル・学習プロセス改善
- Task 7.1: 状態観測精度の改善
- Task 7.2: 行動選択方法の高度化（価値ベース・方策ベース手法検討）
- Task 7.3: 報酬設計の改善・最適化
- Task 7.4: 学習効率向上のためのアルゴリズムチューニング

## Milestone 8: 一定レベルのAI完成
- Task 8.1: 完成モデルによる一定のレーティング到達の確認
- Task 8.2: 最終評価用のベンチマークテスト実施・分析
- Task 8.3: 完成モデルのパッケージングとリリース準備
