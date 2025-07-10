# トレーニング例とコマンド参考資料

## 基本的なトレーニング

### 設定ファイルを使用した基本実行

```bash
# デフォルト設定でテスト実行（100エピソード、LSTM、混合対戦相手）
python train_selfplay.py --config config/train_config.yml

# 長時間学習設定（1000エピソード、Attention、セルフプレイ）
python train_selfplay.py --config config/train_config_long.yml
```

### デバイス選択

```bash
# 自動デバイス検出（推奨）
python train_selfplay.py --config config/train_config.yml

# 特定デバイス強制実行
python train_selfplay.py --config config/train_config.yml --device cuda  # NVIDIA GPU
python train_selfplay.py --config config/train_config.yml --device mps   # Apple Silicon
python train_selfplay.py --config config/train_config.yml --device cpu   # CPU強制
```

## GPU最適化トレーニング

### Apple Silicon (MPS) 最適化

```bash
# Apple Silicon での最適化実行
python train_selfplay.py \
  --config config/train_config.yml \
  --device mps \
  --parallel 8 \
  --episodes 200

# バッチサイズ調整でメモリ効率向上
python train_selfplay.py \
  --config config/train_config.yml \
  --device mps \
  --lr 0.0003 \
  --episodes 500
```

### NVIDIA GPU (CUDA) 最適化

```bash
# CUDA GPU での高速トレーニング
python train_selfplay.py \
  --config config/train_config_long.yml \
  --device cuda \
  --parallel 12 \
  --episodes 1000

# 大容量GPU向け設定
python train_selfplay.py \
  --config config/train_config_long.yml \
  --device cuda \
  --parallel 16 \
  --episodes 2000
```

## LSTM並列トレーニング

### 安全な並列LSTM実行

```bash
# LSTM + 並列実行（隠れ状態競合問題解決済み）
python train_selfplay.py \
  --config config/train_config.yml \
  --parallel 10 \
  --episodes 300

# 高並列度でのLSTM学習
python train_selfplay.py \
  --config config/train_config.yml \
  --parallel 20 \
  --episodes 500 \
  --device cuda
```

### ネットワーク種別選択

```bash
# LSTM ネットワーク（シーケンシャル学習）
python train_selfplay.py --config config/train_config.yml --episodes 200

# Attention ネットワーク（高度な特徴抽出）
python train_selfplay.py --config config/train_config_long.yml --episodes 500

# 基本ネットワーク（軽量・高速）
python train_selfplay.py \
  --config config/train_config.yml \
  --episodes 100 \
  --lr 0.001
```

## 対戦相手バリエーション

### セルフプレイ学習

```bash
# 勝率ベース対戦相手更新
python train_selfplay.py \
  --config config/train_config.yml \
  --episodes 300 \
  --win-rate-threshold 0.65 \
  --win-rate-window 100

# 積極的な対戦相手更新（低い閾値）
python train_selfplay.py \
  --config config/train_config.yml \
  --episodes 200 \
  --win-rate-threshold 0.55 \
  --win-rate-window 50
```

### 混合対戦相手学習

```bash
# バランス型混合対戦相手
python train_selfplay.py \
  --config config/train_config.yml \
  --opponent-mix "random:0.2,max:0.3,rule:0.2,self:0.3" \
  --episodes 400

# セルフプレイ重視
python train_selfplay.py \
  --config config/train_config.yml \
  --opponent-mix "max:0.2,self:0.8" \
  --episodes 300

# 多様性重視
python train_selfplay.py \
  --config config/train_config.yml \
  --opponent-mix "random:0.4,max:0.3,rule:0.3" \
  --episodes 250
```

### 単一対戦相手学習

```bash
# 最大ダメージ相手特化
python train_selfplay.py \
  --config config/train_config.yml \
  --opponent max \
  --episodes 200

# ルールベース相手特化
python train_selfplay.py \
  --config config/train_config.yml \
  --opponent rule \
  --episodes 150

# ランダム相手でのロバスト性向上
python train_selfplay.py \
  --config config/train_config.yml \
  --opponent random \
  --episodes 300
```

## チーム戦略

### ランダムチーム学習

```bash
# 多様なチーム構成での学習
python train_selfplay.py \
  --config config/train_config.yml \
  --team random \
  --teams-dir config/teams \
  --episodes 400

# カスタムチームディレクトリ
python train_selfplay.py \
  --config config/train_config.yml \
  --team random \
  --teams-dir my_teams \
  --episodes 300
```

### 固定チーム学習

```bash
# デフォルトチームでの一貫学習
python train_selfplay.py \
  --config config/train_config.yml \
  --team default \
  --episodes 500
```

## 継続学習・モデル管理

### チェックポイントからの再開

```bash
# 保存済みモデルから学習再開
python train_selfplay.py \
  --config config/train_config.yml \
  --load-model checkpoints/checkpoint_ep500.pt \
  --episodes 200

# 学習率調整して再開
python train_selfplay.py \
  --config config/train_config.yml \
  --load-model checkpoints/checkpoint_ep1000.pt \
  --lr 0.0002 \
  --episodes 500
```

### 定期的な保存

```bash
# 50エピソードごとの自動保存
python train_selfplay.py \
  --config config/train_config.yml \
  --checkpoint-interval 50 \
  --checkpoint-dir my_checkpoints \
  --episodes 500

# 最終モデル保存
python train_selfplay.py \
  --config config/train_config.yml \
  --episodes 200 \
  --save final_model.pt
```

## アルゴリズム設定

### PPO学習

```bash
# 標準PPO設定
python train_selfplay.py \
  --config config/train_config.yml \
  --algo ppo \
  --ppo-epochs 4 \
  --clip 0.2 \
  --episodes 300

# 保守的PPO（小さなclip範囲）
python train_selfplay.py \
  --config config/train_config.yml \
  --algo ppo \
  --ppo-epochs 6 \
  --clip 0.1 \
  --episodes 400

# 積極的PPO（大きなclip範囲）
python train_selfplay.py \
  --config config/train_config.yml \
  --algo ppo \
  --ppo-epochs 3 \
  --clip 0.3 \
  --episodes 250
```

### REINFORCE学習

```bash
# シンプルなREINFORCE
python train_selfplay.py \
  --config config/train_config.yml \
  --algo reinforce \
  --lr 0.001 \
  --episodes 200

# エントロピー正則化強化
python train_selfplay.py \
  --config config/train_config.yml \
  --algo reinforce \
  --entropy-coef 0.02 \
  --episodes 300
```

## デバッグ・開発用

### 高速テスト実行

```bash
# 1エピソードでの動作確認
python train_selfplay.py \
  --config config/train_config.yml \
  --episodes 1 \
  --parallel 1

# 少数エピソードでのGPU動作確認
python train_selfplay.py \
  --config config/train_config.yml \
  --episodes 5 \
  --device mps \
  --parallel 2
```

### ログ・可視化

```bash
# TensorBoard有効化
python train_selfplay.py \
  --config config/train_config.yml \
  --tensorboard \
  --episodes 200

# 詳細ログレベル
python train_selfplay.py \
  --log-level DEBUG \
  --config config/train_config.yml \
  --episodes 50
```

## パフォーマンスチューニング

### メモリ効率重視

```bash
# 小さなバッチサイズでメモリ節約
python train_selfplay.py \
  --config config/train_config.yml \
  --episodes 300 \
  --parallel 5

# バッファ容量調整
python train_selfplay.py \
  --config config/train_config.yml \
  --episodes 400 \
  --parallel 8
```

### 速度重視

```bash
# 大きな並列度で高速化
python train_selfplay.py \
  --config config/train_config.yml \
  --parallel 15 \
  --episodes 500 \
  --device cuda

# GPU最適化バッチサイズ
python train_selfplay.py \
  --config config/train_config_long.yml \
  --device mps \
  --episodes 800
```

## 実験設計例

### A/Bテスト

```bash
# 設定A: LSTM + セルフプレイ
python train_selfplay.py \
  --config config/train_config.yml \
  --episodes 300 \
  --save model_lstm_selfplay.pt

# 設定B: Attention + 混合対戦相手  
python train_selfplay.py \
  --config config/train_config_long.yml \
  --opponent-mix "max:0.5,self:0.5" \
  --episodes 300 \
  --save model_attention_mixed.pt
```

### ハイパーパラメータ探索

```bash
# 学習率探索
for lr in 0.0001 0.0005 0.001 0.002; do
  python train_selfplay.py \
    --config config/train_config.yml \
    --lr $lr \
    --episodes 200 \
    --save model_lr_${lr}.pt
done

# バッチサイズ探索
for batch in 1024 2048 4096; do
  python train_selfplay.py \
    --config config/train_config.yml \
    --episodes 150 \
    --save model_batch_${batch}.pt
done
```

### 長期学習実験

```bash
# 段階的学習（カリキュラム学習）
# ステップ1: 基礎学習
python train_selfplay.py \
  --config config/train_config.yml \
  --opponent max \
  --episodes 200 \
  --save curriculum_step1.pt

# ステップ2: 混合学習  
python train_selfplay.py \
  --config config/train_config.yml \
  --load-model curriculum_step1.pt \
  --opponent-mix "max:0.5,rule:0.5" \
  --episodes 300 \
  --save curriculum_step2.pt

# ステップ3: セルフプレイ
python train_selfplay.py \
  --config config/train_config.yml \
  --load-model curriculum_step2.pt \
  --episodes 500 \
  --save curriculum_final.pt
```

## トラブルシューティング

### メモリ不足対策

```bash
# 並列度を下げる
python train_selfplay.py \
  --config config/train_config.yml \
  --parallel 3 \
  --episodes 200

# バッチサイズを下げる
python train_selfplay.py \
  --config config/train_config.yml \
  --episodes 200
```

### GPU利用できない場合

```bash
# CPU強制実行
python train_selfplay.py \
  --config config/train_config.yml \
  --device cpu \
  --parallel 4 \
  --episodes 100
```

### 学習が不安定な場合

```bash
# 学習率を下げる
python train_selfplay.py \
  --config config/train_config.yml \
  --lr 0.0002 \
  --episodes 300

# クリップ範囲を小さくする
python train_selfplay.py \
  --config config/train_config.yml \
  --clip 0.1 \
  --episodes 250
```

## 評価とテスト

### モデル評価

```bash
# 基本評価
python evaluate_rl.py \
  --model checkpoints/checkpoint_ep500.pt \
  --opponent random \
  --n 20

# ランダムチームでの評価
python evaluate_rl.py \
  --model checkpoints/final_model.pt \
  --opponent rule \
  --team random \
  --teams-dir config/teams \
  --n 10

# 複数モデル比較
python evaluate_rl.py \
  --models checkpoints/model_a.pt checkpoints/model_b.pt \
  --n 15
```

### 結果の可視化

```bash
# 学習結果比較
python plot_compare.py

# TensorBoard起動（トレーニング中/後）
tensorboard --logdir runs/
```

これらの例を参考に、目的に応じたトレーニング設定を選択してください。GPU対応とLSTM並列実行の改善により、従来よりも効率的で安定した学習が可能になっています。