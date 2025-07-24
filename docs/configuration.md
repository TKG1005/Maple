# 設定ファイルガイド

このドキュメントでは、Mapleの設定ファイルシステムについて詳しく説明します。

## 設定ファイルの概要

Mapleは YAML ベースの設定ファイルシステムを使用して、トレーニングパラメータを管理します。設定ファイルにより、コマンドライン引数の複雑さを軽減し、実験の再現性を向上させます。

## メイン設定ファイル

### train_config.yml (デフォルト設定)

テスト・短時間駆動用の設定ファイルです：

```yaml
# 基本設定
episodes: 100
lr: 0.0005
algorithm: ppo
parallel: 10

# PPO設定
ppo_epochs: 4
clip_range: 0.2
value_coef: 0.6
entropy_coef: 0.01

# GAE設定
gamma: 0.997
gae_lambda: 0.95

# バッチ設定
batch_size: 2048
buffer_capacity: 4096

# 対戦相手設定
opponent: max
opponent_mix: "max:0.5,self:0.5"

# チーム設定
team: "random"

# ネットワーク設定
network:
  type: "lstm"
  hidden_size: 128
  lstm_hidden_size: 128
  use_lstm: true
  use_2layer: true
```

### train_config_long.yml (長時間学習用)

本格的な学習用の設定ファイルです：

```yaml
# 長時間学習設定
episodes: 1000
lr: 0.0003
parallel: 8

# 高度なネットワーク
network:
  type: "attention"
  hidden_size: 256
  use_attention: true
  attention_heads: 8
  use_2layer: true

# セルフプレイ設定
win_rate_threshold: 0.65
win_rate_window: 100

# GPU最適化
batch_size: 4096
buffer_capacity: 8192
```

## ネットワーク設定

### LSTM ネットワーク

```yaml
network:
  type: "lstm"
  hidden_size: 128          # 隠れ層サイズ
  lstm_hidden_size: 128     # LSTM隠れ状態サイズ
  use_lstm: true            # LSTM有効化
  use_2layer: true          # 2層MLP使用
```

### Attention ネットワーク

```yaml
network:
  type: "attention"
  hidden_size: 256          # 隠れ層サイズ
  use_attention: true       # Attention有効化
  attention_heads: 8        # Multi-head数
  attention_dropout: 0.1    # Attentionドロップアウト
  use_lstm: false           # LSTM無効化
  use_2layer: true          # 2層MLP使用
```

### 基本ネットワーク

```yaml
network:
  type: "basic"
  hidden_size: 128
  use_lstm: false
  use_attention: false
  use_2layer: true
```

## 対戦相手設定

### 単一対戦相手

```yaml
opponent: "max"  # Options: random, max, rule
```

### 混合対戦相手

```yaml
opponent_mix: "random:0.3,max:0.3,self:0.4"
```

形式: `"type1:ratio1,type2:ratio2,type3:ratio3"`

利用可能な対戦相手タイプ：
- `random`: ランダム行動
- `max`: 最大ダメージ優先
- `rule`: ルールベース
- `self`: セルフプレイ

### セルフプレイ設定

```yaml
# セルフプレイモード（opponent と opponent_mix が未指定の場合）
win_rate_threshold: 0.6   # 対戦相手更新の勝率閾値
win_rate_window: 50       # 勝率計算の直近バトル数
```

## GPU・デバイス設定

### 自動デバイス検出

設定ファイルでは `device` を指定せず、コマンドラインで指定：

```bash
python train.py --config config/train_config.yml --device auto
```

### GPU最適化設定

GPU使用時の推奨設定：

```yaml
# GPU最適化パラメータ
batch_size: 2048          # 大きなバッチサイズ
buffer_capacity: 4096     # 十分なバッファ容量
parallel: 10              # 並列環境数（GPU性能に応じて調整）

# メモリ効率化
ppo_epochs: 4             # 適切なエポック数
clip_range: 0.2           # 勾配クリッピング
```

## チーム設定

### デフォルトチーム

```yaml
team: "default"
teams_dir: null
```

### ランダムチーム

```yaml
team: "random"
teams_dir: "config/teams"  # チームファイルディレクトリ
```

## 報酬設定

### 基本報酬設定

```yaml
reward: "composite"
reward_config: "config/reward.yaml"
```

### reward.yaml の例

```yaml
composite:
  enabled: true
  rewards:
    knockout:
      enabled: true
      weight: 1.0
    turn_penalty:
      enabled: true
      weight: 0.02
    fail_immune:
      enabled: true
      weight: 0.02
    pokemon_count:
      enabled: true
      weight: 1.0
```

## チェックポイント設定

```yaml
checkpoint_interval: 100     # 100エピソードごとに保存
checkpoint_dir: "checkpoints"
load_model: null             # 再開用モデルパス
save_model: "final_model.pt" # 最終モデル保存パス
```

## ログ設定

```yaml
tensorboard: true           # TensorBoardログ有効化
```

## 設定ファイルの使用例

### 基本的な使用

```bash
# デフォルト設定で実行
python train.py --config config/train_config.yml

# 長時間学習設定で実行
python train.py --config config/train_config_long.yml
```

### 設定の上書き

```bash
# エピソード数を上書き
python train.py --config config/train_config.yml --episodes 200

# 学習率とデバイスを上書き
python train.py --config config/train_config.yml --lr 0.001 --device cuda

# 複数パラメータの上書き
python train.py \
  --config config/train_config.yml \
  --episodes 50 \
  --parallel 5 \
  --device mps
```

### カスタム設定ファイル

独自の設定ファイルを作成：

```bash
# カスタム設定ファイルを使用
python train.py --config my_custom_config.yml
```

## 設定の優先順位

1. コマンドライン引数（最高優先度）
2. 設定ファイルの値
3. デフォルト値（最低優先度）

## 設定ファイルのベストプラクティス

### 1. 用途別設定ファイル

```
config/
├── train_config.yml          # 一般テスト用
├── train_config_long.yml     # 長時間学習用
├── train_config_gpu.yml      # GPU最適化用
├── train_config_debug.yml    # デバッグ用
└── train_config_production.yml # プロダクション用
```

### 2. コメントの活用

```yaml
# === 基本設定 ===
episodes: 100        # テスト用の短いエピソード数
lr: 0.0005          # 安定した学習率

# === GPU最適化 ===
batch_size: 2048    # GPU効率を考慮したバッチサイズ
parallel: 10        # 並列環境数（GPU性能に依存）
```

### 3. 段階的な設定

```yaml
# 開発段階での設定例
development:
  episodes: 10
  parallel: 2
  tensorboard: false

# テスト段階での設定例  
testing:
  episodes: 100
  parallel: 5
  tensorboard: true

# プロダクション段階での設定例
production:
  episodes: 1000
  parallel: 10
  tensorboard: true
  checkpoint_interval: 100
```

## トラブルシューティング

### 設定ファイルが見つからない

```bash
# 絶対パスで指定
python train.py --config /path/to/config.yml

# 相対パスの確認
ls config/train_config.yml
```

### GPU設定の問題

```bash
# デバイス確認
python -c "from src.utils.device_utils import get_device; print(get_device())"

# CPU強制実行
python train.py --config config/train_config.yml --device cpu
```

### メモリ不足エラー

設定を調整：

```yaml
# メモリ使用量を削減
batch_size: 1024      # バッチサイズを減らす
buffer_capacity: 2048 # バッファ容量を減らす
parallel: 5           # 並列数を減らす
```