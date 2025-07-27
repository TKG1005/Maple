# プロファイリングシステム

Mapleフレームワークの訓練性能分析のための包括的プロファイリングシステム。

## 概要

このシステムは、Pokemon強化学習の訓練過程における詳細な性能分析を提供します。ThreadPoolExecutorによる並列実行環境での正確な計測を実現し、ボトルネック特定と最適化指針を提供します。

## 主要機能

### 🔍 **性能メトリクス**

**Environment Operations**:
- `env_reset`: 環境リセット時間
- `env_step`: 環境ステップ実行時間（最重要ボトルネック）
- `env_parallel_execution`: 並列環境実行全体時間

**Learning Operations**:
- `gradient_calculation`: 勾配計算とGAE処理
- `optimizer_step`: オプティマイザー更新処理
- `loss_calculation`: 損失計算処理

**Agent Operations**:
- `agent_action_selection`: エージェントの行動選択
- `agent_value_calculation`: 価値関数計算

**System Metrics**:
- CPU使用率、メモリ使用量、GPU使用率
- ピークメモリ使用量

### 🏗️ **ハイブリッドアーキテクチャ**

1. **メイン関数レベル**: 全体時間とThreadPoolExecutor処理
2. **代表エピソード詳細計測**: 最初のエピソードの1環境で詳細プロファイリング
3. **スレッドセーフ**: グローバルプロファイラーによる安全な並列計測

## 使用方法

### 基本的な使用

```bash
# プロファイリング有効化
python train.py --profile --profile-name session_name

# 設定例
python train.py --device cpu --episodes 10 --profile --profile-name debug_test
```

### 出力ファイル

プロファイリング結果は以下に保存されます：

```
logs/profiling/
├── raw/                    # JSONデータ
│   └── session_name_timestamp.json
└── reports/               # 人間可読レポート
    └── session_name_timestamp_summary.txt
```

### レポート例

```
PERFORMANCE SUMMARY (per episode)
----------------------------------------
Total Episode Time: 11.964s

Environment Operations:
  Reset: 0.160s (1.3%)
  Step: 1.397s (11.7%)       ← 最重要ボトルネック

Learning Operations:
  Gradient Calculation: 4.202s (35.1%)
  Optimizer Step: 4.056s (33.9%)

Agent Operations:
  Action Selection: 0.125s (1.0%)
  Value Calculation: 0.126s (1.1%)

SYSTEM RESOURCE USAGE
------------------------------
Peak Memory Usage: 32.83 GB
Average CPU Usage: 9.0%
```

## 技術的実装

### ThreadPoolExecutor問題の解決

**問題**: 並列実行により同じプロファイラーに重複計測が発生
- 修正前: `env_step` 1108% (205.6秒/18.6秒)
- 修正後: `env_step` 11.7% (正常)

**解決策**: ハイブリッド方式
- メイン関数: 全体時間計測
- 代表環境: 詳細計測（`ep == 0 and i == 0`）

### 設定統合

`batch_size`読み込み問題を修正し、全31項目の設定を完全対応：

```python
# train.py:430-431
batch_size = int(cfg.get("batch_size", 4096))
buffer_capacity = int(cfg.get("buffer_capacity", 800000))
```

### コード構造

**プロファイラー取得**:
```python
from src.profiling import get_global_profiler

profiler = get_global_profiler() if enable_profiling else None
```

**計測例**:
```python
if profiler:
    with profiler.profile('env_step'):
        observations, rewards, terms, truncs, _, next_masks = env.step(actions)
else:
    observations, rewards, terms, truncs, _, next_masks = env.step(actions)
```

## 性能分析指針

### ボトルネック特定

1. **Environment Step (10-15%)**: Pokemon Showdown通信が主要ボトルネック
   - 対策: サーバー最適化、並列数調整

2. **Learning Operations (70%)**: 学習処理が期待通りの大部分
   - gradient_calculation: GAE計算とバッチ処理
   - optimizer_step: ネットワーク更新

3. **Agent Operations (2-3%)**: 推論処理は軽量
   - 最適化余地は限定的

### 最適化優先順位

1. **高優先**: Environment通信最適化
2. **中優先**: バッチサイズとネットワーク調整
3. **低優先**: エージェント推論最適化

## トラブルシューティング

### よくある問題

**Q: 統計が0%の項目がある**
A: 極短時間の処理は計測されない場合があります。主要メトリクス（env_step, learning）が正常であれば問題ありません。

**Q: CPU使用率が0%**
A: システムモニタリングの問題です。メモリ使用量は正常に取得されます。

**Q: 時間が異常に長い/短い**
A: ThreadPoolExecutor問題の可能性があります。ハイブリッド方式で解決済みです。

## 将来の拡張

- バトル詳細メトリクス
- GPU使用率の詳細分析  
- ネットワーク通信プロファイリング
- カスタムメトリクス追加

## 関連ファイル

- `src/profiling/profiler.py`: コアプロファイラー実装
- `src/profiling/logger.py`: ログ出力とレポート生成
- `train.py`: メイン訓練スクリプトでの統合
- `config/train_config.yml`: プロファイリング設定