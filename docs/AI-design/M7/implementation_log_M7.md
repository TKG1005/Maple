# 実装ログ M7 - 設定ファイルシステムと勝率ベース対戦相手更新機能

## 実装概要

### 1. TensorBoardログ分析
- **ファイル**: `runs/Jul09_17-01-05_MacBook-Pro.local/events.out.tfevents.1752048065.MacBook-Pro.local.32766.0`
- **結果**: 5000エピソード後の勝率59.5%、報酬12.5%改善を確認
- **評価**: 強い学習進捗と安定した性能向上を確認

### 2. 勝率ベース対戦相手更新システム実装

#### 問題
- 従来のセルフプレイでは毎エピソード後に対戦相手ネットワークをコピー
- 過度なコピー頻度により学習効率が低下する可能性

#### 解決策
勝率が閾値（60%）を超えた場合のみ対戦相手を更新する条件付きシステムを実装

#### 実装詳細

**主要コンポーネント (train.py)**:
```python
# 勝率ベース対戦相手更新システム
recent_battle_results = []  # 最近の戦闘結果保存 (1=勝利, 0=引き分け, -1=敗北)
opponent_snapshots = {}     # 対戦相手ネットワークスナップショット保存
current_opponent_id = 0     # 現在使用中の対戦相手スナップショットID

def should_update_opponent(episode_num, battle_results, window_size, threshold):
    """最近の勝率に基づいて対戦相手を更新すべきかチェック"""
    if len(battle_results) < window_size:
        return False
    recent_results = battle_results[-window_size:]
    wins = sum(1 for result in recent_results if result == 1)
    win_rate = wins / len(recent_results)
    return win_rate >= threshold
```

**設定パラメータ**:
- `win_rate_threshold`: 勝率閾値（デフォルト0.6）
- `win_rate_window`: 勝率計算用の最近戦闘数（デフォルト50）

### 3. 設定ファイルベースパラメータ管理システム

#### 問題
- 毎回長いコマンドラインオプションを入力する必要があった
- パラメータの管理と再利用が困難

#### 解決策
YAMLベースの設定ファイルシステムを実装し、デフォルト値を一括管理

#### 実装詳細

**設定ファイル構造**:

1. **`config/train_config.yml`** - デフォルト設定
```yaml
episodes: 10
lr: 0.0005
batch_size: 2048
algorithm: ppo
reward: composite

# 勝率ベース対戦相手更新
win_rate_threshold: 0.6
win_rate_window: 50

# ネットワーク設定
network:
  type: "lstm"
  hidden_size: 128
  use_lstm: true
```

2. **`config/train_config_quick.yml`** - 高速テスト用
```yaml
episodes: 5
opponent_mix: "random:0.3,max:0.3,self:0.4"
win_rate_threshold: 0.5
win_rate_window: 10
network:
  type: "basic"
  hidden_size: 64
```

3. **`config/train_config_long.yml`** - 本格訓練用
```yaml
episodes: 1000
win_rate_threshold: 0.65
win_rate_window: 100
network:
  type: "attention"
  hidden_size: 256
  attention_heads: 8
```

**設定読み込み機能**:
```python
def load_config(path: str) -> dict:
    """YAML設定ファイルから訓練設定を読み込み"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data
    except FileNotFoundError:
        logger.warning("Config file %s not found, using defaults", path)
        return {}
```

### 4. ネットワーク互換性修正

#### 問題
基本ネットワークと拡張ネットワーク（LSTM/Attention）でforward()メソッドのシグネチャが異なる

#### 解決策
ネットワークタイプに応じた条件分岐を実装
```python
# Call value network with hidden state only if supported
if hasattr(value_net, 'hidden_state'):
    val0_tensor = value_net(obs0_tensor, value_net.hidden_state)
else:
    val0_tensor = value_net(obs0_tensor)
```

## 使用方法

### 基本的な使用法
```bash
# デフォルト設定で訓練
python train.py --config config/train_config.yml

# 高速テスト
python train.py --config config/train_config_quick.yml

# 本格訓練
python train.py --config config/train_config_long.yml
```

### コマンドライン上書き
```bash
# 設定ファイル + 個別パラメータ上書き
python train.py --config config/train_config.yml --episodes 20 --lr 0.001
```

## 技術的詳細

### 勝率追跡メカニズム
- win_loss報酬コンポーネントを使用して自動的に勝敗を検出
- 循環バッファで最近の戦闘結果を管理
- 設定可能なウィンドウサイズで勝率を計算

### 対戦相手スナップショット管理
- 条件を満たした時のみネットワーク状態を保存
- メモリ効率的なスナップショット保持
- セルフプレイ時の対戦相手選択の最適化

### 設定パラメータの優先順位
1. コマンドライン引数（最高優先度）
2. 設定ファイル値
3. プログラムデフォルト値（最低優先度）

## 効果と改善点

### 効果
- **操作性向上**: 長いコマンドライン入力が不要
- **設定管理**: 異なる訓練シナリオの設定を簡単に切り替え
- **学習効率**: 勝率ベース更新により過度なネットワークコピーを削減
- **再現性**: 設定ファイルによる完全な実験再現が可能

### 検証結果
- 設定ファイルシステムが正常に動作することを確認
- 勝率ベース対戦相手更新システムが設定通りに機能
- 全パラメータが設定ファイルから正しく読み込まれることを確認
- 基本・LSTM・Attentionネットワーク全てで動作することを確認

## ファイル更新一覧

### 主要ファイル
- `train.py`: 勝率システムと設定ファイル読み込み機能を追加
- `config/train_config.yml`: デフォルト設定ファイル
- `config/train_config_quick.yml`: 高速テスト用設定
- `config/train_config_long.yml`: 本格訓練用設定  
- `config/train_config_attention.yml`: Attentionネットワーク設定

### 設定パラメータ追加
- `win_rate_threshold`: 勝率閾値
- `win_rate_window`: 勝率計算ウィンドウ
- `network`: ネットワーク設定セクション
- その他全ての訓練パラメータの設定ファイル対応

## 今後の拡張可能性

1. **動的閾値調整**: 訓練進捗に応じた閾値の自動調整
2. **複数スナップショット管理**: 異なる強さの対戦相手プールの維持
3. **設定テンプレート**: 特定タスク用の事前定義済み設定セット
4. **性能メトリクス**: 設定ファイルでの詳細ログ設定管理

---

**実装日**: 2025年7月10日  
**関連バージョン**: M7実装フェーズ  
**動作確認**: 設定ファイルシステム及び勝率ベース更新システムの正常動作を確認済み