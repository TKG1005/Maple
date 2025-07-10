# 価値ネットワーク隠れ状態管理実装 - 2025-07-10

## 概要

LSTM/Attentionネットワークにおける価値ネットワークの隠れ状態管理を統合し、ポリシーネットワークと同様の一貫したインターフェースを提供する実装を完了しました。

## 背景

### 問題

- 価値ネットワークが直接呼び出されており、隠れ状態の管理が不十分
- ポリシーネットワークのみがRLAgent経由で隠れ状態を管理
- 訓練ループでvalue_netを直接呼び出すため、LSTM/Attentionの隠れ状態が活用されない

### 解決すべき課題

1. 価値ネットワークの隠れ状態管理の統合
2. 訓練ループでの統一されたインターフェース
3. エピソード境界での同期リセット
4. シーケンシャル学習の最適化

## 実装内容

### 1. RLAgent.get_value()メソッドの追加

```python
def get_value(self, observation: np.ndarray) -> float:
    """Get state value from value network through RLAgent interface."""
    obs_tensor = torch.as_tensor(observation, dtype=torch.float32, device=next(self.value_net.parameters()).device)
    
    # Handle LSTM/Attention networks with hidden states
    if self.has_hidden_states:
        # Add batch dimension if needed for LSTM
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        # Use stored hidden state
        value, self.value_hidden = self.value_net(obs_tensor, self.value_hidden)
        # Remove batch dimension if we added it
        if value.dim() == 2 and value.size(0) == 1:
            value = value.squeeze(0)
    else:
        # Basic network without hidden states
        value = self.value_net(obs_tensor)
    
    return float(value.item())
```

### 2. 訓練ループの修正

**修正前（train_selfplay.py）:**
```python
obs0_tensor = torch.as_tensor(obs0, dtype=torch.float32, device=device)
if obs0_tensor.dim() == 1:
    obs0_tensor = obs0_tensor.unsqueeze(0)
# Call value network with hidden state only if supported
if hasattr(value_net, 'use_lstm') and (value_net.use_lstm or (hasattr(value_net, 'use_attention') and value_net.use_attention)):
    val0_tensor, _ = value_net(obs0_tensor, None)  # Use None for fresh hidden state in training
else:
    val0_tensor = value_net(obs0_tensor)
if val0_tensor.dim() > 0:
    val0_tensor = val0_tensor.squeeze(0)
val0 = float(val0_tensor.item())
```

**修正後:**
```python
# Get value through RLAgent interface to handle hidden states properly
val0 = rl_agent.get_value(obs0)
```

### 3. 隠れ状態管理の強化

**RLAgent.__init__()の更新:**
```python
# Check if networks support hidden states (LSTM/Attention)
self.has_hidden_states = hasattr(policy_net, 'use_lstm') and (policy_net.use_lstm or (hasattr(policy_net, 'use_attention') and policy_net.use_attention))
if value_net is not None:
    self.has_hidden_states = self.has_hidden_states and (hasattr(value_net, 'use_lstm') and (value_net.use_lstm or (hasattr(value_net, 'use_attention') and value_net.use_attention)))
self.policy_hidden = None
self.value_hidden = None
```

**reset_hidden_states()の更新:**
```python
def reset_hidden_states(self) -> None:
    """Reset hidden states for LSTM/Attention networks at episode boundaries."""
    if self.has_hidden_states:
        self.policy_hidden = None
        self.value_hidden = None
        self._logger.debug("Hidden states reset for %s", self._get_player_id())
```

## 技術的改善点

### 1. 統一されたインターフェース

- ポリシーネットワーク: `agent.select_action()`
- 価値ネットワーク: `agent.get_value()`
- 両方ともRLAgent経由で隠れ状態を管理

### 2. デバイス自動管理

- テンソルのデバイス自動検出
- GPU/CPU間の自動転送
- バッチ次元の自動調整

### 3. エピソード境界処理

- 両ネットワークの隠れ状態を同期リセット
- エピソード開始時の適切な初期化
- 訓練ループでの自動呼び出し

## パフォーマンス改善

### 1. シーケンシャル学習の最適化

- 価値ネットワークも過去の状態を記憶
- より正確な状態価値推定
- 長期的な戦略学習の向上

### 2. メモリ効率化

- 不要なテンソル操作の削減
- バッチ処理の最適化
- 隠れ状態の適切な管理

## テスト結果

### 動作確認

```bash
python train_selfplay.py --config config/train_config.yml --episodes 1 --parallel 1
```

**結果:**
- LSTM価値ネットワークが正常に動作
- 隠れ状態が適切に管理される
- エピソード境界でのリセットが正常実行
- GPU加速が正常に機能

## 今後の展望

### 1. 追加の最適化

- 隠れ状態の保存・復元機能
- より高度なシーケンシャル学習アルゴリズム
- 動的な隠れ状態サイズ調整

### 2. 監視・分析機能

- 隠れ状態の可視化
- シーケンシャル学習の効果測定
- 価値推定精度の分析

## 関連ファイル

- `src/agents/RLAgent.py`: 主要な実装
- `train_selfplay.py`: 訓練ループの修正
- `CLAUDE.md`: ドキュメント更新
- `README.md`: 変更ログ追加

## 結論

価値ネットワークの隠れ状態管理統合により、LSTM/Attentionネットワークの完全なシーケンシャル学習が実現されました。この実装により、より高度で一貫性のあるポケモンバトル戦略の学習が可能となります。