# Model Evaluation Shape Mismatch Fix - Implementation Log

## 問題発生 (2025-07-12)

### 症状
```bash
python evaluate_rl.py --model model.pt --opponent max --n 1 --team random
```

実行時にshapeサイズのミスマッチエラーが発生：

```
RuntimeError: Error(s) in loading state_dict for LSTMPolicyNetwork:
	Missing key(s) in state_dict: "mlp.0.weight", "mlp.0.bias", ...
	Unexpected key(s) in state_dict: "input_proj.weight", "input_proj.bias", "output_mlp.0.weight", ...
	size mismatch for lstm.weight_ih_l0: copying a param with shape torch.Size([1024, 256]) from checkpoint, the shape in current model is torch.Size([512, 1136]).
```

## 原因分析

### 1. 保存されたモデルの構造分析
```python
# model.pt の構造確認
policy_keys = ['input_proj.weight', 'input_proj.bias', 'lstm.weight_ih_l0', 'lstm.weight_hh_l0', 
               'lstm.bias_ih_l0', 'lstm.bias_hh_l0', 'output_mlp.0.weight', ...]

# 重要な寸法情報
input_proj.weight: torch.Size([256, 1136])  # 入力: 1136 → 隠れ層: 256
lstm.weight_ih_l0: torch.Size([1024, 256])  # LSTM入力: 256, 隠れ層: 256
```

### 2. 評価スクリプトの問題点
```python
# evaluate_rl.py の問題のあったネットワーク検出ロジック
if any("lstm" in key for key in policy_keys):
    network_config = {
        "type": "lstm",           # ❌ 間違い: AttentionNetworkであるべき
        "hidden_size": 128,       # ❌ 間違い: 256であるべき
        "lstm_hidden_size": 128,  # ❌ 間違い: 256であるべき
        "use_lstm": True,
        "use_2layer": True
    }
```

### 3. ネットワーク構造の不一致
- **保存されたモデル**: `AttentionPolicyNetwork` (input_proj + output_mlp 構造)
- **評価スクリプト**: `LSTMPolicyNetwork` (直接LSTM + mlp 構造)

## 解決策の実装

### 1. ネットワーク検出ロジックの修正
```python
# 修正後のネットワーク検出
if any("input_proj" in key for key in policy_keys):
    # AttentionNetworkを正しく識別
    has_attention_layers = any("attention" in key for key in policy_keys)
    network_config = {
        "type": "attention",                    # ✅ 正しいネットワークタイプ
        "hidden_size": 256,                     # ✅ 正しいサイズ
        "use_attention": has_attention_layers,  # ✅ 動的検出
        "use_lstm": any("lstm" in key for key in policy_keys),
        "use_2layer": True
    }
```

### 2. 両方の検出パスを修正
- **新形式検出** (`state_dict["policy"]` 形式)
- **直接形式検出** (直接state_dict形式)

両方で同様の修正を適用

### 3. 状態空間サイズの確認
```python
# 現在の状態空間サイズ確認
state_observer = StateObserver('config/state_spec.yml')
print('Current state dimension:', state_observer.get_observation_dimension())
# 出力: 1136 (保存されたモデルの入力サイズと一致)
```

## 修正結果

### 成功ログ
```
Detected Attention network (with LSTM) from state_dict structure
Using network config: {'type': 'attention', 'hidden_size': 256, 'use_attention': False, 'use_lstm': True, 'use_2layer': True}
Policy network: {'type': 'AttentionPolicyNetwork', 'total_params': 1083147, 'trainable_params': 1083147, 'has_lstm': True}
Value network: {'type': 'AttentionValueNetwork', 'total_params': 1080577, 'trainable_params': 1080577, 'has_lstm': True}
```

### バトル開始確認
```
Hidden states reset for player_0
Selecting team for player_0...
Selecting team for player_1...
2025-07-12 20:39:21,682 - model_36083776 - INFO - Starting listening to showdown websocket
```

## 技術的詳細

### AttentionPolicyNetwork vs LSTMPolicyNetwork の違い

| 特徴 | AttentionPolicyNetwork | LSTMPolicyNetwork |
|------|----------------------|-------------------|
| 入力処理 | `input_proj` (1136→256) | 直接LSTM (1136→隠れ層) |
| 出力処理 | `output_mlp` | `mlp` |
| 構造 | input_proj → LSTM → output_mlp | obs → LSTM → mlp |
| 柔軟性 | attention有無を選択可 | LSTM専用 |

### ネットワーク作成フロー
```python
# network_factory.py における作成ロジック
if network_type == "attention":
    return AttentionPolicyNetwork(
        observation_space=observation_space,
        action_space=action_space,
        hidden_size=hidden_size,           # 256
        num_heads=attention_heads,
        use_attention=use_attention,       # False (この場合)
        use_lstm=use_lstm,                # True
        use_2layer=use_2layer
    )
```

## 学習した教訓

1. **モデル保存時の構成記録**: モデル保存時にネットワーク構成も保存すべき
2. **動的検出の重要性**: ハードコードではなく、保存されたstate_dictから構成を推定
3. **レイヤー名による識別**: `input_proj`の存在でAttentionNetworkを確実に識別
4. **寸法チェック**: 事前に状態空間サイズとモデル期待サイズを確認

## 予防策

1. **設定保存**: 将来的にはモデルと一緒にnetwork_configを保存
2. **検証機能**: モデル読み込み前の構成検証
3. **明確なエラーメッセージ**: より詳細なミスマッチ情報の提供

この修正により、モデル評価が正常に動作し、ネットワーク構成の自動検出が大幅に改善されました。