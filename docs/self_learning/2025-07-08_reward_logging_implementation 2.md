# Mapleプロジェクト：報酬ログ出力システムの実装

*2025年7月8日*

## 概要

本日、Mapleプロジェクトの強化学習フレームワークにおいて、エピソード毎の詳細な報酬ログ出力機能を実装しました。この機能により、学習過程で各報酬コンポーネントがどのように働いているかを詳細に追跡できるようになります。

## 実装した機能

### 1. エピソード毎の報酬内訳ログ

従来は総報酬のみが表示されていましたが、新しいシステムでは各サブ報酬の詳細な内訳が表示されるようになりました。

#### 実装前の出力例：
```
Episode 1 reward 6.75 time/episode: 4.876 opponents: self
```

#### 実装後の出力例：
```
Episode 1 reward 16.83 time/episode: 3.355 opponents: self
Episode 1 reward breakdown: fail_immune: 0.000, knockout: 2.000, pokemon_count: 5.000, switch_penalty: 0.000, turn_penalty: -0.170, win_loss: 10.000
```

### 2. 勝敗報酬の追加

重要な改善として、勝敗報酬（win_loss）が報酬内訳に含まれるようになりました。これまで勝敗に関する +10.0/-10.0 の報酬は合計には含まれていましたが、内訳には表示されていませんでした。

## 技術的詳細

### ファイル変更箇所

#### 1. `train_selfplay.py`の修正

エピソード終了時のログ出力部分を拡張し、サブ報酬の詳細内訳を追加しました：

```python
# Calculate sub-reward totals
sub_totals = {}
for logs in sub_logs_list:
    for name, val in logs.items():
        sub_totals[name] = sub_totals.get(name, 0.0) + val

# Log total reward and sub-reward breakdown
logger.info(
    "Episode %d reward %.2f time/episode: %.3f opponents: %s",
    ep + 1,
    total_reward,
    duration,
    ", ".join(opponents_used),
)

# Log detailed reward breakdown if sub-rewards exist
if sub_totals:
    breakdown_parts = [f"{name}: {val:.3f}" for name, val in sorted(sub_totals.items())]
    logger.info("Episode %d reward breakdown: %s", ep + 1, ", ".join(breakdown_parts))
```

#### 2. `src/env/pokemon_env.py`の修正

`_calc_reward`メソッドを拡張し、勝敗報酬をサブ報酬ログに含めるように修正しました：

```python
if self.reward_type == "composite" and pid in self._composite_rewards:
    total = self._composite_rewards[pid].calc(battle)
    self._sub_reward_logs[pid] = dict(
        self._composite_rewards[pid].last_values
    )
    
    # Add win/loss reward to the breakdown
    win_reward = 0.0
    if getattr(battle, "finished", False):
        win_reward = 10.0 if getattr(battle, "won", False) else -10.0
        self._sub_reward_logs[pid]["win_loss"] = win_reward
    
    return float(total + win_reward)
```

## 報酬コンポーネントの詳細

現在のシステムで追跡される報酬コンポーネント：

### 1. **knockout** (撃破報酬)
- 相手ポケモンを撃破時：+1.0～+2.0
- 自分のポケモンが撃破時：-0.5

### 2. **pokemon_count** (残数差報酬)
- 対戦終了時の残りポケモン数の差に基づく：
  - 1匹差：0点
  - 2匹差：±2点
  - 3匹以上差：±5点

### 3. **turn_penalty** (ターンペナルティ)
- 毎ターン：-0.01～-0.02程度
- 長期戦を防ぐ効果

### 4. **fail_immune** (無効行動ペナルティ)
- 失敗・無効技使用時：-0.02（デフォルト）

### 5. **switch_penalty** (交代ペナルティ)
- 特定条件下での交代時のペナルティ

### 6. **win_loss** (勝敗報酬) ★新規追加
- 勝利時：+10.0
- 敗北時：-10.0
- エピソード進行中：0.0

## 実際の学習ログ例

```
Episode 3 reward breakdown: fail_immune: -0.140, knockout: -0.500, pokemon_count: 0.000, switch_penalty: 0.000, turn_penalty: -0.400, win_loss: -10.000
Episode 4 reward breakdown: fail_immune: -0.360, knockout: -0.500, pokemon_count: 0.000, switch_penalty: 0.000, turn_penalty: -0.580, win_loss: -10.000
Episode 5 reward breakdown: fail_immune: -0.340, knockout: 2.000, pokemon_count: 5.000, switch_penalty: 0.000, turn_penalty: -0.450, win_loss: 10.000
```

この例から以下のことが読み取れます：
- Episode 3, 4: 敗北（win_loss: -10.000）
- Episode 5: 勝利（win_loss: +10.000）で、大きなアドバンテージを獲得

## 開発効果

### 1. **学習状況の可視化**
各報酬コンポーネントの寄与度が明確になり、学習の進行状況を詳細に把握できます。

### 2. **デバッグの向上**
特定の報酬コンポーネントが期待通りに動作しているかを即座に確認できます。

### 3. **報酬設計の改善**
各コンポーネントの重みバランスが適切かを判断する材料が得られます。

## ログの保存と活用

全てのログは`logs/`ディレクトリ内のタイムスタンプ付きファイルに保存され、後の分析に活用できます。また、TensorBoardとの連携も維持されているため、グラフィカルな可視化も可能です。

## まとめ

今回の実装により、Mapleプロジェクトの強化学習システムはより透明性が高く、分析しやすいものになりました。特に勝敗報酬の内訳表示により、エージェントの学習における勝利への貢献度を正確に把握できるようになったことは大きな改善です。

この詳細なログ機能により、今後の報酬設計の最適化や学習アルゴリズムの改善において、より科学的なアプローチが可能になると期待されます。