# FailAndImmuneReward実装ドキュメント

## 概要
FailAndImmuneRewardシステムは、AIが無効なアクション（失敗、無効）を行った際にペナルティを課すことで、より効果的な戦略学習を促進するためのシステムです。

## ✅ 実装完了状況 (2025-07-16)

### 1. FailAndImmuneReward クラス実装
- **実装場所**: `src/rewards/fail_and_immune.py`
- **機能**: `battle.last_fail_action` と `battle.last_immune_action` を監視
- **ペナルティ**: デフォルト -0.3 (configurable)

### 2. CompositeReward統合
- **実装場所**: `src/rewards/composite.py`
- **機能**: `config/reward.yaml` からペナルティ値を設定可能
- **設定例**:
  ```yaml
  fail_immune:
    weight: 2.0
    enabled: true
    penalty: -0.3
  ```

### 3. CustomBattle実装 (メイン解決策)
- **実装場所**: `src/env/custom_battle.py`
- **機能**: 標準poke-envを拡張し、`-fail`と`-immune`メッセージを処理
- **特徴**:
  - `last_fail_action`: プレイヤーのアクションが失敗した際にTrue
  - `last_immune_action`: 相手ポケモンがプレイヤーの攻撃に免疫の際にTrue
  - ターン開始時に自動リセット

### 4. EnvPlayer統合
- **実装場所**: `src/env/env_player.py`
- **機能**: `_create_battle()` メソッドでCustomBattleを使用
- **効果**: 全てのバトルで自動的にfail/immuneトラッキングが有効

### 5. 完全なテスト実装
- **実装場所**: `tests/test_custom_battle.py`
- **テスト数**: 14のテストケース
- **カバレッジ**: メッセージ処理、フラグ管理、報酬統合

## 修正された問題 (2025-07-16)

### 問題: fail_immune報酬がカウントされない
**根本原因**: 
- 標準poke-envライブラリに`last_fail_action`と`last_immune_action`属性が存在しない
- `-fail`と`-immune`メッセージが`MESSAGES_TO_IGNORE`に含まれ処理されない

**解決策**:
1. **CustomBattleクラス**: 標準poke-envを拡張してメッセージ処理を追加
2. **メッセージインターセプト**: 親クラス処理前に`-fail`と`-immune`メッセージを検出
3. **プレイヤー識別**: `_player_role`を使用してプレイヤーのアクションのみを対象

### 技術的詳細
```python
# CustomBattle.parse_message()での処理
if split_message[1] == "-fail":
    if pokemon_ident.startswith(f"p{self._player_role}"):
        self._last_fail_action = True

if split_message[1] == "-immune":
    opponent_role = "1" if self._player_role == "2" else "2"
    if pokemon_ident.startswith(f"p{opponent_role}"):
        self._last_immune_action = True
```

## 使用方法

### 設定
```yaml
# config/reward.yaml
rewards:
  fail_immune:
    weight: 2.0      # 報酬の重み
    enabled: true    # 有効化
    penalty: -0.3    # ペナルティ値
```

### 効果
- 失敗アクション: -0.3 × 2.0 = -0.6の報酬ペナルティ
- 無効アクション: -0.3 × 2.0 = -0.6の報酬ペナルティ
- 正常アクション: 0の報酬影響

## パフォーマンス
- **メッセージ処理**: 追加のオーバーヘッドは最小限
- **メモリ使用**: 2つのbooleanフラグのみ追加
- **CPU使用**: 文字列比較のみ、計算負荷は無視可能

## 今後の拡張予定
- 異なるタイプの失敗に対する個別ペナルティ
- 連続失敗に対する増加ペナルティ
- 成功率に基づく動的ペナルティ調整
