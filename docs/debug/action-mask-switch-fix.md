# Action Mask and Switch Command Debug Log

## 概要
Pokemon Showdownでの交代コマンドでエラーが発生する問題のデバッグと修正記録。

**Issue**: Dittoがアクティブな時のaction maskエラーと、交代コマンドの位置番号不正問題

## エラーパターンの発見

### 1. 初期エラー: Dittoの変身による交代エラー
```
|error|[Invalid choice] Can't switch: You can't switch to an active Pokémon
```

**症状**:
- Dittoがアクティブ時にのみ発生
- Dittoの変身により、species名が相手ポケモンと同じになる
- Action maskが正しく計算されていない

**原因の仮説**:
Dittoの変身後、species名による識別で混乱が発生

### 2. Path環境問題の発見と修正
```
copy_of_poke-env/poke_env/ps_client/ps_client.py:225
```

**問題**: プロジェクト内の`copy_of_poke-env`が使用されていた
**修正**: 全ファイルから`copy_of_poke-env`への参照を削除し、`.venv`内のpoke-envを使用

**修正ファイル**:
- `train.py`
- `evaluate_rl.py` 
- `train_rl.py`
- `test/run_battle.py`

## 根本原因の特定

### 3. Species名による交代コマンドの問題
**仮説**: Pokemon Showdownでspecies名を使った交代が、Dittoの変身時に問題を起こす

**解決案**: 位置番号ベースの交代コマンド使用
- `/choose switch {species}` → `/choose switch {position}`

### 4. Position-based Switch実装

**実装したカスタムBattleOrder**:
```python
class PositionalSwitchOrder(BattleOrder):
    def __init__(self, position: int):
        self.position = position
    
    @property
    def message(self) -> str:
        return f"/choose switch {self.position}"
```

**修正箇所**: `src/action/action_helper.py`の`action_index_to_order_from_mapping`関数

### 5. 新たな問題: 位置番号の不正

```
|error|[Invalid choice] Can't switch: You can't switch to a fainted Pokémon
|error|[Invalid choice] Can't switch: You do not have a Pokémon in slot 4 to switch to
```

**原因分析**:
1. **気絶ポケモンの除外不足**: action maskが気絶ポケモンを正しく除外していない
2. **位置番号の不整合**: チーム全体(6匹)の位置を使用しているが、選出は3匹のみ

## Action Mask修正

### 6. 気絶ポケモンのフィルタリング追加
```python
# Filter out fainted Pokemon from switches
switches = [p for p in switches if not getattr(p, 'fainted', False)]
```

### 7. Action Mask Mapping修正

**問題**: 全ポケモンが`active=True`として誤判定される
```
[MAPPING DEBUG] Team idx 0: terapagos, active=True, fainted=False, can_switch=False
```

**原因**: `getattr(active_pokemon, '_ident', '')` が `'?'` を返し、全ポケモンの`_ident`も `'?'` のため

**修正**: `battle.available_switches`を権威的ソースとして使用
```python
is_in_available_switches = any(
    getattr(sw, 'species', '') == getattr(team_pokemon, 'species', '') and
    getattr(sw, '_ident', '') == getattr(team_pokemon, '_ident', '')
    for sw in switches
)
can_switch = is_in_available_switches and not is_fainted
```

## 位置番号システムの理解

### 8. Pokemon Showdownの位置番号仕様

**Team Preview時の構成**:
```
EnvPlayer 1 (6匹フル):
1. Calyrex-Shadow
2. Grimmsnarl  
3. Chien-Pao
4. Eternatus
5. Gliscor
6. Dondozo
```

**Battle時の選出** (3匹):
```
1. Chien-Pao (active)
2. Eternatus
3. Gliscor
```

**問題**: `battle.team`は6匹フルチームの順序を保持しているが、Pokemon Showdownの交代は選出された3匹の順序を使用

### 9. 最終修正: Request Message活用

**解決策**: `battle._last_request['side']['pokemon']`から選出順序を取得

```python
request = getattr(battle, '_last_request', None)
selected_team = request['side']['pokemon']

for i, selected_mon in enumerate(selected_team):
    if selected_mon['ident'] == pokemon._ident:
        team_position = i + 1  # 1-based indexing
        break
```

### 10. Pokemon属性アクセスエラーの修正

**エラー**: `AttributeError: 'Pokemon' object has no attribute '_ident'`

**原因**: poke-envの`Pokemon`オブジェクトは`_ident`属性を持たない
- 正しい属性: `pokemon.name` (例: "Chien-Pao")
- Request messageのformat: `"p1: Chien-Pao"`

**修正内容**:
```python
# 修正前
pokemon_ident = pokemon._ident  # AttributeError

# 修正後
player_role = getattr(battle, '_player_role', None)  # "p1" or "p2"
pokemon_full_ident = f"{player_role}: {pokemon.name}"  # "p1: Chien-Pao"
```

**poke-env Pokemon属性の理解**:
- `pokemon.name`: ポケモン名のみ (例: "Chien-Pao")
- `pokemon.identifier(player_role)`: 完全な識別子 (例: "p1: Chien-Pao")
- `pokemon.species`: 種族名 (例: "chienpao")
- `battle._player_role`: プレイヤーのロール ("p1" or "p2")

## 現在のステータス

### 実装済み修正
1. ✅ copy_of_poke-env参照の削除
2. ✅ Position-basedスイッチコマンド実装
3. ✅ 気絶ポケモンのフィルタリング
4. ✅ Action mask mapping修正
5. ✅ Request messageベースの位置計算
6. ✅ 厳密なエラーハンドリング（フォールバック削除）

### デバッグ機能
- **詳細ログ**: Action mask計算過程の可視化
- **Switch debug**: `"プレイヤー名: choose switch to ポケモン名 (position N in selected team)"`
- **エラー詳細**: Request data、選出チーム、バトルチームの状態表示

### 次回テスト確認項目
1. 選出された3匹の正しい位置での交代実行
2. Dittoの変身時でも正常動作
3. Force switch状況での正常動作

## 技術的学習事項

1. **Pokemon Showdown プロトコル**:
   - Position番号は1-based indexing
   - 交代は選出された3匹の順序に基づく
   - Species名ではなく位置番号が確実

2. **poke-env ライブラリ**:
   - `battle.team`: 6匹フルチーム (Team Preview順序)
   - `battle.available_switches`: 交代可能ポケモン（気絶・アクティブ除く）
   - `battle._last_request`: 最新のrequest messageデータ

3. **Action Mask システム**:
   - Mapping段階とExecution段階での一貫性が重要
   - 権威的データソース（available_switches）の活用
   - エラー時の詳細情報出力の重要性